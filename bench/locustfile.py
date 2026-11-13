"""Locust load shapes for Kiln.

Two user classes, because the experiments ask two different questions:

  TraceUser   replays a recorded trace from bench/traces at its original arrival
              times. Every policy and router arm sees the identical request sequence,
              which is what makes the comparisons mean anything.

              Select it by name on the command line. Locust runs *every* HttpUser
              subclass in a file unless told which to use, so omitting the name lets
              SteadyUser generate traffic alongside the replay.

  SteadyUser  generates open-loop traffic for throughput and saturation runs where a
              fixed trace would just cap the offered load.

Both measure time-to-first-token separately from total latency. TTFT is the number
the whole project turns on: a wake that takes 600ms is visible in TTFT and almost
invisible in total latency for a long generation, so reporting only end-to-end
latency would hide exactly the effect being studied.

Usage:
    locust -f bench/locustfile.py --host http://127.0.0.1:8080 \
           --headless -u 16 -r 4 -t 5m \
           --trace bench/traces/zipf_s10.jsonl --api-key sk-kiln-...
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import random
import threading
import time

from locust import HttpUser, between, events, task


# ------------------------------------------------------------------ CLI options

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--trace", default="", help="trace .jsonl to replay")
    parser.add_argument("--api-key", default=os.environ.get("KILN_API_KEY", ""),
                        help="Bearer token for the gateway")
    parser.add_argument("--models", default="", help="comma-separated models for SteadyUser")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="override the trace's max_tokens; 0 keeps it")
    parser.add_argument("--stream", default="true", help="use SSE streaming")
    parser.add_argument("--results", default="", help="write a per-request CSV here")


# Shared across users: a trace is replayed once between all of them, not once each.
_TRACE: list[dict] = []
_TRACE_META: dict = {}
_cursor = itertools.count()
_results_lock = threading.Lock()
_results_file = None
_start_wall = 0.0


@events.test_start.add_listener
def on_start(environment, **_):
    global _TRACE, _TRACE_META, _results_file, _start_wall
    opts = environment.parsed_options
    _start_wall = time.perf_counter()

    if opts.trace:
        path = pathlib.Path(opts.trace)
        lines = path.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        if "_meta" in first:
            _TRACE_META = first["_meta"]
            lines = lines[1:]
        _TRACE = [json.loads(ln) for ln in lines if ln.strip()]
        print(f"[kiln] replaying {len(_TRACE)} requests from {path} (meta: {_TRACE_META})")

    if opts.results:
        p = pathlib.Path(opts.results)
        p.parent.mkdir(parents=True, exist_ok=True)
        _results_file = p.open("w", encoding="utf-8")
        # Per-request rows, not just Locust's aggregates. The experiments need
        # percentiles computed over a chosen window and grouped by model, which
        # aggregate-only output cannot give.
        _results_file.write("t_wall,model,team,status,ttft_ms,total_ms,"
                            "prompt_tokens,completion_tokens,replica,session,turn\n")


@events.test_stop.add_listener
def on_stop(**_):
    if _results_file:
        _results_file.flush()
        _results_file.close()


def record(row: dict) -> None:
    if not _results_file:
        return
    with _results_lock:
        _results_file.write(
            f"{row['t_wall']:.4f},{row['model']},{row['team']},{row['status']},"
            f"{row['ttft_ms']:.2f},{row['total_ms']:.2f},"
            f"{row['prompt_tokens']},{row['completion_tokens']},"
            f"{row['replica']},{row['session']},{row['turn']}\n")


# ------------------------------------------------------------------ the request

def chat(client, api_key: str, model: str, prompt: str, max_tokens: int,
         stream: bool, team: str, session: str, turn: int):
    """Issue one chat completion and record TTFT separately from total latency."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # Greedy. Sampling would add variance to output length, and output length
        # drives total latency, so temperature > 0 would inject noise into the very
        # metric being compared across arms.
        "temperature": 0.0,
        "stream": stream,
    }

    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = completion_tokens = 0
    replica = ""

    # catch_response so a 429 can be reported as a distinct outcome rather than a
    # generic failure: under quota tests a 429 is the correct behaviour, and marking
    # it as an error would make the run look broken.
    with client.post("/v1/chat/completions", json=body, headers=headers,
                     stream=stream, catch_response=True,
                     name=f"chat:{model}") as resp:
        replica = resp.headers.get("X-Kiln-Replica", "")
        status = resp.status_code

        if status != 200:
            total = (time.perf_counter() - t0) * 1000
            if status == 429:
                resp.success()  # expected under a quota test
            else:
                resp.failure(f"HTTP {status}")
            return dict(status=status, ttft_ms=0.0, total_ms=total,
                        prompt_tokens=0, completion_tokens=0, replica=replica)

        if stream:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # TTFT is the first chunk carrying actual content, not the first
                # frame. vLLM emits a role-only delta first, and counting it would
                # under-report TTFT by the whole first decode step.
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
        else:
            data = resp.json()
            ttft = (time.perf_counter() - t0) * 1000
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        resp.success()

    total = (time.perf_counter() - t0) * 1000
    return dict(status=200, ttft_ms=ttft or total, total_ms=total,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                replica=replica)


# ------------------------------------------------------------------ user classes

class TraceUser(HttpUser):
    """Replays the shared trace, honouring each request's original arrival time."""

    # No think time: pacing comes from the trace's own arrival times.
    wait_time = between(0, 0)

    @task
    def replay(self):
        opts = self.environment.parsed_options
        if not _TRACE:
            self.environment.runner.quit()
            return

        i = next(_cursor)
        if i >= len(_TRACE):
            # The trace is exhausted. Stop rather than looping: a second pass would
            # start with every model already resident, so the warm-up cost would be
            # counted once and amortised over two passes, quietly flattering whatever
            # policy is under test.
            self.environment.runner.quit()
            return

        req = _TRACE[i]
        # Replay at the recorded arrival time so bursts stay bursts. Without this the
        # trace degenerates into a closed-loop stream whose rate is set by the model's
        # own latency, which is precisely the variable under test.
        due = _start_wall + req["t"]
        delay = due - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

        # The CLI wins over the trace: output length is a load knob, and tying it to
        # the trace would mean regenerating every trace to change the load, which
        # would break comparability with runs already taken.
        max_tokens = opts.max_tokens or req.get("max_tokens", 64)
        res = chat(self.client, opts.api_key, req["model"], req["prompt"],
                   max_tokens,
                   opts.stream.lower() == "true", req.get("team", ""),
                   req.get("session", ""), req.get("turn", 0))
        record({**res, "t_wall": time.perf_counter() - _start_wall,
                "model": req["model"], "team": req.get("team", ""),
                "session": req.get("session", ""), "turn": req.get("turn", 0)})


class SteadyUser(HttpUser):
    """Open-loop traffic for throughput and saturation runs."""

    wait_time = between(0.5, 2.0)

    def on_start(self):
        opts = self.environment.parsed_options
        self.models = [m for m in opts.models.split(",") if m]
        if not self.models:
            # No silent default. A hard-coded model id here once ran alongside the
            # trace replay and sent nine requests for a non-existent model for every
            # real one, which dominated every figure the experiment produced.
            raise RuntimeError(
                "SteadyUser needs --models; it is for open-loop runs, and the "
                "trace-replay experiments should select TraceUser explicitly")
        self.rng = random.Random(id(self))

    @task
    def send(self):
        opts = self.environment.parsed_options
        model = self.rng.choice(self.models)
        prompt = self.rng.choice([
            "Explain how a write-ahead log works.",
            "What is the difference between a mutex and a semaphore?",
            "Summarise the CAP theorem in three sentences.",
            "Why would a p99 latency be much worse than a p50?",
        ])
        res = chat(self.client, opts.api_key, model, prompt, opts.max_tokens,
                   opts.stream.lower() == "true", "", "", 0)
        record({**res, "t_wall": time.perf_counter() - _start_wall,
                "model": model, "team": "", "session": "", "turn": 0})
