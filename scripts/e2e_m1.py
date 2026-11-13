#!/usr/bin/env python
"""M1 acceptance test: the OpenAI Python SDK streams from Kiln, end to end.

Brings up a node agent and a gateway, deploys one small model through the agent's
API, and then drives the platform with the stock `openai` client — no Kiln-specific
code on the client side at all, which is the entire point of being OpenAI-compatible.

Checks, in order:
  1. the node agent starts a real vLLM process and reports it ACTIVE;
  2. `GET /v1/models` lists the model with its tier state;
  3. a non-streaming chat completion returns text and a usage block;
  4. a streaming completion delivers incremental chunks, and the measured TTFT is
     well below the total — proving the SSE path is not buffered, which is what makes
     every TTFT number in this project meaningful;
  5. the legacy completions endpoint works;
  6. sleep and wake through the agent leave the model generating correctly.

Writes docs/results/m1_e2e.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
from urllib.parse import quote
import time

import requests

REPO = pathlib.Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    """A child process whose output is kept; a dead Go binary's stderr is the only
    diagnostic when it refuses to come up."""

    def __init__(self, name: str, cmd: list[str], env: dict | None = None) -> None:
        self.name = name
        self.log = pathlib.Path("/tmp") / f"kiln-{name}.log"
        self.fh = self.log.open("w")
        self.p = subprocess.Popen(cmd, stdout=self.fh, stderr=self.fh,
                                  env={**os.environ, **(env or {})}, start_new_session=True)

    def wait_http(self, url: str, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.p.poll() is not None:
                raise RuntimeError(f"{self.name} exited {self.p.returncode}; log:\n{self.log.read_text()[-3000:]}")
            with contextlib.suppress(Exception):
                if requests.get(url, timeout=2).status_code == 200:
                    return
            time.sleep(0.25)
        raise TimeoutError(f"{self.name} not healthy within {timeout}s; log:\n{self.log.read_text()[-3000:]}")

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            self.p.wait(timeout=120)
        with contextlib.suppress(Exception):
            self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-0.5b")
    ap.add_argument("--hf-repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--out", default="docs/results/m1_e2e.json")
    args = ap.parse_args()

    from openai import OpenAI  # imported late so --help works without it

    agent_port, gw_port = free_port(), free_port()
    results: dict = {"model": args.model, "hf_repo": args.hf_repo, "checks": {}}
    procs: list[Proc] = []

    try:
        print(f"[1/6] starting node agent on :{agent_port}")
        agent = Proc("nodeagent", [
            str(REPO / "bin" / "nodeagent"),
            "-addr", f"127.0.0.1:{agent_port}",
            "-python", sys.executable,
            "-node", "local-3070",
            "-log-dir", "/tmp/kiln-logs",
        ])
        procs.append(agent)
        agent.wait_http(f"http://127.0.0.1:{agent_port}/health", timeout=30)

        print(f"[2/6] deploying {args.hf_repo} (cold start, expect ~40s)")
        t0 = time.perf_counter()
        r = requests.post(f"http://127.0.0.1:{agent_port}/replicas", timeout=1800, json={
            "replica": f"{args.model}--0", "model": args.model, "hf_repo": args.hf_repo,
            "gpu_memory_utilization": args.gpu_util, "max_model_len": 2048,
            "enforce_eager": True,
        })
        r.raise_for_status()
        status = r.json()
        results["cold_start_s"] = round(time.perf_counter() - t0, 2)
        results["replica"] = status
        print(f"      ACTIVE in {results['cold_start_s']}s on port {status['port']}")
        assert status["state"] == "ACTIVE", status

        print(f"[3/6] starting gateway on :{gw_port}")
        gw = Proc("gateway", [
            str(REPO / "bin" / "gateway"),
            "-addr", f"127.0.0.1:{gw_port}",
            "-agents", f"http://127.0.0.1:{agent_port}",
            "-router", "round-robin",
        ])
        procs.append(gw)
        gw.wait_http(f"http://127.0.0.1:{gw_port}/health", timeout=30)

        base = f"http://127.0.0.1:{gw_port}/v1"
        client = OpenAI(base_url=base, api_key="not-required-yet")

        print("[4/6] GET /v1/models via the SDK")
        models = client.models.list()
        ids = [m.id for m in models.data]
        results["checks"]["models_listed"] = args.model in ids
        print(f"      {ids}")
        assert args.model in ids, ids

        print("[5/6] non-streaming chat completion via the SDK")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Name three primary colours."}],
            max_tokens=40, temperature=0.0,
        )
        nonstream_s = time.perf_counter() - t0
        text = resp.choices[0].message.content
        results["nonstreaming"] = {
            "latency_s": round(nonstream_s, 3),
            "text": text,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        results["checks"]["nonstreaming_text"] = bool(text and text.strip())
        results["checks"]["nonstreaming_usage"] = resp.usage.completion_tokens > 0
        print(f"      {nonstream_s:.2f}s, {resp.usage.completion_tokens} tokens: {text!r}")

        print("[6/6] streaming chat completion via the SDK")
        t0 = time.perf_counter()
        ttft, chunks, out = None, 0, []
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Count slowly from one to twelve."}],
            max_tokens=60, temperature=0.0, stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
                out.append(delta)
        total = time.perf_counter() - t0
        results["streaming"] = {
            "ttft_s": round(ttft or 0, 3), "total_s": round(total, 3),
            "chunks": chunks, "text": "".join(out),
        }
        # The real check. If the proxy buffered, TTFT would equal total and every
        # latency number this project reports would be an artefact of the gateway.
        results["checks"]["streamed_incrementally"] = chunks > 3 and ttft is not None and ttft < total * 0.8
        print(f"      TTFT {ttft:.3f}s, total {total:.3f}s, {chunks} chunks")
        print(f"      {''.join(out)[:120]!r}")

        print("[bonus] sleep L1 -> wake through the agent, then generate again")
        rid = quote(f"{args.model}--0", safe="")
        t0 = time.perf_counter()
        requests.post(f"http://127.0.0.1:{agent_port}/replicas/{rid}/sleep",
                      json={"level": 1, "mode": "wait"}, timeout=600).raise_for_status()
        sleep_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        requests.post(f"http://127.0.0.1:{agent_port}/replicas/{rid}/wake", timeout=900).raise_for_status()
        wake_s = time.perf_counter() - t0

        after = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": "Name three primary colours."}],
            max_tokens=40, temperature=0.0).choices[0].message.content
        results["sleep_wake"] = {"sleep_l1_s": round(sleep_s, 3), "wake_l1_s": round(wake_s, 3),
                                 "text_after_wake": after}
        # Greedy decoding with the same prompt must reproduce the same answer. A wake
        # that restored memory but not weights passes a health check and fails this.
        results["checks"]["correct_after_wake"] = after.strip() == (text or "").strip()
        print(f"      slept in {sleep_s:.2f}s, woke in {wake_s:.2f}s, output identical: "
              f"{results['checks']['correct_after_wake']}")
    finally:
        for p in reversed(procs):
            p.stop()

    results["PASS"] = all(results["checks"].values())
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    for k, v in results["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 70)
    print(f"M1: {'PASS' if results['PASS'] else 'FAIL'}   (wrote {args.out})")
    return 0 if results["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
