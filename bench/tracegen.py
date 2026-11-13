#!/usr/bin/env python
"""Generate the fixed request traces every Kiln experiment replays.

Traces are generated once, written to disk, and replayed identically by every policy
and router arm. That is the whole reason this file exists: if each arm generated its
own traffic, a difference in measured p95 could be a difference in the policy or a
difference in the random draw, and there would be no way to tell which. With a fixed
trace, one variable changes per comparison.

Four workloads, each targeting a specific claim:

  steady   - uniform load across a few models. The baseline: no memory pressure, so
             any difference between policies here is noise and a useful sanity check.

  zipf     - many models, a few hot, with skew s in {0.6, 1.0, 1.4}. This is the
             workload tiering exists for. Low skew means no working set fits and
             everything thrashes; high skew means the hot set fits in VRAM and
             eviction barely matters. The interesting results live in between, which
             is why three skews are generated rather than one.

  bursty   - popularity shifts abruptly between phases. Punishes policies that
             extrapolate from the distant past (plain LFU) and rewards decay. It is
             also the only workload where predictive prewarming can win, because it
             is the only one with a rising edge to predict.

  agent    - multi-turn conversations and tool-calling loops that resend the same
             system prompt and tool definitions every turn. Most of each prompt is a
             shared prefix, which is what prefix-aware routing is for.

Every trace records absolute arrival times, so replay reproduces the original burst
structure rather than a uniform stream of the same requests.

Usage:
    python bench/tracegen.py --out bench/traces
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from dataclasses import asdict, dataclass

# The local model fleet. Sizes are real vLLM footprints at
# gpu_memory_utilization=0.35 on an 8GB card, so the "how many fit" arithmetic in the
# capacity experiment reflects the actual device rather than parameter counts.
LOCAL_MODELS = [
    ("qwen-0.5b", 2867),
    ("qwen3-0.6b", 2900),
    ("llama-1b", 3400),
    ("qwen-1.5b", 4100),
    ("smollm2-1.7b", 4300),
]

SYSTEM_PROMPTS = [
    # Long enough to span several 64-character prefix blocks, which is what makes the
    # shared-prefix effect measurable at all.
    "You are a meticulous research assistant. Always cite your sources, think step by "
    "step, and say explicitly when you are uncertain rather than guessing. Prefer "
    "primary sources over summaries. When asked for code, give complete runnable "
    "examples with no placeholders.",
    "You are a senior backend engineer reviewing code for correctness and clarity. "
    "Point out concurrency hazards, unchecked errors and resource leaks first. Do not "
    "comment on formatting. Suggest the smallest change that fixes the problem.",
    "You are a patient tutor. Explain concepts from first principles, check "
    "understanding with a short question after each step, and never skip the algebra. "
    "Use concrete numbers before general formulas.",
]

TOOL_DEFINITIONS = json.dumps([
    {"name": "search_web", "description": "Search the public web for a query string.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "read_file", "description": "Read a file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "run_sql", "description": "Execute a read-only SQL query.",
     "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}},
], indent=None)

USER_TURNS = [
    "Summarise the tradeoffs in the design I just described.",
    "What breaks first if traffic increases tenfold?",
    "Write the migration for that schema change.",
    "Why is the p99 so much worse than the p50 here?",
    "Give me three alternatives and the strongest objection to each.",
    "Walk me through what happens on a cache miss.",
    "Which of these would you cut to ship a week sooner?",
]


@dataclass
class Request:
    """One request in a trace."""
    t: float          # arrival time in seconds from the start of the trace
    model: str
    team: str
    prompt: str
    max_tokens: int
    # session ties the turns of one conversation together, so the replayer can keep a
    # conversation's turns in order and the prefix router has something to learn.
    session: str
    turn: int


def zipf_weights(n: int, s: float) -> list[float]:
    """Normalised Zipf weights for n items at skew s."""
    raw = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(raw)
    return [r / total for r in raw]


def poisson_arrivals(rng: random.Random, rate: float, duration: float) -> list[float]:
    """Arrival times from a Poisson process.

    Exponential inter-arrival gaps rather than a fixed cadence: real traffic clumps,
    and a perfectly even stream never produces the transient queue depth that makes
    the difference between policies visible.
    """
    out, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return out
        out.append(t)


def gen_steady(rng: random.Random, models: list[str], duration: float,
               rate: float, teams: list[str]) -> list[Request]:
    reqs = []
    for i, t in enumerate(poisson_arrivals(rng, rate, duration)):
        reqs.append(Request(
            t=t, model=models[i % len(models)], team=teams[i % len(teams)],
            prompt=rng.choice(USER_TURNS), max_tokens=64,
            session=f"steady-{i}", turn=0))
    return reqs


def gen_zipf(rng: random.Random, models: list[str], duration: float,
             rate: float, skew: float, teams: list[str]) -> list[Request]:
    weights = zipf_weights(len(models), skew)
    reqs = []
    for i, t in enumerate(poisson_arrivals(rng, rate, duration)):
        model = rng.choices(models, weights=weights, k=1)[0]
        reqs.append(Request(
            t=t, model=model, team=teams[i % len(teams)],
            prompt=rng.choice(USER_TURNS), max_tokens=64,
            session=f"zipf-{i}", turn=0))
    return reqs


def gen_bursty(rng: random.Random, models: list[str], duration: float,
               rate: float, teams: list[str], phases: int = 4) -> list[Request]:
    """Popularity shifts between phases.

    Each phase promotes a different model to dominance abruptly. The transition is a
    step, not a ramp, because a step is the hardest case: a decayed-rate policy must
    forget the old favourite quickly without forgetting so fast that it thrashes.
    """
    phase_len = duration / phases
    reqs = []
    for i, t in enumerate(poisson_arrivals(rng, rate, duration)):
        phase = min(int(t / phase_len), phases - 1)
        hot = models[phase % len(models)]
        # 70/30: the hot model dominates but the tail never disappears, so a policy
        # cannot win by simply pinning one model forever.
        model = hot if rng.random() < 0.7 else rng.choice(models)
        reqs.append(Request(
            t=t, model=model, team=teams[i % len(teams)],
            prompt=rng.choice(USER_TURNS), max_tokens=64,
            session=f"burst-{i}", turn=0))
    return reqs


def gen_agent(rng: random.Random, models: list[str], duration: float,
              sessions: int, teams: list[str]) -> list[Request]:
    """Multi-turn agent traffic with a large shared prefix.

    Each session picks one system prompt and the same tool definitions, then grows its
    conversation turn by turn. Every turn resends the whole history, so consecutive
    turns of a session share a long exact prefix and different sessions on the same
    system prompt share a shorter one. Both are real effects in agent workloads and
    both are what prefix-aware routing exploits.
    """
    reqs = []
    for s in range(sessions):
        model = rng.choice(models)
        system = rng.choice(SYSTEM_PROMPTS)
        start = rng.uniform(0, duration * 0.6)
        history = f"{system}\n\nAvailable tools:\n{TOOL_DEFINITIONS}\n\n"
        t = start
        n_turns = rng.randint(3, 8)
        for turn in range(n_turns):
            user = rng.choice(USER_TURNS)
            history += f"User: {user}\n"
            reqs.append(Request(
                t=t, model=model, team=teams[s % len(teams)],
                prompt=history, max_tokens=48,
                session=f"agent-{s}", turn=turn))
            # Think time between turns. Without a gap, a session's turns arrive
            # together and the routing comparison degenerates into one batch.
            t += rng.uniform(1.5, 6.0)
            history += "Assistant: (response elided for the trace)\n"
            if t > duration:
                break
    reqs.sort(key=lambda r: r.t)
    return reqs


def write(path: pathlib.Path, name: str, reqs: list[Request], meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    f = path / f"{name}.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        # The first line is the trace's own metadata, so a results file can always be
        # traced back to the exact trace and seed that produced it.
        fh.write(json.dumps({"_meta": meta}) + "\n")
        for r in reqs:
            fh.write(json.dumps(asdict(r)) + "\n")

    models = {}
    for r in reqs:
        models[r.model] = models.get(r.model, 0) + 1
    top = sorted(models.items(), key=lambda kv: -kv[1])
    share = ", ".join(f"{m}={n}" for m, n in top[:4])
    print(f"  {name:22} {len(reqs):6d} reqs over {meta['duration']:.0f}s  [{share}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bench/traces")
    ap.add_argument("--duration", type=float, default=600.0, help="trace length in seconds")
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--teams", default="team-a,team-b,team-c")
    ap.add_argument("--models", default="",
                    help="comma-separated model ids; defaults to the local fleet. Must match "
                         "the models the experiment deploys, or part of the trace addresses "
                         "models that do not exist and silently measures nothing.")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    teams = args.teams.split(",")
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [m for m, _ in LOCAL_MODELS]

    print(f"generating traces into {out} (seed {args.seed})")

    # One seeded RNG per trace, derived from the base seed. Sharing one RNG across
    # traces would mean adding a trace silently changes every later one, and results
    # from different runs would stop being comparable.
    def rng_for(name: str) -> random.Random:
        return random.Random(f"{args.seed}:{name}")

    base = {"seed": args.seed, "duration": args.duration, "rate": args.rate,
            "teams": teams, "models": models}

    write(out, "steady", gen_steady(rng_for("steady"), models, args.duration, args.rate, teams),
          {**base, "kind": "steady"})

    for skew in (0.6, 1.0, 1.4):
        name = f"zipf_s{skew}".replace(".", "")
        write(out, name,
              gen_zipf(rng_for(name), models, args.duration, args.rate, skew, teams),
              {**base, "kind": "zipf", "skew": skew})

    write(out, "bursty", gen_bursty(rng_for("bursty"), models, args.duration, args.rate, teams),
          {**base, "kind": "bursty", "phases": 4})

    # Agent traffic is driven by session count rather than arrival rate, since its
    # arrivals are produced by conversations rather than by a Poisson process.
    sessions = max(8, int(args.duration / 25))
    write(out, "agent", gen_agent(rng_for("agent"), models[:2], args.duration, sessions, teams),
          {**base, "kind": "agent", "sessions": sessions, "models": models[:2]})

    print(f"\nwrote {len(list(out.glob('*.jsonl')))} traces to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
