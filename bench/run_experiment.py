#!/usr/bin/env python
"""Run Kiln's experiments and write the result tables.

One runner for all five experiments so that the arms of a comparison differ in
exactly one variable. Each experiment restarts the gateway with a different flag,
replays the *same* recorded trace, and writes per-request rows plus a summary table.

    1  platform baseline   throughput and TTFT for 1..N replicas under steady load
    2  sleep vs cold       L1 wake, L2 wake and cold start per model size
    3  eviction policies   the five policies across Zipf skews and a bursty trace
    4  capacity            how many models one GPU serves at a p95 TTFT target
    5  routing             the four routers on shared-prefix agent traffic

Results go to docs/results/ as CSV and Markdown. Raw per-request rows are kept, not
just summaries: a percentile computed over the wrong window is the easiest way to
publish a wrong number, and keeping the rows means the window can be re-chosen later
without re-running the GPU work.

    python bench/run_experiment.py --experiment 3
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import quote

import requests

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "docs" / "results"
TRACES = REPO / "bench" / "traces"
# Not /tmp: WSL clears it when the distro restarts on its idle timeout, and a lost
# gateway log makes a surprising benchmark result impossible to explain afterwards.
LOGS = REPO / "docs" / "results" / "logs"


# --------------------------------------------------------------------- plumbing

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Service:
    """A Kiln binary under test."""

    def __init__(self, name: str, cmd: list[str], env: dict | None = None) -> None:
        self.name = name
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log = LOGS / f"{name}.log"
        self.fh = self.log.open("a")
        self.p = subprocess.Popen(cmd, stdout=self.fh, stderr=self.fh,
                                  env={**os.environ, **(env or {})}, start_new_session=True)

    def wait(self, url: str, timeout: float = 90) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.p.poll() is not None:
                raise RuntimeError(f"{self.name} exited {self.p.returncode}:\n"
                                   f"{self.log.read_text()[-2500:]}")
            with contextlib.suppress(Exception):
                if requests.get(url, timeout=2).status_code == 200:
                    return
            time.sleep(0.25)
        raise TimeoutError(f"{self.name} unhealthy:\n{self.log.read_text()[-2500:]}")

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            self.p.wait(timeout=120)
        with contextlib.suppress(Exception):
            self.fh.close()


def pct(values: list[float], q: float) -> float:
    """Percentile by nearest rank.

    Not statistics.quantiles: it interpolates, which for a p99 over a few hundred
    samples invents a value that was never observed. For latency reporting the
    observed value is the honest one.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def summarise(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if int(r["status"]) == 200]
    ttfts = [float(r["ttft_ms"]) for r in ok if float(r["ttft_ms"]) > 0]
    totals = [float(r["total_ms"]) for r in ok]
    tokens = sum(int(r["completion_tokens"]) for r in ok)
    span = (max(float(r["t_wall"]) for r in rows) -
            min(float(r["t_wall"]) for r in rows)) if rows else 0.0

    return {
        "arm": label,
        "requests": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "ttft_p50_ms": round(pct(ttfts, 0.50), 1),
        "ttft_p95_ms": round(pct(ttfts, 0.95), 1),
        "ttft_p99_ms": round(pct(ttfts, 0.99), 1),
        "total_p50_ms": round(pct(totals, 0.50), 1),
        "total_p95_ms": round(pct(totals, 0.95), 1),
        "completion_tokens": tokens,
        "tokens_per_s": round(tokens / span, 1) if span > 0 else 0.0,
        "requests_per_s": round(len(ok) / span, 2) if span > 0 else 0.0,
    }


def prom_scrape(url: str) -> dict[str, float]:
    """Pull the gateway's counters straight from /metrics.

    Counters the gateway maintains are more trustworthy than anything reconstructed
    from the client side: the client cannot see an eviction, and inferring one from a
    latency spike would be guessing.
    """
    out: dict[str, float] = {}
    with contextlib.suppress(Exception):
        for line in requests.get(url, timeout=5).text.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            name, _, value = line.rpartition(" ")
            with contextlib.suppress(ValueError):
                out[name] = out.get(name, 0.0) + float(value)
    return out


def counter_total(metrics: dict[str, float], prefix: str) -> float:
    return sum(v for k, v in metrics.items() if k.startswith(prefix))


def trace_models(trace: pathlib.Path) -> list[str]:
    """Model ids a trace addresses, from its metadata header."""
    with trace.open(encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    return list((first.get("_meta") or {}).get("models") or [])


def check_trace_matches_fleet(trace: pathlib.Path, deployed: list[str]) -> None:
    """Abort if a trace addresses models the experiment did not deploy.

    This is the failure that looks like a result. A trace generated for five models
    replayed against a fleet of three sends nearly half its requests at models that
    do not exist: they fail instantly, never trigger a swap, and drag the measured
    latency *down* while the eviction counters stay at zero. The run looks finished
    and the numbers look plausible. Better to refuse.
    """
    missing = [m for m in trace_models(trace) if m not in deployed]
    if missing:
        raise SystemExit(
            f"\ntrace {trace.name} addresses models that are not deployed: {missing}\n"
            f"deployed: {deployed}\n"
            "regenerate the traces for this fleet:\n"
            f"  python bench/tracegen.py --models {','.join(deployed)}\n"
        )


def run_locust(gateway: str, trace: pathlib.Path, out_csv: pathlib.Path,
               users: int, api_key: str, duration: str) -> list[dict]:
    """Replay a trace through the gateway and return the per-request rows."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(REPO / "bench" / "locustfile.py"),
        "--host", gateway, "--headless",
        "-u", str(users), "-r", str(max(1, users // 2)),
        "-t", duration,
        "--trace", str(trace),
        "--results", str(out_csv),
        "--only-summary",
    ]
    if api_key:
        cmd += ["--api-key", api_key]
    subprocess.run(cmd, check=False, capture_output=True, text=True)

    if not out_csv.exists():
        return []
    with out_csv.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_table(name: str, rows: list[dict], notes: str = "") -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  (no rows for {name})")
        return

    csv_path = RESULTS / f"{name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = RESULTS / f"{name}.md"
    cols = list(rows[0].keys())
    with md.open("w", encoding="utf-8") as fh:
        fh.write(f"# {name}\n\n")
        if notes:
            fh.write(notes.strip() + "\n\n")
        fh.write("| " + " | ".join(cols) + " |\n")
        fh.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    print(f"  wrote {csv_path.name} and {md.name}")


# ------------------------------------------------------------------ experiments

class Bench:
    """Shared setup: a node agent, a control plane, and a gateway per arm."""

    def __init__(self, args) -> None:
        self.args = args
        self.agent_port = free_port()
        self.cp_port = free_port()
        self.services: list[Service] = []
        self.admin_token = "bench-admin"
        self.api_key = ""

    def admin(self, method: str, path: str, **kw) -> requests.Response:
        return requests.request(
            method, f"http://127.0.0.1:{self.cp_port}{path}",
            headers={"Authorization": f"Bearer {self.admin_token}"},
            timeout=kw.pop("timeout", 120), **kw)

    def start_platform(self) -> None:
        print(f"  node agent  :{self.agent_port}")
        agent = Service("agent", [
            str(REPO / "bin" / "nodeagent"), "-addr", f"127.0.0.1:{self.agent_port}",
            "-python", sys.executable, "-node", "bench",
            "-log-dir", str(LOGS / "vllm"),
        ])
        agent.wait(f"http://127.0.0.1:{self.agent_port}/health", 30)
        self.services.append(agent)

        print(f"  control plane :{self.cp_port}")
        cp = Service("cp", [
            str(REPO / "bin" / "controlplane"), "-addr", f"127.0.0.1:{self.cp_port}",
            "-database-url", self.args.database_url,
            "-agents", f"http://127.0.0.1:{self.agent_port}",
            "-admin-token", self.admin_token, "-workers", "2",
            f"-measure-wakes={'true' if self.args.measure_wakes else 'false'}",
        ])
        cp.wait(f"http://127.0.0.1:{self.cp_port}/health", 60)
        self.services.append(cp)

        with contextlib.suppress(Exception):
            self.admin("POST", "/admin/teams", json={
                "id": "bench", "name": "Bench", "rpm": 1000000,
                "tokens_per_day": 10**12, "max_concurrency": 512})
        with contextlib.suppress(Exception):
            self.api_key = self.admin("POST", "/admin/teams/bench/keys").json()["api_key"]

    def deploy(self, model_id: str, repo: str, replicas: int = 1,
               gpu_util: float = 0.35, max_len: int = 2048) -> None:
        self.admin("POST", "/admin/models", json={
            "id": model_id, "hf_repo": repo, "replicas": replicas,
            "gpu_memory_utilization": gpu_util, "max_model_len": max_len,
            "enforce_eager": True,
        }).raise_for_status()

        deadline = time.time() + 1800
        while time.time() < deadline:
            st = self.admin("GET", f"/admin/models/{model_id}").json()["model"]
            if st["status"] == "READY":
                print(f"  {model_id}: READY  vram={st['footprint_vram_bytes']/2**20:.0f}MiB "
                      f"cold={st['cold_start_ms']}ms wakeL1={st['wake_host_ms']}ms "
                      f"wakeL2={st['wake_disk_ms']}ms")
                return st
            if st["status"] == "FAILED":
                raise RuntimeError(f"deploy of {model_id} failed: {st['error']}")
            time.sleep(3)
        raise TimeoutError(f"{model_id} never became READY")

    def sleep_model(self, model_id: str, level: int = 2) -> None:
        """Park a model in a sleep tier via the node agent.

        Setup needs this because the deploy path leaves every model ACTIVE, and the
        whole point of experiment 3 is a fleet that does not fit in VRAM at once.
        Deploying the second model while the first is still resident would simply
        fail with "no available memory for the cache blocks" -- so each model is slept
        immediately after it is measured, and the scheduler pages them back in on
        demand once the trace starts.
        """
        reps = self.admin("GET", f"/admin/models/{model_id}").json().get("replicas") or []
        for r in reps:
            requests.post(
                f"http://127.0.0.1:{self.agent_port}/replicas/{quote(r['id'], safe='')}/sleep",
                json={"level": level, "mode": "wait"}, timeout=600).raise_for_status()

    def gpu_free_mib(self) -> int:
        st = requests.get(f"http://127.0.0.1:{self.agent_port}/status", timeout=30).json()
        gpus = st.get("gpus") or []
        return round(gpus[0]["free_bytes"] / 2**20) if gpus else 0

    def gateway(self, *, router: str = "round-robin", policy: str = "",
                extra: list[str] | None = None) -> tuple[Service, str]:
        port = free_port()
        cmd = [
            str(REPO / "bin" / "gateway"), "-addr", f"127.0.0.1:{port}",
            "-agents", f"http://127.0.0.1:{self.agent_port}",
            "-database-url", self.args.database_url,
            "-router", router,
            "-gpu-bytes", str(self.args.gpu_bytes),
            "-gpu-reserve-bytes", str(self.args.gpu_reserve_bytes),
        ]
        if policy:
            cmd += ["-policy", policy]
        cmd += extra or []
        svc = Service(f"gw-{router}-{policy or 'none'}", cmd)
        svc.wait(f"http://127.0.0.1:{port}/health", 60)
        self.services.append(svc)
        return svc, f"http://127.0.0.1:{port}"

    def stop_all(self) -> None:
        for s in reversed(self.services):
            s.stop()
        self.services.clear()


def experiment_2(b: Bench) -> None:
    """Sleep vs cold, per model size.

    Uses the deploy path's own measurements rather than a separate harness, so the
    numbers are the ones the scheduler actually scores against. Measuring them twice
    in two different ways would be a good way to publish a figure the system does not
    use.
    """
    print("\n=== experiment 2: sleep vs cold start ===")
    b.start_platform()
    rows, failures = [], []
    try:
        for model_id, repo, util in b.args.model_set:
            # One model failing must not discard the others' measurements. Each of
            # these costs minutes of GPU time, and a gated repo or a bad flag on the
            # last model is no reason to throw away the first two.
            try:
                st = b.deploy(model_id, repo, gpu_util=util)
            except Exception as exc:
                print(f"  !! {model_id} failed, continuing: {exc}")
                failures.append({"model": model_id, "error": str(exc)[:500]})
                with contextlib.suppress(Exception):
                    b.admin("DELETE", f"/admin/models/{model_id}")
                    time.sleep(5)
                continue

            cold_ms = st["cold_start_ms"]
            l1, l2 = st["wake_host_ms"], st["wake_disk_ms"]
            rows.append({
                "model": model_id,
                "vram_mib": round(st["footprint_vram_bytes"] / 2**20),
                "cold_start_s": round(cold_ms / 1000, 2),
                "wake_l1_s": round(l1 / 1000, 3),
                "wake_l2_s": round(l2 / 1000, 3),
                "l1_speedup_vs_cold": round(cold_ms / l1, 1) if l1 else 0,
                "l2_speedup_vs_cold": round(cold_ms / l2, 1) if l2 else 0,
            })
            # Remove it again so each model is measured on a quiet GPU; otherwise the
            # second model's footprint measurement includes the first.
            with contextlib.suppress(Exception):
                b.admin("DELETE", f"/admin/models/{model_id}")
                time.sleep(10)
    finally:
        b.stop_all()

    if failures:
        (RESULTS / "exp2_failures.json").write_text(json.dumps(failures, indent=2))
        print(f"\n  {len(failures)} model(s) failed; recorded in exp2_failures.json")

    write_table("exp2_sleep_vs_cold", rows, notes=(
        "Measured by the deploy job on the local RTX 3070 (8 GB), vLLM 0.32.0, "
        "`--enforce-eager`. Wake L2 includes the `collective_rpc{\"method\": "
        "\"reload_weights\"}` step that vLLM's own `/wake_up` omits; without it the "
        "wake is faster and produces garbage (see docs/PROGRESS.md, M0)."))


def experiment_3(b: Bench) -> None:
    """The five eviction policies on identical traces."""
    print("\n=== experiment 3: eviction policies ===")
    traces = [TRACES / f"{n}.jsonl" for n in b.args.traces.split(",")]
    for t in traces:
        if not t.exists():
            raise SystemExit(f"missing trace {t}; run `python bench/tracegen.py` first")

    b.start_platform()
    rows = []
    try:
        # Deploy each model, then immediately put it to sleep. The fleet is chosen to
        # exceed VRAM on purpose -- that is the condition the whole feature exists for
        # -- so they cannot all be resident at once, and deploying the second while
        # the first is still active would just fail to allocate a KV cache.
        total_mib = 0
        for model_id, repo, util in b.args.model_set:
            st = b.deploy(model_id, repo, gpu_util=util)
            total_mib += round(st["footprint_vram_bytes"] / 2**20)
            b.sleep_model(model_id, level=2)
            print(f"      slept {model_id}; GPU free: {b.gpu_free_mib()} MiB")

        deployed = [m for m, _, _ in b.args.model_set]
        for trace in traces:
            check_trace_matches_fleet(trace, deployed)

        budget_mib = round((b.args.gpu_bytes - b.args.gpu_reserve_bytes) / 2**20)
        print(f"\n  fleet wants {total_mib} MiB, budget is {budget_mib} MiB "
              f"({'OVERSUBSCRIBED -- good' if total_mib > budget_mib else 'FITS -- no pressure, results will be uninformative'})")
        rows_meta = {"fleet_mib": total_mib, "budget_mib": budget_mib}

        for trace in traces:
            for policy in b.args.policies.split(","):
                print(f"\n  -- {trace.stem} x {policy}")
                svc, url = b.gateway(policy=policy, extra=[
                    "-min-residency", b.args.min_residency,
                    "-drain-deadline", b.args.drain_deadline,
                ])
                try:
                    raw = RESULTS / "raw" / f"exp3_{trace.stem}_{policy}.csv"
                    reqs = run_locust(url, trace, raw, b.args.users,
                                      b.api_key, b.args.duration)
                    m = prom_scrape(f"{url}/metrics")
                    s = summarise(reqs, f"{trace.stem}/{policy}")
                    hits = counter_total(m, "kiln_residency_hits_total")
                    misses = counter_total(m, "kiln_residency_misses_total")
                    s.update({
                        "trace": trace.stem,
                        "policy": policy,
                        "evictions": int(counter_total(m, "kiln_evictions_total")),
                        "wakes": int(counter_total(m, "kiln_wakes_total")),
                        "cold_starts": int(counter_total(m, "kiln_cold_starts_total")),
                        "thrash": int(counter_total(m, "kiln_thrash_total")),
                        "drain_timeouts": int(counter_total(m, "kiln_drain_timeouts_total")),
                        "residency_hit_ratio": round(hits / (hits + misses), 3) if hits + misses else 0.0,
                    })
                    s.update(rows_meta)
                    rows.append(s)
                    print(f"     p50 {s['ttft_p50_ms']}ms  p95 {s['ttft_p95_ms']}ms  "
                          f"p99 {s['ttft_p99_ms']}ms  hit {s['residency_hit_ratio']}  "
                          f"evict {s['evictions']}  thrash {s['thrash']}")
                finally:
                    svc.stop()
                    b.services.remove(svc)
                    # Reset the fleet to all-asleep between arms. Without this the
                    # next policy starts with whatever the previous one happened to
                    # leave resident, which silently advantages whichever arm runs
                    # after the one with the best hit ratio.
                    for model_id, _, _ in b.args.model_set:
                        with contextlib.suppress(Exception):
                            b.sleep_model(model_id, level=2)
                    time.sleep(5)
    finally:
        b.stop_all()

    write_table("exp3_eviction_policies", rows, notes=(
        "Every arm replays the identical recorded trace, so the only variable is the "
        "eviction policy. Counters are read from the gateway's own /metrics rather "
        "than inferred from client-side latency: a client cannot observe an eviction."))


def experiment_5(b: Bench) -> None:
    """The four routers on shared-prefix agent traffic."""
    print("\n=== experiment 5: prefix-aware routing ===")
    trace = TRACES / "agent.jsonl"
    if not trace.exists():
        raise SystemExit("missing bench/traces/agent.jsonl; run bench/tracegen.py")

    b.start_platform()
    rows = []
    try:
        model_id, repo, util = b.args.model_set[0]
        check_trace_matches_fleet(trace, [model_id])
        b.deploy(model_id, repo, replicas=b.args.replicas, gpu_util=util)

        for router in b.args.routers.split(","):
            print(f"\n  -- router {router}")
            svc, url = b.gateway(router=router)
            try:
                raw = RESULTS / "raw" / f"exp5_{router}.csv"
                reqs = run_locust(url, trace, raw, b.args.users, b.api_key, b.args.duration)
                m = prom_scrape(f"{url}/metrics")
                s = summarise(reqs, router)
                s["router"] = router
                # How evenly work landed. A prefix router that pins everything to one
                # replica wins on cache and loses on queueing, and this is the column
                # that shows the trade rather than hiding it inside the p95.
                by_replica: dict[str, int] = {}
                for r in reqs:
                    by_replica[r["replica"]] = by_replica.get(r["replica"], 0) + 1
                counts = sorted(by_replica.values(), reverse=True)
                s["replica_spread"] = "/".join(str(c) for c in counts)
                s["max_replica_share"] = round(counts[0] / sum(counts), 3) if counts else 0
                rows.append(s)
                print(f"     p50 {s['ttft_p50_ms']}ms  p95 {s['ttft_p95_ms']}ms  "
                      f"spread {s['replica_spread']}")
            finally:
                svc.stop()
                b.services.remove(svc)
                time.sleep(5)
    finally:
        b.stop_all()

    write_table("exp5_routing", rows, notes=(
        "Agent-style traffic: multi-turn conversations resending the same system "
        "prompt and tool definitions each turn. `replica_spread` shows how evenly "
        "work landed, which is where pure prefix affinity pays for its cache hits."))


def experiment_1(b: Bench) -> None:
    """Platform baseline across replica counts."""
    print("\n=== experiment 1: platform baseline ===")
    trace = TRACES / "steady.jsonl"
    if not trace.exists():
        raise SystemExit("missing bench/traces/steady.jsonl; run bench/tracegen.py")

    rows = []
    for replicas in [int(x) for x in b.args.replica_sweep.split(",")]:
        print(f"\n  -- {replicas} replica(s)")
        b.start_platform()
        try:
            model_id, repo, util = b.args.model_set[0]
            b.deploy(model_id, repo, replicas=replicas, gpu_util=util)
            svc, url = b.gateway(router="round-robin")
            raw = RESULTS / "raw" / f"exp1_r{replicas}.csv"
            reqs = run_locust(url, trace, raw, b.args.users, b.api_key, b.args.duration)
            s = summarise(reqs, f"{replicas} replicas")
            s["replicas"] = replicas
            rows.append(s)
            print(f"     p50 {s['ttft_p50_ms']}ms  p95 {s['ttft_p95_ms']}ms  "
                  f"{s['tokens_per_s']} tok/s")
        finally:
            b.stop_all()
            time.sleep(8)

    write_table("exp1_baseline", rows, notes=(
        "Steady load, one model, round-robin routing, identical trace per arm."))


EXPERIMENTS = {1: experiment_1, 2: experiment_2, 3: experiment_3, 5: experiment_5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", type=int, required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--database-url", default=os.environ.get("KILN_DATABASE_URL", ""))
    ap.add_argument("--duration", default="5m")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--replica-sweep", default="1,2")
    ap.add_argument("--policies", default=",".join(
        ["cold", "lru", "lfu-decay", "cost-aware", "predictive-prewarm"]))
    ap.add_argument("--routers", default="round-robin,least-loaded,prefix-aware,prefix-aware+load")
    ap.add_argument("--traces", default="zipf_s10,bursty")
    ap.add_argument("--min-residency", default="15s")
    ap.add_argument("--drain-deadline", default="10s")
    ap.add_argument("--gpu-bytes", type=int, default=8 * 1024**3)
    ap.add_argument("--gpu-reserve-bytes", type=int, default=1024**3)
    ap.add_argument("--measure-wakes", action="store_true", default=True)
    ap.add_argument("--models", default="",
                    help="id:repo:gpu_util triples, comma separated; defaults to the local small fleet")
    args = ap.parse_args()

    if not args.database_url:
        print("set KILN_DATABASE_URL", file=sys.stderr)
        return 2

    if args.models:
        args.model_set = []
        for spec in args.models.split(","):
            mid, repo, util = spec.split(":")
            args.model_set.append((mid, repo, float(util)))
    else:
        # Small models so several fit, or nearly fit, on an 8GB card. The point of
        # the local fleet is memory pressure, not model quality.
        # All openly licensed: a gated repo (Llama 3.2, for instance) 403s without a
        # Hugging Face token, and a benchmark nobody else can run is not a benchmark.
        args.model_set = [
            ("qwen-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 0.30),
            ("qwen3-0.6b", "Qwen/Qwen3-0.6B", 0.32),
            # 0.55, not 0.42: a 1.5B in bf16 is ~3.1 GB of weights, and
            # gpu_memory_utilization is the ceiling for weights *plus* the KV cache
            # arena. At 0.42 of an 8 GB card vLLM reports "Available KV cache memory:
            # -0.04 GiB" and refuses to start. That it needs over half the device is
            # the memory pressure this project exists to manage.
            ("qwen-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 0.55),
        ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)

    started = time.time()
    EXPERIMENTS[args.experiment](Bench(args))
    print(f"\nexperiment {args.experiment} finished in {(time.time()-started)/60:.1f} min")

    # A manifest beside the results, so a table can always be traced back to the
    # conditions that produced it. A results file without its conditions is not a
    # result.
    (RESULTS / f"exp{args.experiment}_conditions.json").write_text(json.dumps({
        "experiment": args.experiment,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": args.duration, "users": args.users,
        "policies": args.policies, "routers": args.routers, "traces": args.traces,
        "models": args.model_set,
        "gpu_bytes": args.gpu_bytes, "gpu_reserve_bytes": args.gpu_reserve_bytes,
        "min_residency": args.min_residency, "drain_deadline": args.drain_deadline,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
