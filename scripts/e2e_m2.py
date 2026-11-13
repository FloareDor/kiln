#!/usr/bin/env python
"""M2 acceptance test: deploy in one API call, per-team quotas, crash-safe jobs.

Brings up Postgres-backed control plane + node agent + gateway and checks the three
things M2 claims:

  1. **Deploying a model is one API call.** `POST /admin/models` with a Hugging Face
     repo returns immediately with a job id; a worker downloads the weights, starts
     vLLM, measures the footprint and the wake costs, and marks the model READY.
  2. **Two teams with different quotas are throttled correctly.** A tight team gets
     429s with a sane `Retry-After` while a generous team keeps serving, and the two
     do not interfere.
  3. **A job survives a worker crash.** The control plane is SIGKILLed mid-deploy;
     a second control plane picks the job up from its expired lease and finishes it.

Writes docs/results/m2_e2e.json.
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
import time

import requests

REPO = pathlib.Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    def __init__(self, name: str, cmd: list[str], env: dict | None = None) -> None:
        self.name, self.cmd, self.env = name, cmd, env or {}
        self.log = pathlib.Path("/tmp") / f"kiln-{name}.log"
        self.start()

    def start(self) -> None:
        self.fh = self.log.open("a")
        self.p = subprocess.Popen(self.cmd, stdout=self.fh, stderr=self.fh,
                                  env={**os.environ, **self.env}, start_new_session=True)

    def wait_http(self, url: str, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.p.poll() is not None:
                raise RuntimeError(f"{self.name} exited {self.p.returncode}:\n{self.log.read_text()[-2500:]}")
            with contextlib.suppress(Exception):
                if requests.get(url, timeout=2).status_code == 200:
                    return
            time.sleep(0.25)
        raise TimeoutError(f"{self.name} unhealthy:\n{self.log.read_text()[-2500:]}")

    def kill9(self) -> None:
        """SIGKILL the process group: a crash, not a shutdown. No deferred cleanup
        runs, which is exactly the case the lease exists to recover from."""
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
        with contextlib.suppress(Exception):
            self.p.wait(timeout=30)

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            self.p.wait(timeout=90)
        with contextlib.suppress(Exception):
            self.fh.close()


ADMIN_TOKEN = "m2-admin-token"


def admin(cp: str, method: str, path: str, **kw) -> requests.Response:
    return requests.request(method, f"{cp}{path}",
                            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                            timeout=kw.pop("timeout", 60), **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-0.5b")
    ap.add_argument("--hf-repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--database-url", default=os.environ.get("KILN_DATABASE_URL", ""))
    ap.add_argument("--out", default="docs/results/m2_e2e.json")
    args = ap.parse_args()
    if not args.database_url:
        print("set KILN_DATABASE_URL", file=sys.stderr)
        return 2

    agent_port, cp_port, gw_port = free_port(), free_port(), free_port()
    cp_url = f"http://127.0.0.1:{cp_port}"
    results: dict = {"checks": {}}
    procs: list[Proc] = []

    # A clean database each run so quota and job assertions are not polluted by a
    # previous run's rows.
    subprocess.run(["psql", args.database_url, "-q", "-c",
                    "TRUNCATE deploy_jobs, replicas, models, usage, quotas, api_keys, teams CASCADE"],
                   check=False, capture_output=True)

    try:
        print(f"[1/6] node agent on :{agent_port}")
        agent = Proc("m2-nodeagent", [
            str(REPO / "bin" / "nodeagent"), "-addr", f"127.0.0.1:{agent_port}",
            "-python", sys.executable, "-node", "local-3070", "-log-dir", "/tmp/kiln-logs",
        ])
        procs.append(agent)
        agent.wait_http(f"http://127.0.0.1:{agent_port}/health", 30)

        print(f"[2/6] control plane on :{cp_port}")
        cp_cmd = [
            str(REPO / "bin" / "controlplane"), "-addr", f"127.0.0.1:{cp_port}",
            "-database-url", args.database_url,
            "-agents", f"http://127.0.0.1:{agent_port}",
            "-admin-token", ADMIN_TOKEN, "-workers", "2",
            # Wake measurement costs a sleep and two wakes; M5 needs the numbers, and
            # measuring them here proves the deploy path records them.
            "-measure-wakes=true",
        ]
        cp = Proc("m2-controlplane", cp_cmd)
        procs.append(cp)
        cp.wait_http(f"{cp_url}/health", 30)

        print("[3/6] teams, quotas and keys")
        r = admin(cp_url, "POST", "/admin/teams", json={
            "id": "tight", "name": "Tight", "rpm": 3, "tokens_per_day": 1000000, "max_concurrency": 2})
        r.raise_for_status()
        r = admin(cp_url, "POST", "/admin/teams", json={
            "id": "generous", "name": "Generous", "rpm": 600, "tokens_per_day": 1000000, "max_concurrency": 16})
        r.raise_for_status()
        tight_key = admin(cp_url, "POST", "/admin/teams/tight/keys").json()["api_key"]
        gen_key = admin(cp_url, "POST", "/admin/teams/generous/keys").json()["api_key"]
        results["checks"]["keys_issued"] = tight_key.startswith("sk-") and gen_key.startswith("sk-")
        print(f"      tight={tight_key[:16]}... generous={gen_key[:16]}...")

        print(f"[4/6] deploying {args.hf_repo} in ONE call")
        t0 = time.perf_counter()
        r = admin(cp_url, "POST", "/admin/models", json={
            "id": args.model, "hf_repo": args.hf_repo, "replicas": 1,
            "gpu_memory_utilization": 0.35, "max_model_len": 2048, "enforce_eager": True})
        r.raise_for_status()
        body = r.json()
        print(f"      accepted, job {body['job']['id'][:8]} (HTTP {r.status_code})")
        results["checks"]["deploy_is_one_call"] = r.status_code == 202 and body["job"]["id"] != ""

        # Idempotency: the obvious client behaviour on a timeout is to POST again, and
        # a second live job for one model would race to start replicas.
        r2 = admin(cp_url, "POST", "/admin/models", json={
            "id": args.model, "hf_repo": args.hf_repo, "replicas": 1})
        results["checks"]["duplicate_deploy_is_idempotent"] = (
            r2.status_code == 202 and r2.json()["job"]["id"] == body["job"]["id"])

        deadline = time.time() + 1200
        status = None
        while time.time() < deadline:
            status = admin(cp_url, "GET", f"/admin/models/{args.model}").json()
            if status["model"]["status"] in ("READY", "FAILED"):
                break
            time.sleep(2)
        deploy_s = time.perf_counter() - t0
        m = status["model"]
        results["deploy"] = {
            "seconds": round(deploy_s, 1), "status": m["status"],
            "footprint_vram_bytes": m["footprint_vram_bytes"],
            "cold_start_ms": m["cold_start_ms"],
            "wake_host_ms": m["wake_host_ms"], "wake_disk_ms": m["wake_disk_ms"],
        }
        results["checks"]["model_ready"] = m["status"] == "READY"
        # The cost-aware eviction policy scores against these; zeros would make every
        # wake look free and the policy would stop distinguishing models at all.
        results["checks"]["measurements_recorded"] = (
            m["footprint_vram_bytes"] > 0 and m["cold_start_ms"] > 0
            and m["wake_host_ms"] > 0 and m["wake_disk_ms"] > 0)
        print(f"      {m['status']} in {deploy_s:.0f}s | vram={m['footprint_vram_bytes']/2**20:.0f}MiB "
              f"cold={m['cold_start_ms']}ms wakeL1={m['wake_host_ms']}ms wakeL2={m['wake_disk_ms']}ms")

        print(f"[5/6] gateway on :{gw_port}, quota enforcement")
        gw = Proc("m2-gateway", [
            str(REPO / "bin" / "gateway"), "-addr", f"127.0.0.1:{gw_port}",
            "-agents", f"http://127.0.0.1:{agent_port}",
            "-database-url", args.database_url, "-router", "round-robin",
        ])
        procs.append(gw)
        gw.wait_http(f"http://127.0.0.1:{gw_port}/health", 30)
        gw_url = f"http://127.0.0.1:{gw_port}"

        def chat(key: str) -> requests.Response:
            return requests.post(f"{gw_url}/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"},
                                 json={"model": args.model, "max_tokens": 4, "temperature": 0,
                                       "messages": [{"role": "user", "content": "hi"}]},
                                 timeout=120)

        # No key at all.
        anon = requests.post(f"{gw_url}/v1/chat/completions",
                             json={"model": args.model, "messages": []}, timeout=30)
        results["checks"]["unauthenticated_rejected"] = anon.status_code == 401
        bad = chat("sk-kiln-totally-invalid")
        results["checks"]["bad_key_rejected"] = bad.status_code == 401
        print(f"      no key -> {anon.status_code}, bad key -> {bad.status_code}")

        # The tight team: 3 rpm means the 4th request in a burst must be throttled.
        tight_codes = [chat(tight_key).status_code for _ in range(6)]
        throttled = [c for c in tight_codes if c == 429]
        results["tight_team_codes"] = tight_codes
        results["checks"]["tight_team_throttled"] = len(throttled) >= 2
        last = chat(tight_key)
        retry_after = last.headers.get("Retry-After")
        results["checks"]["retry_after_present"] = (
            last.status_code != 429 or (retry_after is not None and int(retry_after) >= 1))
        print(f"      tight(3rpm) codes={tight_codes} Retry-After={retry_after}")

        # The generous team must be unaffected by the other tenant's throttling.
        gen_codes = [chat(gen_key).status_code for _ in range(6)]
        results["generous_team_codes"] = gen_codes
        results["checks"]["generous_team_unaffected"] = all(c == 200 for c in gen_codes)
        print(f"      generous(600rpm) codes={gen_codes}")

        # The gateway buffers usage rows and flushes on a timer, so that the
        # database is not on the latency path of every request. Wait for a flush
        # rather than racing it.
        usage = []
        deadline = time.time() + 20
        while time.time() < deadline:
            usage = admin(cp_url, "GET", "/admin/usage?since=1h").json()["usage"] or []
            if any(u["team_id"] == "generous" for u in usage):
                break
            time.sleep(1)
        results["usage"] = usage
        by_team = {u["team_id"]: u for u in usage}
        results["checks"]["usage_tracked_per_team"] = (
            "generous" in by_team and by_team["generous"]["completion_tokens"] > 0)
        print(f"      usage rows: {usage}")

        print("[6/6] killing the control plane mid-deploy; a second one must finish the job")
        # A fresh model so there is a real deploy in flight to interrupt.
        crash_model = "crash-test"
        r = admin(cp_url, "POST", "/admin/models", json={
            "id": crash_model, "hf_repo": args.hf_repo, "replicas": 1,
            "gpu_memory_utilization": 0.3, "max_model_len": 1024, "enforce_eager": True})
        r.raise_for_status()
        crash_job = r.json()["job"]["id"]

        # Wait until a worker has genuinely claimed it, so the kill interrupts work
        # rather than racing the enqueue.
        deadline = time.time() + 120
        claimed = False
        while time.time() < deadline:
            jobs = admin(cp_url, "GET", "/admin/jobs").json()["jobs"]
            j = next((x for x in jobs if x["id"] == crash_job), None)
            if j and j["status"] == "RUNNING" and j.get("locked_by"):
                claimed = True
                break
            time.sleep(0.5)
        results["checks"]["job_was_claimed_before_crash"] = claimed
        print(f"      job claimed: {claimed}; SIGKILLing the control plane")

        cp.kill9()
        procs.remove(cp)
        # The worker is gone without failing, completing or renewing anything. Only
        # the lease can recover this.
        time.sleep(2)

        cp2 = Proc("m2-controlplane-2", cp_cmd)
        procs.append(cp2)
        cp2.wait_http(f"{cp_url}/health", 60)
        print("      replacement control plane up; waiting for the orphaned job")

        deadline = time.time() + 1200
        final = None
        while time.time() < deadline:
            jobs = admin(cp_url, "GET", "/admin/jobs").json()["jobs"]
            final = next((x for x in jobs if x["id"] == crash_job), None)
            if final and final["status"] in ("DONE", "FAILED"):
                break
            time.sleep(2)
        results["crash_job"] = final
        results["checks"]["job_survived_worker_crash"] = bool(final and final["status"] == "DONE")
        results["checks"]["crash_job_was_retried"] = bool(final and final["attempts"] >= 2)
        print(f"      job {crash_job[:8]} -> {final['status']} after {final['attempts']} attempts")
    finally:
        for p in reversed(procs):
            p.stop()

    results["PASS"] = all(results["checks"].values())
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 72)
    for k, v in results["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 72)
    print(f"M2: {'PASS' if results['PASS'] else 'FAIL'}   (wrote {args.out})")
    return 0 if results["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
