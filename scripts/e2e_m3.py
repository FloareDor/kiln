#!/usr/bin/env python
"""M3 acceptance test: the full stack on Kubernetes, requests spread across replicas.

Runs against a live cluster rather than local binaries. Checks:

  1. every component is up, and the node agent is the ONLY pod holding a GPU;
  2. a model deploys through the in-cluster control plane with several replicas;
  3. requests are distributed across those replicas (read from the gateway's
     X-Kiln-Replica header, not inferred);
  4. Prometheus is scraping the gateway and the node agent.

Writes docs/results/m3_e2e.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from collections import Counter

import requests

REPO = pathlib.Path(__file__).resolve().parent.parent


def kubectl(*args: str, ctx: str = "kiln") -> str:
    out = subprocess.run(["kubectl", "--context", ctx, "-n", "kiln", *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def minikube_ip(profile: str) -> str:
    out = subprocess.run(["minikube", "-p", profile, "ip"], capture_output=True, text=True)
    return out.stdout.strip().splitlines()[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="kiln")
    ap.add_argument("--model", default="qwen-0.5b")
    ap.add_argument("--hf-repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--replicas", type=int, default=2)
    ap.add_argument("--gpu-util", type=float, default=0.28)
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--out", default="docs/results/m3_e2e.json")
    args = ap.parse_args()

    results: dict = {"checks": {}}
    ip = minikube_ip(args.profile)
    gw = f"http://{ip}:30080"
    prom = f"http://{ip}:30090"
    print(f"cluster at {ip}  gateway={gw}")

    print("\n[1/5] every component is running")
    pods = json.loads(kubectl("get", "pods", "-o", "json"))["items"]
    by_name = {p["metadata"]["name"]: p for p in pods}
    ready = {n: all(c.get("ready") for c in p["status"].get("containerStatuses", []))
             for n, p in by_name.items()}
    for n in sorted(ready):
        print(f"      {'ok  ' if ready[n] else 'DOWN'} {n}")
    results["pods"] = ready
    results["checks"]["all_pods_ready"] = all(ready.values()) and len(ready) >= 7

    # The architectural claim, checked rather than asserted: exactly one workload
    # requests a GPU, and it is the node agent. If anything else did, Kubernetes
    # would hand it the device for its whole lifetime and tiering would be impossible.
    gpu_holders = []
    for n, p in by_name.items():
        for c in p["spec"]["containers"]:
            lim = (c.get("resources") or {}).get("limits") or {}
            if "nvidia.com/gpu" in lim:
                gpu_holders.append(n)
    results["gpu_holders"] = gpu_holders
    results["checks"]["only_nodeagent_holds_gpu"] = (
        len(gpu_holders) == 1 and gpu_holders[0].startswith("kiln-nodeagent"))
    print(f"      GPU-requesting pods: {gpu_holders}")

    print("\n[2/5] admin token and tenant")
    token = subprocess.run(
        ["kubectl", "--context", args.profile, "-n", "kiln", "get", "secret",
         "kiln-secrets", "-o", "jsonpath={.data.admin-token}"],
        capture_output=True, text=True).stdout.strip()
    import base64
    admin_token = base64.b64decode(token).decode()

    cp = f"http://{ip}:30082"
    # The control plane has no NodePort by design (it is an internal API), so reach it
    # by port-forward for the duration of the test.
    pf = subprocess.Popen(
        ["kubectl", "--context", args.profile, "-n", "kiln", "port-forward",
         "svc/kiln-controlplane", "18082:8082"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    cp = "http://127.0.0.1:18082"

    def admin(method: str, path: str, **kw):
        return requests.request(method, cp + path,
                                headers={"Authorization": f"Bearer {admin_token}"},
                                timeout=kw.pop("timeout", 120), **kw)

    try:
        admin("POST", "/admin/teams", json={
            "id": "k8s", "name": "K8s", "rpm": 100000,
            "tokens_per_day": 10**12, "max_concurrency": 64})
        key = admin("POST", "/admin/teams/k8s/keys").json()["api_key"]
        results["checks"]["tenant_created"] = key.startswith("sk-")
        print(f"      api key {key[:16]}...")

        print(f"\n[3/5] deploying {args.hf_repo} with {args.replicas} replicas")
        t0 = time.perf_counter()
        r = admin("POST", "/admin/models", json={
            "id": args.model, "hf_repo": args.hf_repo, "replicas": args.replicas,
            "gpu_memory_utilization": args.gpu_util, "max_model_len": 1024,
            "enforce_eager": True})
        r.raise_for_status()

        deadline = time.time() + 2400
        model = None
        while time.time() < deadline:
            model = admin("GET", f"/admin/models/{args.model}").json()
            if model["model"]["status"] in ("READY", "FAILED"):
                break
            time.sleep(5)
        took = time.perf_counter() - t0
        results["deploy"] = {"seconds": round(took, 1),
                             "status": model["model"]["status"],
                             "replicas": len(model.get("replicas") or [])}
        results["checks"]["model_ready"] = model["model"]["status"] == "READY"
        results["checks"]["replicas_created"] = len(model.get("replicas") or []) == args.replicas
        print(f"      {model['model']['status']} in {took:.0f}s with "
              f"{len(model.get('replicas') or [])} replicas")
        for rep in model.get("replicas") or []:
            print(f"        {rep['id']}  port={rep['port']}  state={rep['state']}")

        print(f"\n[4/5] sending {args.requests} requests through the gateway")
        seen = Counter()
        ok = 0
        for i in range(args.requests):
            resp = requests.post(
                f"{gw}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": args.model, "max_tokens": 8, "temperature": 0,
                      "messages": [{"role": "user", "content": f"say hello {i}"}]},
                timeout=180)
            if resp.status_code == 200:
                ok += 1
                seen[resp.headers.get("X-Kiln-Replica", "?")] += 1
        results["replica_distribution"] = dict(seen)
        results["checks"]["requests_succeeded"] = ok == args.requests
        # The M3 acceptance criterion: work actually lands on more than one replica.
        results["checks"]["spread_across_replicas"] = len(seen) >= min(args.replicas, 2)
        print(f"      {ok}/{args.requests} ok, distribution: {dict(seen)}")

        print("\n[5/5] Prometheus is scraping Kiln")
        up = requests.get(f"{prom}/api/v1/query", params={"query": "up"}, timeout=30).json()
        targets = {m["metric"].get("component", m["metric"].get("job", "?")): m["value"][1]
                   for m in up["data"]["result"]}
        results["prometheus_targets"] = targets
        ttft = requests.get(f"{prom}/api/v1/query",
                            params={"query": "sum(kiln_requests_total)"}, timeout=30).json()
        series = ttft["data"]["result"]
        results["checks"]["prometheus_has_kiln_metrics"] = bool(series)
        print(f"      targets: {targets}")
        print(f"      kiln_requests_total: {series[0]['value'][1] if series else 'none'}")
    finally:
        pf.terminate()

    results["PASS"] = all(results["checks"].values())
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 70)
    for k, v in results["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 70)
    print(f"M3: {'PASS' if results['PASS'] else 'FAIL'}   (wrote {args.out})")
    return 0 if results["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
