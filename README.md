# Kiln

**A self-hosted platform for serving Hugging Face LLMs on Kubernetes — with an OpenAI-compatible API, multi-tenancy, and GPU memory tiering that serves more models than fit in VRAM.**

Deploy a model with one API call. Point any OpenAI SDK at it. Kiln handles auth, quotas, usage, routing and monitoring — and when you have more models than GPU memory, it pages the idle ones out to CPU RAM or disk instead of unloading them.

Paging a model back in costs **0.6–1.9 seconds**. Reloading it from scratch costs **34–41 seconds**.

<div align="center">
  <img src="docs/img/architecture.svg" alt="Kiln architecture: clients hit a Go gateway that owns GPU residency and routing; a control plane owns deployments and drives deploy jobs out of Postgres; one node agent per GPU runs vLLM processes as children" width="100%">
</div>

---

## Why this exists

GPUs are expensive and VRAM is the binding constraint. An 8 GB card fits two small models. A team that wants to serve six either buys six GPUs' worth of capacity, or loads on demand and pays a cold start on the request path — and a 35-second cold start is not a slow request, it's a timeout.

Kiln treats it as a memory hierarchy problem, because that's what it is:

| Tier | OS analogue | Mechanism | Measured wake |
|---|---|---|---|
| **GPU VRAM** | resident pages | — | 0 |
| **CPU RAM** | swap | vLLM sleep level 1 | **0.59–0.88 s** |
| **Disk** | page file | sleep level 2 + weight reload | **0.71–1.87 s** |
| **Not loaded** | not in the working set | spawn, load, profile | **34–41 s** |

<sub>RTX 3070 Laptop (8 GB), Qwen2.5-0.5B-Instruct, vLLM 0.32.0, `--enforce-eager`. Reproduce with `make m0`.</sub>

A scheduler decides what to evict, which tier to evict it to, and what to wake before demand arrives.

> [!NOTE]
> The cold-start baseline is measured against a **warm page cache**. The first-ever load pays a disk read no later cold start pays; using it would overstate sleep mode's advantage by about 13%.

---

## Quickstart

**Requirements:** an NVIDIA GPU, Linux (or WSL2), Python 3.12, Go 1.24+, Docker, PostgreSQL.

```bash
git clone https://github.com/sairavig/kiln && cd kiln
make setup          # pins vLLM, installs the Go toolchain and Postgres
make build          # gateway, controlplane, nodeagent
make m0             # verify sleep mode actually frees and restores VRAM
```

Run the stack locally:

```bash
make db             # start Postgres, apply migrations
./bin/nodeagent    -addr :8081 -python "$(which python)" &
./bin/controlplane -addr :8082 -database-url "$KILN_DATABASE_URL" \
                   -agents http://127.0.0.1:8081 -admin-token secret &
./bin/gateway      -addr :8080 -database-url "$KILN_DATABASE_URL" \
                   -agents http://127.0.0.1:8081 -policy cost-aware &
```

Create a tenant and deploy a model — one call each:

```bash
curl -X POST localhost:8082/admin/teams -H "Authorization: Bearer secret" \
  -d '{"id":"acme","rpm":60,"tokens_per_day":1000000,"max_concurrency":8}'

KEY=$(curl -sX POST localhost:8082/admin/teams/acme/keys \
      -H "Authorization: Bearer secret" | jq -r .api_key)

curl -X POST localhost:8082/admin/models -H "Authorization: Bearer secret" \
  -d '{"id":"qwen","hf_repo":"Qwen/Qwen2.5-0.5B-Instruct","gpu_memory_utilization":0.35}'
```

Then use it like any OpenAI endpoint — no Kiln-specific client code:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=KEY)

for chunk in client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Count to twelve."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

On Kubernetes:

```bash
make cluster        # minikube with GPU passthrough + NVIDIA device plugin
make up             # build images, load them, deploy the stack + Prometheus/Grafana
```

---

## What you get

**OpenAI-compatible API.** `/v1/chat/completions`, `/v1/completions`, `/v1/models`, SSE streaming. Existing SDKs work by changing a base URL.

**One-call deployment.** `POST /admin/models` with a Hugging Face repo. A worker downloads the weights, starts vLLM, measures the footprint and both wake costs, and marks it `READY`.

**Multi-tenancy.** API keys stored as hashes, per-team limits on requests/minute, tokens/day and concurrency, `429` with a correct `Retry-After`, and usage tracked per team and model — including for streaming requests.

**Reliability.** Deploy jobs are claimed with `SELECT … FOR UPDATE SKIP LOCKED` plus leases, so several control planes can drain one queue and a crashed worker's job is reclaimed rather than lost.

**Observability.** Prometheus metrics for TTFT, throughput, queue depth, GPU and host memory, swaps, evictions, residency hit ratio and thrash. [Grafana dashboard](dashboards/kiln.json) committed as JSON.

**GPU memory tiering.** The part this project goes deep on — see below.

---

## GPU memory tiering

Every model moves through one state machine. Eviction always drains first; a wake never starts while a victim still holds its memory arena.

<div align="center">
  <img src="docs/img/state-machine.svg" alt="Model state machine: COLD to STARTING to ACTIVE; eviction goes ACTIVE to DRAINING to SLEEPING_L1 or SLEEPING_L2 and back through WAKING; an abandoned drain returns straight to ACTIVE" width="100%">
</div>

Five eviction policies, selected with `-policy`. They share one fitting loop, one tier-selection rule and one thrash guard, and differ **only in a scoring function** — so any benchmark gap between them is attributable to the score and nothing else:

| Policy | Evicts the model with the lowest… | Blind to |
|---|---|---|
| `cold` | last-request time, and **kills the process** | everything — this is the no-tiering baseline |
| `lru` | last-request time | size, wake cost |
| `lfu-decay` | exponentially decayed arrival rate | size, wake cost |
| `cost-aware` | `rate × wake_cost / GB_freed` | correlations between models |
| `predictive-prewarm` | same, plus wakes models whose demand is rising | — |

The per-GB normalisation in `cost-aware` is the part that wins: LRU will evict a 6 GB model that takes 40 s to wake in order to admit a 1 GB one, when evicting two small cheap models would have been strictly better.

**Three anti-thrash guards.** Two models that each receive steady traffic, on a GPU that fits one, would otherwise evict each other on every alternating request:

- **Minimum residency** — a model that has just started serving is not evictable.
- **Hysteresis** — free slightly more than needed, so the next model doesn't immediately trigger another eviction cycle.
- **Idle demotion** — a host-tier sleeper nobody wants is demoted to disk, returning pinned RAM.

> [!IMPORTANT]
> When a swap can't proceed, Kiln distinguishes `ErrWouldThrash` (every candidate is inside its residency window — **waiting will help**, so the request queues) from `ErrCannotFit` (the model is bigger than the budget, or everything else is pinned — **waiting never helps**, so it fails fast). Collapsing these means either failing requests that would have succeeded, or hanging on ones that never could.

**Swap coalescing.** 50 concurrent requests for one sleeping model produce exactly **one** wake. The other 49 wait on it — and if it fails, all 49 are told, rather than being routed at a model that isn't there.

**Draining with a deadline.** A victim is never slept while requests are in flight. But waiting forever lets one slow generation block eviction indefinitely, so the wait is bounded and timeouts are counted in `kiln_drain_timeouts_total` rather than hidden.

---

## Prefix-aware routing

With several replicas of one model, round-robin scatters consecutive turns of a conversation, so vLLM's prefix cache rarely gets reused. Agent traffic — a long system prompt plus tool definitions resent every turn — is mostly shared prefix.

The gateway hashes prompts into **cumulative** leading blocks (block *k*'s hash covers the prompt from the start through block *k*), then scores replicas by prefix match minus a load penalty. Four routers: `round-robin`, `least-loaded`, `prefix-aware`, `prefix-aware+load`.

> [!WARNING]
> Pure prefix affinity creates hot spots: every request for a popular prefix lands on one replica, which queues while its peers idle. That's why `prefix-aware+load` exists, and why both are kept as separate arms — the failure mode is worth showing, not asserting.

---

## Results

All numbers measured on an **RTX 3070 Laptop (8 GB)** under WSL2, vLLM 0.32.0, `--enforce-eager`. Raw per-request rows and the exact conditions for each run are in [`docs/results/`](docs/results).

### Sleep vs cold start

| Metric | Value |
|---|---|
| Resident VRAM when active | 2867 MiB |
| Freed by sleep L1 / L2 | **2695 MiB (92.1%)** both |
| Cold start (warm page cache) | 34.3 s |
| Wake from host tier (L1) | **0.88 s — 39× faster** |
| Wake from disk tier (L2) | **1.87 s — 18× faster** |
| Output identical after wake | L1 ✅ L2 ✅ |

### Platform

| Metric | Value |
|---|---|
| Deploy → `READY` (one API call) | 46 s |
| Streaming TTFT vs total | **0.071 s vs 1.107 s** (37 chunks) |
| Quota enforcement, 3 rpm team | `200 200 200 429 429 429`, `Retry-After: 19` |
| Quota enforcement, 600 rpm team | six 200s, unaffected by the other tenant |
| Deploy job survives `SIGKILL` | reclaimed from expired lease, **DONE after 2 attempts** |

<sub>Reproduce: `make m1`, `make m2`.</sub>

### Eviction policies

> [!NOTE]
> **The policy ranking is not claimed, and the table is deliberately absent.**
> Comparing LRU against cost-aware needs sustained memory pressure, and on an 8 GB
> card that means running at the edge of the device — where swap operations contend
> with inference and a wake measured at 0.59 s idle took **81 s** under load.
> Throughput collapses to ~0.14 req/s, so an arm records ~40 requests and most of the
> latency measured is contention, not policy. Backing the load off removes the
> pressure the experiment is about. There is no setting that gives both here.
>
> What *is* claimed: the mechanism above, measured on hardware, and the five policy
> implementations, covered by 29 unit tests that pin down the case each one exists
> for. Which policy wins under load needs a GPU with headroom — see
> [`docs/PROGRESS.md`](docs/PROGRESS.md) for the full reasoning and what it cost to
> establish.

### Routing

Run with `make bench EXP=5`. Each arm replays the **identical recorded trace** from `bench/traces/`, so the only variable is the router.

---

## How it's built

```
cmd/            gateway · controlplane · nodeagent
internal/
  api/          OpenAI handlers, SSE proxy, admin API, auth middleware
  auth/         hashed API keys, token-bucket quotas          (pure)
  routing/      four routers, cumulative prefix hashing       (pure)
  scheduler/    state machine, VRAM accounting, 5 policies    (pure)
  swap/         the tiering executor: drain, evict, wake
  jobs/         deploy workers — SKIP LOCKED + leases
  nodeagent/    vLLM process supervision, sleep/wake, GPU probe
  store/        PostgreSQL (pgx), embedded migrations
  metrics/      Prometheus instrumentation
bench/          trace generator, Locust harness, experiment runner
deploy/         Kubernetes manifests, Dockerfiles, Modal
docs/           DESIGN.md, PROGRESS.md, results/
```

Three decisions shape everything else:

**The node agent owns the GPU, not one pod per model.** Kubernetes allocates a GPU to a pod for the pod's *entire lifetime* — there's no "release the GPU but keep running". One Deployment per model would pin exactly one model to each GPU forever, making tiering unexpressible. So a long-lived supervisor holds the GPU and runs vLLM as child processes. The same binary runs standalone under plain Docker, which is how the Modal runs work.

**Residency lives in the gateway, deployments in the control plane.** Draining decides this: a model may only be slept once in-flight requests reach zero, and the gateway is the only component that knows that count exactly and instantly. The corollary is that the gateway keeps serving when the control plane is down.

**Policy code is pure.** `internal/scheduler` and `internal/routing` perform no I/O and read no clock — they take state in and return decisions out. The interesting logic is covered by unit tests that run in milliseconds with no GPU attached, because a suite that needs a GPU is a suite that stops being run.

Full rationale, including where the OS-memory-hierarchy analogy breaks down and what I'd do next: **[docs/DESIGN.md](docs/DESIGN.md)**.

---

## Measurement integrity

Several design decisions exist only to stop this project from publishing numbers that are wrong in a flattering direction. Two worth calling out:

> [!CAUTION]
> **vLLM's level-2 wake is silently broken without a weight reload.** `POST /wake_up` returns `200` in 0.45 s and then generates `'!!!!!!!!!!!!'` — level-2 sleep discards the weights and `wake_up` restores only buffers. The correct protocol adds `collective_rpc{"method": "reload_weights"}` and costs 1.87 s.
>
> The broken wake is **four times faster than the correct one**. A timing-only benchmark would have put a garbage-producing tier at the top of the results table. Every wake is now checked against a pre-sleep reference output.

> [!CAUTION]
> **The first version of the M0 gate passed having measured nothing.** NVML per-process accounting returns an empty list under WSL2, so it compared *0 MiB freed* against *0 MiB resident*, concluded `0 >= 0.5 × 0`, and printed PASS. It now detects that, falls back to device-wide memory minus a measured idle baseline, states which method it used, and refuses to grade a run whose footprint came out as zero.

Also: TTFT is asserted to be well below total latency (a buffering proxy would make every latency number a property of the gateway, not the model); percentiles are nearest-rank rather than interpolated; swap counters are read from the gateway's own `/metrics` rather than inferred from client-side latency, because a client cannot observe an eviction.

---

## Development

```bash
make test       # unit tests — no GPU, no database, no network
make test-all   # adds the PostgreSQL integration suite
make vet
make traces     # regenerate the fixed benchmark traces
```

The default suite is offline and GPU-free. The Postgres suites skip unless `KILN_TEST_DATABASE_URL` is set, and each migrates into its own schema so `go test ./...` can run packages in parallel.

---

## Reference

- [vLLM sleep mode](https://docs.vllm.ai) — the mechanism tiering is built on
- [AlpaServe](https://arxiv.org/abs/2302.11665) — multiplexing models on shared GPUs
- [ServerlessLLM](https://arxiv.org/abs/2401.14351) — fast cold starts for LLMs
- [llm-d fast model actuation](https://llm-d.ai) — sleep/wake on Kubernetes
- [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) — upstream KV-aware routing
