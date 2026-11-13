# Kiln: Self-Hosted LLM Serving Platform

> One line pitch: a self-hosted platform for deploying and serving Hugging Face LLMs on Kubernetes, with an OpenAI-compatible API, auth, quotas and monitoring. Its standout feature is GPU memory tiering: it serves more models than fit on the GPU by swapping idle ones out to CPU RAM or disk.

Stack: Go, vLLM, Kubernetes, PostgreSQL, Prometheus, Grafana, Locust. GPU runs on a local RTX 3070 (8GB) and Modal for bigger benchmarks.

(Kiln is a working name. Rename freely.)

---

## 0. Instructions for Claude Code

- Build this milestone by milestone (section 7). Do not start a milestone until the previous one meets its acceptance criteria.
- After each milestone: run the tests, update `docs/PROGRESS.md` with what was done and any numbers measured, and commit.
- Keep scheduling and routing logic pure and unit-testable (no HTTP or vLLM calls inside policy code). Policies take state in and return decisions out.
- Pin the vLLM version in the Dockerfile. Before building the swapping feature, verify that sleep and wake actually free and restore GPU memory on the pinned version (some vLLM versions had sleep mode bugs). Record the result in `docs/PROGRESS.md`.
- Use small models for local dev so everything runs on an 8GB RTX 3070.
- Ask me before adding any new major dependency or changing the architecture.

---

## 1. What the platform does

A team wants to run their own LLMs instead of paying an API. They need:
1. A way to **deploy** a Hugging Face model with one API call.
2. A **standard API** (OpenAI-compatible, streaming) so existing SDKs just work.
3. **Multi-tenancy:** API keys, per-team quotas, usage tracking.
4. **Reliability:** deployments that don't break when two jobs run at once or a worker crashes.
5. **Visibility:** latency, throughput, queue depth, GPU memory in Grafana.
6. **Efficiency:** GPUs are expensive, so serve as many models and requests as possible per GPU.

Sections 3 and 4 cover 1 to 5 (the core platform). Section 5 covers 6 (the features that make this project stand out).

---

## 2. Architecture

```
            clients (OpenAI SDK, curl, Locust)
                          |
                 +------------------+
                 |   Go Gateway     |  OpenAI-compatible API, SSE streaming,
                 |                  |  API keys, quotas, per-model queues,
                 |                  |  replica routing
                 +--------+---------+
                          |
                 +------------------+
                 |  Control Plane   |  model registry, deploy job workers,
                 |  (Go)            |  placement + swap scheduler
                 +--------+---------+
                          |
          +---------------+---------------+
          |                               |
 +------------------+            +------------------+
 | Node Agent (Go)  |  1 per GPU | Node Agent (Go)  |
 | DaemonSet, owns  |            |                  |
 | the GPU, runs    |            |                  |
 | vLLM processes,  |            |                  |
 | sleep/wake, NVML |            |                  |
 +--+-----+-----+---+            +------------------+
    |     |     |
  vLLM  vLLM  vLLM

 PostgreSQL: teams, api_keys, quotas, usage, models, deployments, deploy_jobs
 Prometheus + Grafana: TTFT, throughput, queue depth, GPU memory, swaps
```

### Why a node agent instead of one pod per model
Kubernetes gives a pod exclusive use of a GPU for its whole life. One pod per model would mean one model per GPU forever. So one node agent pod owns each GPU and runs vLLM processes inside it. Kubernetes manages the gateway, control plane, node agents, Postgres and monitoring. The node agent handles sharing a GPU between models. The node agent must also run **standalone without Kubernetes** (plain Docker) so it works on Modal.

---

## 3. Core platform

### 3.1 Gateway (Go)
- `POST /v1/chat/completions` and `POST /v1/completions`, OpenAI-compatible, with `stream: true` SSE passthrough.
- `GET /v1/models` lists deployed models and their state.
- Auth: `Authorization: Bearer <api_key>`, keys stored hashed in Postgres.
- Per-team quotas: requests per minute, tokens per day, max concurrent requests. Return 429 with `Retry-After`.
- Usage tracking per team and model (prompt and completion tokens).
- Per-model request queue with timeout and max depth (needed later for swapping).
- Routing to replicas through a pluggable `Router` interface (round-robin first, smarter routing in 5.2).

### 3.2 Deployment API and jobs
- `POST /admin/models` with `{hf_repo, revision, replicas, gpu_memory_utilization, max_model_len}` creates a deployment and a deploy job.
- `GET /admin/models/{id}`, `DELETE /admin/models/{id}`, `POST /admin/models/{id}/scale`.
- Job workers pick jobs with `SELECT ... FOR UPDATE SKIP LOCKED` so jobs are concurrency-safe across multiple workers.
- Job steps: download weights to a shared cache volume, ask a node agent to start the model, warm up, measure footprint and cold start time, mark READY.
- Jobs are idempotent and retryable. A crashed worker's job is picked up again after a lease timeout.

### 3.3 Node Agent (Go)
- Runs one vLLM OpenAI server process per model replica, each on its own port, with a fixed `--gpu-memory-utilization` (and `--enable-sleep-mode`, used by 5.1).
- API: `start`, `stop`, `sleep(level)`, `wake`, `status`, `health`.
- Reports GPU memory via NVML (or `nvidia-smi` fallback) and host RAM.
- Exposes Prometheus metrics and forwards vLLM's own `/metrics`.

### 3.4 Postgres schema (starting point)
- `teams(id, name)`
- `api_keys(id, team_id, key_hash, created_at, revoked_at)`
- `quotas(team_id, rpm, tokens_per_day, max_concurrency)`
- `usage(team_id, model_id, ts, prompt_tokens, completion_tokens)`
- `models(id, hf_repo, revision, footprint_vram_mb, cold_start_ms, pinned, status)`
- `replicas(id, model_id, node, port, state)`
- `deploy_jobs(id, model_id, kind, status, attempts, error, locked_by, lease_until, updated_at)`

### 3.5 Kubernetes
- Local cluster with GPU: minikube with `--gpus all`, or kind via NVIDIA's nvkind. NVIDIA device plugin installed.
- Gateway and control plane as Deployments, node agent as a DaemonSet on GPU nodes, Postgres as a StatefulSet, Prometheus + Grafana.
- `make up` brings up everything. Manifests or a Helm chart live in `deploy/k8s/`.

### 3.6 Observability
Prometheus metrics, with the Grafana dashboard JSON committed:
- `kiln_ttft_seconds` histogram (model, team, replica)
- `kiln_tokens_per_second`, `kiln_requests_total` (status code)
- `kiln_queue_depth`, `kiln_queue_wait_seconds`
- `kiln_quota_rejections_total`
- GPU memory used/total, host RAM, in-flight requests per replica
- Feature metrics from section 5 (swaps, residency hit ratio, prefix hit ratio)

---

## 4. Load testing

Locust plus a Python trace generator in `bench/`:
- **Steady load** across a few models and teams (baseline throughput and latency).
- **Zipf popularity:** many models, a few hot (skew s = 0.6, 1.0, 1.4).
- **Bursty:** popularity shifts over time.
- **Shared-prefix / agent traffic:** multi-turn chats and tool-calling loops that resend the same system prompt and tool definitions.
- Fixed seeds, traces saved to files so every policy is tested on the same requests. Results go to CSV and `docs/results/`.

---

## 5. Standout features

### 5.1 Main feature: GPU memory tiering (model swapping)
**Problem:** you have more models than fit in VRAM. Keeping all of them loaded needs more GPUs. Loading on demand means a cold start (load weights, compile CUDA graphs, warm up) of tens of seconds per switch.

**Idea:** treat it like an OS memory hierarchy using vLLM sleep mode:

| Tier | Like an OS | State | Switch cost |
|---|---|---|---|
| GPU VRAM | RAM | Active | 0 |
| CPU RAM | Swap | Sleep L1 (weights offloaded to CPU) | fast wake |
| Disk | Page file | Sleep L2 (weights dropped, process kept) | seconds |
| Not started | Not loaded | Cold | tens of seconds |

Model state machine:
```
COLD -> STARTING -> ACTIVE <-> DRAINING -> SLEEPING_L1 <-> WAKING -> ACTIVE
                                       \-> SLEEPING_L2 <-> WAKING
any state -> FAILED (retry with backoff)
```

Scheduler responsibilities:
- Track VRAM budget per GPU and host RAM budget for L1 sleepers.
- On a request for a non-active model: pick victims until it fits, drain them, sleep them, wake the target.
- **Draining:** never sleep a model with in-flight requests. Stop routing to it and wait until in-flight is 0 or a deadline passes.
- **Swap coalescing:** 50 requests for the same sleeping model trigger exactly one wake.
- **Tier choice:** L1 if host RAM allows and the model will likely come back soon, otherwise L2.
- **Anti-thrash:** minimum residency time after a wake, plus hysteresis.
- **Pinning:** pinned models are never evicted.

Pluggable eviction policies (compared in experiments):
1. `cold` baseline: no sleep mode, kill and restart (naive on-demand loading).
2. `lru`
3. `lfu-decay`: lowest exponentially-decayed request rate.
4. `cost-aware`: lowest (predicted request rate x wake cost) per GB freed.
5. `predictive-prewarm`: cost-aware plus waking models whose arrival rate is rising.

```go
type EvictionPolicy interface {
    PickVictims(state ClusterState, target ModelID) ([]Eviction, error)
    Tick(state ClusterState, now time.Time) []Action // prewarm, demote L1->L2
}
```

Metrics: `kiln_swap_duration_seconds` (from/to state), `kiln_evictions_total`, `kiln_wakes_total`, `kiln_cold_starts_total`, `kiln_residency_hit_ratio`, `kiln_thrash_total`.

### 5.2 Second feature: prefix-aware routing across replicas
**Problem:** with several replicas of one model, round-robin scatters requests, so vLLM's prefix cache rarely gets reused.

**Idea:** the gateway hashes the prompt into token blocks and remembers which replica recently served each prefix, then scores replicas by prefix match minus a load penalty (queue depth, in-flight).

```go
type Router interface {
    Pick(req Request, replicas []ReplicaState) (ReplicaID, error)
}
```
Routers: `round-robin`, `least-loaded`, `prefix-aware`, `prefix-aware + load`.
Metrics: `kiln_prefix_hit_ratio`, vLLM prefix cache hit rate, TTFT per router.

### Stretch features (only after everything else)
- Tiered KV cache with LMCache (GPU to CPU to Redis) shared across replicas.
- Multi-LoRA: per-team adapters on a shared base model.
- KEDA autoscaling of replicas on queue depth.

---

## 6. Experiments

Each produces a table plus a Grafana screenshot in `docs/results/`:
1. **Platform baseline:** throughput and p50/p95 TTFT for 1 to 4 replicas under steady load.
2. **Sleep vs cold:** L1 wake, L2 wake and cold start time per model size.
3. **Eviction policies:** p50/p95/p99 TTFT, residency hit ratio, swaps, thrash for each policy across Zipf skews and bursty traces.
4. **Capacity:** how many models one GPU can serve at a p95 TTFT target, with swapping vs cold loading.
5. **Routing:** p95 TTFT and prefix hit ratio for each router on shared-prefix / agent traffic.

Hardware:
- **Local (RTX 3070, 8GB):** small models such as Qwen3-0.6B, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, SmolLM2-1.7B-Instruct. Only 1 or 2 fit active, which gives real memory pressure.
- **Modal (A10G/L4/A100):** node agents run standalone with 7-8B models and multiple replicas for the headline numbers. Benchmark scripts target local or Modal via a config file.

---

## 7. Milestones

**M0. Sanity check (half day)**
- Run 2 small vLLM models on the 3070. Measure cold start, and verify sleep L1/L2 and wake actually free and restore GPU memory.
- Done when: a printed table shows timings and memory before/after.

**M1. Gateway + node agent, one model (2-3 days)**
- Node agent starts a vLLM process. Gateway exposes the OpenAI-compatible streaming API and proxies to it.
- Done when: the OpenAI Python SDK streams from the platform.

**M2. Postgres, auth, quotas, deploy jobs (3 days)**
- API keys, per-team quotas with 429s, usage tracking, deployment API, concurrency-safe deploy jobs with leases.
- Done when: deploying a model is one API call, two teams with different quotas get throttled correctly, and a job survives a worker crash.

**M3. Multiple replicas + Kubernetes (3 days)**
- Multiple replicas per model, round-robin router. Full stack on a local k8s cluster with GPU.
- Done when: `make up` brings up everything and requests are spread across replicas.

**M4. Observability + load testing (2 days)**
- All core metrics, Grafana dashboard, Locust + trace generator.
- Done when: a Locust run fills the dashboard and saves results. Experiment 1 is done.

**M5. Model swapping (4-5 days)**
- State machine, budgets, draining, coalescing, anti-thrash, LRU and cold baseline, then lfu-decay, cost-aware and predictive-prewarm.
- Done when: scheduler unit tests pass and experiments 2 to 4 are done locally.

**M6. Prefix-aware routing (2-3 days)**
- Prefix hashing, replica scoring, all four routers.
- Done when: experiment 5 is done.

**M7. Modal scale run + polish (2-3 days)**
- Headline benchmarks on Modal with 7-8B models.
- README with architecture diagram, results tables, Grafana screenshot, short demo GIF. `docs/DESIGN.md` explains the main tradeoffs.

---

## 8. Repo layout

```
kiln/
  cmd/
    gateway/
    controlplane/
    nodeagent/
  internal/
    api/              OpenAI-compatible handlers, SSE
    auth/             api keys, quotas, usage
    routing/          routers (pure, unit tested)
    scheduler/        swap state machine, budgets, eviction policies (pure, unit tested)
    jobs/             deploy job workers (SKIP LOCKED + leases)
    nodeagent/        vLLM process mgmt, sleep/wake, NVML
    agentclient/
    store/            postgres (pgx or sqlc)
    metrics/
  deploy/
    k8s/
    docker/
    modal/
  bench/
    locustfile.py
    tracegen.py
    run_experiment.py
  dashboards/
  migrations/
  scripts/
    verify_sleep_mode.py
  docs/
    PROGRESS.md
    DESIGN.md
    results/
  Makefile
  README.md
```

---

## 9. Resume bullets (fill in real numbers at the end)

**Kiln: Self-Hosted LLM Serving Platform** | Go, vLLM, Kubernetes, PostgreSQL, Grafana

- Built a self-hosted LLM serving platform for deploying Hugging Face models on Kubernetes, exposing OpenAI-compatible streaming APIs with API-key authentication, per-team quotas, and concurrency-safe deployment jobs.
- Added GPU memory tiering that swaps idle models between VRAM, CPU RAM and disk using vLLM sleep mode; a cost-aware eviction policy served N models on one GPU and cut p95 time-to-first-token by X% versus LRU and Yx versus cold loading.
- Load-tested with Locust on Zipf and agent-style traffic; prefix-aware routing reduced p95 TTFT by Z% versus round-robin across replicas, with Grafana monitoring latency, throughput, queue depth, and swap activity.

If space is tight, keep bullets 1 and 2.

---

## 10. Interview story

"It's a self-hosted LLM platform: deploy a Hugging Face model with one call, get an OpenAI-compatible API with auth and quotas, all on Kubernetes. The part I went deep on is GPU memory. GPUs are expensive, so I made it work like virtual memory: hot models stay in VRAM, idle ones get paged to CPU RAM or disk, and a scheduler decides what to evict and what to prewarm."

Be ready for:
- Why not one pod per model? (k8s GPU allocation is exclusive for the pod's life.)
- How are deploy jobs concurrency-safe? (SKIP LOCKED plus leases, idempotent steps.)
- What happens to in-flight requests on a model being evicted? (Drain with a deadline.)
- How do you avoid thrashing? (Min residency, hysteresis, cost-aware scoring.)
- How is this different from an OS page cache? (Pages have very different sizes and costs, and you can predict arrivals.)
- Why prefix-aware routing helps, and when it hurts? (Hot spots on one replica, so add a load penalty.)
- What breaks at scale? (Host RAM for L1 sleepers, cross-node placement, cold disk reads.)

---

## 11. Reference links

Inspiration
- Original project: https://github.com/MitchellGray100/InferenceServing

Swapping
- vLLM Sleep Mode blog (levels, API, benchmarks): https://vllm.ai/blog/sleep-mode
- llmux (multiplexer using vLLM sleep/wake, closest existing tool): https://docs.rs/crate/llmux/0.5.0
- llm-d Fast Model Actuation (sleep/wake on k8s): https://llm-d.ai/docs/well-lit-paths/foundations/fast-model-actuation

Routing and k8s serving stacks
- llm-d inference scheduler architecture: https://llm-d.ai/docs/architecture/Components/inf-scheduler
- llm-d KV cache aware scheduling blog: https://llm-d.ai/blog/kvcache-wins-you-can-see
- Gateway API Inference Extension: https://github.com/kubernetes-sigs/gateway-api-inference-extension
- vLLM production-stack KV offloading tutorial: https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.6/tutorials/kv_cache.html
- vLLM repo: https://github.com/vllm-project/vllm
- NVIDIA k8s device plugin: https://github.com/NVIDIA/k8s-device-plugin
- minikube GPU tutorial: https://minikube.sigs.k8s.io/docs/tutorials/nvidia/

Papers (related work for the README)
- AlpaServe (multiplexing models on shared GPUs): https://arxiv.org/abs/2302.11665
- ServerlessLLM (fast cold starts for LLMs): https://arxiv.org/abs/2401.14351
- S-LoRA (multi-LoRA stretch feature): https://arxiv.org/abs/2311.03285
- DistServe retro (prefill/decode disaggregation context): https://hao-ai-lab.github.io/blogs/distserve-retro

Tooling
- Modal vLLM example: https://modal.com/docs/examples/vllm_inference
- Locust: https://locust.io
