# Kiln Inference Serving

Kiln is a self-hosted inference serving platform for deploying Hugging Face LLMs as
vLLM workers on Kubernetes. It includes a Go API gateway with OpenAI-compatible
routes, a control plane with a Postgres metadata store and crash-safe deployment
jobs, a node agent that owns each GPU and supervises vLLM processes, API keys with
per-team quotas and usage tracking, and Prometheus and Grafana monitoring.

It also serves more models than fit in GPU memory, by sleeping idle models out to CPU
RAM or disk and waking them in under a second instead of reloading them in
thirty-five.

This is an implemented MVP. It runs locally on a single GPU with Linux or WSL2,
Docker and minikube.

```mermaid
flowchart TB
    C["OpenAI SDK · curl · Locust"]

    subgraph GW ["Gateway (Go)"]
        API["OpenAI routes · API keys · quotas · per-model queue · replica routing"]
        SWAP["swap controller: drain · evict · wake"]
    end

    subgraph CP ["Control plane (Go)"]
        JOBS["admin API · deploy workers (SKIP LOCKED + leases)"]
    end

    subgraph NA ["Node agent — one per GPU, the only pod holding a GPU"]
        V1["vLLM"]
        V2["vLLM"]
        V3["vLLM"]
    end

    DB[("PostgreSQL")]

    C --> API
    API --> NA
    SWAP -->|sleep / wake| NA
    API --> DB
    JOBS --> DB
    JOBS -->|start / stop| NA
```

The gateway is the request path. The control plane owns deployments. One node agent
per GPU supervises the vLLM processes that do the inference.

## Does sleep mode actually work?

This is the first thing to check, because everything else rests on it. `make m0` loads
a model, sleeps it, wakes it, and verifies both that GPU memory came back and that the
model still produces the same text:

```
  label                          device    vllm    time   note
  baseline (no server)             1318       0       -
  ACTIVE (cold start)              4244    2926   38.90s  ref=' Paris. It was founded in 789 AD by'
  SLEEPING_L1                      1549     231    1.47s  is_sleeping=True
  ACTIVE (woke from L1)            4245    2927    0.88s  out=' Paris. It was founded in 789 AD by'  OK
  SLEEPING_L2                      1549     231    0.05s  is_sleeping=True
  ACTIVE (L2 wake, no reload)      4245    2927    0.45s  out='!!!!!!!!!!!!'  GARBAGE
  ACTIVE (L2 wake + reload)        4505    3187    1.87s  out=' Paris. It was founded in 789 AD by'  OK
  ACTIVE (2nd cold start)          4244    2926   34.34s  warm page cache

  resident 2926 MiB   freed by sleep 2695 MiB (92.1%)
  wake 0.88s vs cold start 34.34s   =  39x
```

Note the `GARBAGE` line. vLLM's level-2 sleep discards the weights, and `/wake_up`
restores the allocator and the model's buffers but not the weights themselves — so it
returns `200` in 0.45s and then generates nonsense. The fix is to follow it with
`collective_rpc{"method": "reload_weights"}`, which costs 1.87s. The broken wake is
four times *faster* than the correct one, so a benchmark that only timed things would
have ranked it first. Every wake is now checked against a pre-sleep reference string.

## Quick Start

```bash
git clone https://github.com/FloareDor/kiln && cd kiln
make setup     # vLLM 0.32.0 in a venv, Go toolchain, PostgreSQL
make build     # bin/gateway, bin/controlplane, bin/nodeagent
make m0        # the check above
```

Start the three services:

```bash
make db
export KILN_DATABASE_URL="postgres://kiln:kiln@127.0.0.1:5433/kiln?sslmode=disable"

./bin/nodeagent    -addr :8081 -python "$(which python)" &
./bin/controlplane -addr :8082 -agents http://127.0.0.1:8081 -admin-token secret &
./bin/gateway      -addr :8080 -agents http://127.0.0.1:8081 -policy cost-aware &
```

Create a team, mint a key, deploy a model:

```bash
curl -sX POST localhost:8082/admin/teams -H "Authorization: Bearer secret" \
  -d '{"id":"acme","rpm":60,"tokens_per_day":1000000,"max_concurrency":8}'

KEY=$(curl -sX POST localhost:8082/admin/teams/acme/keys \
      -H "Authorization: Bearer secret" | jq -r .api_key)

curl -sX POST localhost:8082/admin/models -H "Authorization: Bearer secret" \
  -d '{"id":"qwen","hf_repo":"Qwen/Qwen2.5-0.5B-Instruct","gpu_memory_utilization":0.35}'
```

The deploy call returns a job id straight away. A worker downloads the weights, starts
vLLM, measures the model's footprint and wake costs, and marks it `READY` — about 45
seconds for a 0.5B model.

Then any OpenAI client works, changing only the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key=KEY)

for chunk in client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Count to twelve."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

On Kubernetes, `make cluster` starts minikube with GPU passthrough and the NVIDIA
device plugin, and `make up` deploys everything plus Prometheus and Grafana.

## Prerequisites

- An NVIDIA GPU with recent drivers
- Linux, or Windows with WSL2 — vLLM has no native Windows build
- Python 3.12, Go 1.24+, PostgreSQL 16+
- Docker, plus `minikube` and `kubectl` for the Kubernetes path

On Debian/Ubuntu, including inside WSL2:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv postgresql make curl git

curl -fsSLo go.tar.gz https://go.dev/dl/go1.24.0.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go.tar.gz
export PATH=$PATH:/usr/local/go/bin
```

Check `nvidia-smi` works before going further.

## Tiering

When more models are wanted than fit in VRAM, idle ones get paged out rather than
unloaded. Measured on an RTX 3070 Laptop (8 GB), vLLM 0.32.0, eager mode:

| Model | Resident VRAM | Cold start | Wake from CPU RAM | Wake from disk |
|---|---|---|---|---|
| Qwen2.5-0.5B | 2458 MiB | 35.3 s | 0.59 s | 0.71 s |
| Qwen3-0.6B | 2621 MiB | 35.9 s | 0.82 s | 1.05 s |
| Qwen2.5-1.5B | 4506 MiB | 37.4 s | 1.54 s | 1.81 s |

Cold start barely moves with model size, because it is mostly process spawn, CUDA
init and memory profiling rather than reading weights. Wake cost does scale with the
weights, so the gap narrows from 60x to 24x as models grow.

Every model moves through one state machine:

```mermaid
stateDiagram-v2
    [*] --> COLD
    COLD --> STARTING
    STARTING --> ACTIVE
    ACTIVE --> DRAINING : evict
    DRAINING --> SLEEPING_L1 : hot, host RAM free
    DRAINING --> SLEEPING_L2 : cold, or host RAM full
    DRAINING --> ACTIVE : wanted again, no swap paid
    SLEEPING_L1 --> WAKING
    SLEEPING_L2 --> WAKING
    SLEEPING_L1 --> SLEEPING_L2 : idle demote
    WAKING --> ACTIVE
```

An eviction drains first, so a model is never slept while it still has requests in
flight, and a wake never starts while the model being evicted still holds its memory.
Fifty concurrent requests for one sleeping model produce exactly one wake.

Which model gets evicted is pluggable, chosen with `-policy`: `cold` (kill the
process — the no-tiering baseline), `lru`, `lfu-decay`, `cost-aware`
(`rate × wake_cost / GB_freed`), and `predictive-prewarm`. Three guards stop
thrashing: a minimum residency time, hysteresis when freeing memory, and idle
demotion of sleepers nobody wants.

I am not claiming a ranking between the policies. Comparing them needs sustained
memory pressure, and on an 8 GB card that means running at the edge of the device,
where swap operations contend with inference — a wake measured at 0.59s idle took 81s
under load. The mechanism above is measured; the policy comparison needs a GPU with
headroom. [`docs/PROGRESS.md`](docs/PROGRESS.md) has the full reasoning.

## Routing

With several replicas of one model, round-robin scatters consecutive turns of a
conversation, so vLLM's prefix cache rarely gets reused. Agent traffic resends a long
system prompt and tool definitions every turn, so most of each prompt is shared.

Kiln hashes each prompt into cumulative leading blocks and scores replicas by how much
of that prefix they already hold, minus a penalty for how busy they are. Four
strategies, selected with `-router`: `round-robin`, `least-loaded`, `prefix-aware`,
`prefix-aware+load`. Pure affinity sends every request for a popular prefix to one
replica, which is why the load-penalised variant exists.

Replaying the same recorded agent trace against two replicas of one model:

| Router | Requests | p50 TTFT | Spread across replicas |
|---|---|---|---|
| `round-robin` | 81 | 1029 ms | 41 / 40 |
| `least-loaded` | 28 | 96 ms | 20 / 8 |

An even spread is not automatically better — affinity concentrates a conversation on
one replica on purpose, to reuse its KV cache. Run it with `make bench EXP=5`.

## Other measured behaviour

| | |
|---|---|
| Deploy to `READY`, one API call | 46 s |
| Streaming time-to-first-token vs total | 0.071 s vs 1.107 s |
| Quota, 3 rpm team | `200 200 200 429 429 429`, `Retry-After: 19` |
| Quota, 600 rpm team | six 200s, unaffected by the other tenant |
| Deploy job survives `SIGKILL` of its worker | reclaimed from its lease, done after 2 attempts |

`make m1` and `make m2` reproduce these. Raw output is in
[`docs/results/`](docs/results).

## Why the node agent owns the GPU

Kubernetes gives a pod exclusive use of a GPU for the pod's whole lifetime, with no
way to release it and keep running. One Deployment per model would therefore pin one
model to each GPU permanently, and tiering would be impossible to express. So a single
long-lived agent holds the GPU and runs vLLM as child processes, starting, sleeping,
waking and stopping them underneath itself. The same binary runs standalone under
Docker with no Kubernetes at all.

Deployment jobs are claimed with `SELECT ... FOR UPDATE SKIP LOCKED` so several
control planes can drain one queue without blocking each other, plus a lease, because
the row lock dies with its transaction while a deploy takes minutes. A crashed
worker's lease expires and another worker picks the job up.

[`docs/DESIGN.md`](docs/DESIGN.md) goes into the rest, including where the OS
memory-hierarchy analogy stops holding.

## API

Inference routes take `Authorization: Bearer <api key>`:

```
POST /v1/chat/completions     OpenAI-compatible, stream: true supported
POST /v1/completions          OpenAI-compatible legacy completions
GET  /v1/models               deployed models, state, ready replica count
GET  /health                  no auth
GET  /metrics                 Prometheus, no auth
```

Admin routes take `Authorization: Bearer <admin token>`:

```
POST   /admin/models              deploy a model, returns a job
GET    /admin/models              list deployments
GET    /admin/models/{id}         one deployment, its replicas and live job
DELETE /admin/models/{id}         tear down through a job
POST   /admin/models/{id}/scale   change replica count
POST   /admin/models/{id}/pin     pin so it is never evicted
POST   /admin/teams               create a team with its quota
POST   /admin/teams/{id}/keys     mint an API key, shown once
DELETE /admin/keys/{id}           revoke a key
GET    /admin/jobs                recent deployment jobs
GET    /admin/usage?since=24h     token usage per team and model
```

The node agent's own API is internal: `/status`, `POST /replicas`,
`/replicas/{id}/sleep`, `/replicas/{id}/wake`, `DELETE /replicas/{id}`.

## Make targets

| Target | Purpose |
|---|---|
| `make setup` | Installs vLLM, the Go toolchain and PostgreSQL |
| `make build` | Builds the three binaries |
| `make db` | Starts PostgreSQL and applies migrations |
| `make test` | Unit tests — no GPU, database or network |
| `make test-all` | Adds the PostgreSQL integration suites |
| `make m0` / `m1` / `m2` | The end-to-end checks described above |
| `make cluster` / `up` / `down` | Local Kubernetes |
| `make traces` / `bench EXP=n` | Benchmarks |

## Monitoring

Both the gateway and the node agent expose Prometheus metrics, and
[`dashboards/kiln.json`](dashboards/kiln.json) is a Grafana dashboard in four rows:
client latency, GPU and host memory, scheduler activity, routing.

Beyond the usual latency and throughput series, the ones worth watching are
`kiln_swap_duration_seconds` (labelled from-tier and to-tier),
`kiln_residency_hits_total` against `kiln_residency_misses_total` for how often a
request found its model already loaded, and `kiln_thrash_total`, which counts
evictions inside the minimum residency window and should always be zero.

After `make up`, Grafana is on port 30300 (`admin`/`admin`), Prometheus on 30090.

## Tests

```bash
make test      # unit tests, offline
make test-all  # adds the PostgreSQL suites
```

The default suite needs no GPU, database or network, because a suite that needs a GPU
stops being run. The scheduler and routers are written as pure functions — state in,
decisions out — so their logic is covered without hardware. The PostgreSQL suites skip
unless `KILN_TEST_DATABASE_URL` is set, and each migrates into its own schema so
packages can run in parallel without truncating each other's tables.

## Troubleshooting

**`cudaGetDeviceCount() Error 500: named symbol not found` in a pod.** GPU passthrough
is not reaching the container. Under WSL2 there is no `/dev/nvidia*` at all — the GPU
is paravirtualised through `/dev/dxg` and the libraries in `/usr/lib/wsl/lib`, which
minikube's node image does not pass into pods. `nvidia-smi` still works inside the pod,
which makes this look like a CUDA version mismatch instead. Run the node agent on the
host; the Kubernetes GPU path works on a normal Linux node.

**`No available memory for the cache blocks`.** `gpu_memory_utilization` is too low. It
caps the weights and the KV cache arena together, so a 1.5B model in bf16 needs about
0.55 of an 8 GB card, not 0.4.

**A model stays in `STARTING`.** Check the per-replica vLLM log — the path is in the
error message. A gated Hugging Face repo is the usual cause; `meta-llama/*` returns 403
without a token.

**GPU memory full but nothing deployed.** An interrupted run left orphaned vLLM
processes. They are spawned into their own process group so the agent can signal the
whole tree, but if the agent is killed abruptly they survive holding their allocation.
Run `bash scripts/gpu_reclaim.sh`.

**503 `model_unavailable`.** The gateway logs the reason: usually the model is not
deployed, every eviction candidate is inside its minimum residency window, or the model
is larger than the whole VRAM budget.

**Shell scripts failing with `set: pipefail: invalid option name` on Windows.** Line
endings. The repo sets `* text eol=lf`; a script checked out with CRLF makes bash read
a trailing `\r` on every line.

## Configuration

The gateway takes `-agents`, `-database-url`, `-router`, `-policy`, `-gpu-bytes`,
`-gpu-reserve-bytes`, `-min-residency`, `-drain-deadline`, `-max-queue` and
`-queue-timeout`. Without `-database-url` it runs open, with no auth, quotas or
tiering, which is useful for benchmarking. Without `-policy` tiering is off.

The control plane takes `-database-url` (required), `-admin-token` (empty disables the
admin API), `-workers` and `-measure-wakes`.

The node agent takes `-python`, `-advertise-host` (the pod IP under Kubernetes, since
replicas are advertised to other pods), `-hf-home` and `-start-timeout`.

Each flag has a `KILN_`-prefixed environment variable; run any binary with `-h` for
the full list and defaults.

## Layout

```
cmd/          gateway, controlplane, nodeagent
internal/
  api/        HTTP handlers, SSE proxy, admin API, middleware
  auth/       hashed API keys, token-bucket quotas
  routing/    four routers, cumulative prefix hashing
  scheduler/  state machine, VRAM accounting, five eviction policies
  swap/       the tiering executor: drain, evict, wake
  jobs/       deployment workers, SKIP LOCKED + leases
  nodeagent/  vLLM process supervision, sleep/wake, GPU probe
  store/      PostgreSQL via pgx, embedded migrations
bench/        trace generator, Locust harness, experiment runner
deploy/       Kubernetes manifests, Dockerfiles
docs/         DESIGN.md, PROGRESS.md, raw results
```

## Limits

A single GPU is the tested configuration; cross-node placement just sorts agents by
free VRAM. Host RAM is the real ceiling on the CPU-RAM tier, since VRAM decides how
many models can be active but host RAM decides how many stay warm. The Kubernetes GPU
path is untested on WSL2 for the reason above — every component comes up, but GPU
workloads need a normal Linux node. The admin API uses one shared token and is meant
to be reachable only from inside the cluster.
