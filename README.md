# Kiln Inference Serving

Kiln is a self-hosted inference serving platform for deploying Hugging Face LLMs as
vLLM workers on Kubernetes. It includes a Go API gateway with OpenAI-compatible
routes, a control plane with a Postgres metadata store and crash-safe deployment
jobs, a node agent that owns each GPU and supervises vLLM processes, API keys with
per-team quotas and usage tracking, and Prometheus and Grafana monitoring.

It also serves more models than fit in GPU memory, by sleeping idle models out to CPU
RAM or disk and waking them in under a second instead of reloading them in
thirty-five.

The current project is an implemented MVP. It is intended to run locally on a single
GPU with Linux or WSL2, Docker, and minikube, with a later path to multi-node
clusters and to Modal for larger GPUs.

![Kiln logo](docs/img/logo.svg)

![Kiln system design](docs/img/architecture.svg)

## What It Does

Kiln lets you:

- Deploy a Hugging Face model with a single API call.
- Send OpenAI-compatible `/v1/chat/completions` and `/v1/completions` requests, with
  streaming.
- List deployed models and their state.
- Create teams, issue API keys, and set per-team limits on requests per minute,
  tokens per day, and concurrent requests.
- Track token usage per team and per model, including for streaming responses.
- Scale a model to several replicas and route across them four different ways.
- Run deployment jobs from several control planes at once, and recover a job whose
  worker was killed mid-deploy.
- Serve more models than fit in VRAM, by sleeping idle ones to CPU RAM or disk.
- Watch latency, throughput, queue depth, GPU memory and swap activity in Grafana.

The Go gateway is the request path and router. The control plane owns deployments.
One node agent per GPU supervises the vLLM processes that do the actual inference.

## Prerequisites

Install these before running the project locally:

- An NVIDIA GPU with recent drivers
- Linux, or Windows with WSL2 (vLLM has no native Windows build)
- Python 3.12
- Go 1.24+
- PostgreSQL 16+
- Docker, plus `minikube` and `kubectl` for the Kubernetes path

Debian/Ubuntu, including inside WSL2:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv postgresql make curl git

# Go
curl -fsSLo go.tar.gz https://go.dev/dl/go1.24.0.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go.tar.gz
export PATH=$PATH:/usr/local/go/bin

# kubectl and minikube
curl -fsSLo kubectl "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/
curl -fsSLo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube /usr/local/bin/
```

Confirm the GPU is visible:

```bash
nvidia-smi
```

## Quick Start

```bash
git clone https://github.com/FloareDor/kiln && cd kiln
make setup
make build
make m0
```

`make setup` does the full local setup:

- Creates a Python 3.12 virtualenv and installs vLLM 0.32.0 into it.
- Installs a current Go toolchain.
- Installs PostgreSQL, creates the `kiln` and `kiln_test` databases, and prints their
  connection strings.

`make build` produces `bin/gateway`, `bin/controlplane` and `bin/nodeagent`.

`make m0` checks that vLLM sleep mode actually frees GPU memory and restores a working
model. Run it before anything else.

### Start the services

```bash
make db
export KILN_DATABASE_URL="postgres://kiln:kiln@127.0.0.1:5433/kiln?sslmode=disable"

./bin/nodeagent    -addr :8081 -python "$(which python)" &
./bin/controlplane -addr :8082 -agents http://127.0.0.1:8081 -admin-token secret &
./bin/gateway      -addr :8080 -agents http://127.0.0.1:8081 -policy cost-aware &
```

### Create a team and an API key

```bash
curl -sX POST localhost:8082/admin/teams \
  -H "Authorization: Bearer secret" \
  -d '{"id":"acme","rpm":60,"tokens_per_day":1000000,"max_concurrency":8}'

KEY=$(curl -sX POST localhost:8082/admin/teams/acme/keys \
      -H "Authorization: Bearer secret" | jq -r .api_key)
```

The key is shown once. Only a hash is stored.

### Deploy a model

```bash
curl -sX POST localhost:8082/admin/models \
  -H "Authorization: Bearer secret" \
  -d '{"id":"qwen","hf_repo":"Qwen/Qwen2.5-0.5B-Instruct","gpu_memory_utilization":0.35}'
```

This returns a job id immediately. A worker downloads the weights, starts vLLM,
measures the model's GPU footprint and wake costs, and marks it `READY`. Check on it
with:

```bash
curl -s localhost:8082/admin/models/qwen -H "Authorization: Bearer secret" | jq
```

### Send a request

Any OpenAI client works. Only the base URL changes:

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

### Kubernetes

```bash
make cluster
make up
```

`make cluster` starts minikube with GPU passthrough and the NVIDIA device plugin.
`make up` builds the images, loads them into the cluster, and deploys the stack with
Prometheus and Grafana.

## Make Commands

| Target | Purpose |
|---|---|
| `make setup` | Installs vLLM, the Go toolchain and PostgreSQL. |
| `make build` | Builds the three binaries into `bin/`. |
| `make db` | Starts PostgreSQL and applies migrations. |
| `make test` | Runs unit tests. No GPU, database or network needed. |
| `make test-all` | Adds the PostgreSQL integration suites. |
| `make cover` | Prints a test coverage summary. |
| `make vet` | Runs `go vet`. |
| `make m0` | Verifies vLLM sleep mode frees and restores GPU memory. |
| `make m1` | End-to-end test: the OpenAI SDK streams from the platform. |
| `make m2` | End-to-end test: one-call deploy, quotas, crash-safe jobs. |
| `make images` | Builds the three container images. |
| `make cluster` | Starts a local GPU-enabled minikube cluster. |
| `make up` | Deploys the stack plus Prometheus and Grafana. |
| `make down` | Removes the stack. |
| `make logs` | Tails node agent logs. |
| `make traces` | Regenerates the benchmark traces. |
| `make bench EXP=5` | Runs a benchmark experiment. |
| `make clean` | Removes build output. |

## How It Works

Three processes:

- **Node agent.** Owns one GPU and runs vLLM as child processes. Kubernetes gives a
  pod exclusive use of a GPU for the pod's whole lifetime, so one Deployment per model
  would pin one model to each GPU permanently. A long-lived supervisor that starts,
  sleeps, wakes and stops models underneath it is what lets a GPU be shared. The same
  binary runs standalone under Docker with no Kubernetes.
- **Control plane.** Owns deployments: what models exist, what they cost, where they
  are placed. Runs the deployment job workers.
- **Gateway.** The request path. Authenticates, applies quotas, queues, routes to a
  replica, streams the response back, and decides which models are resident in VRAM.

The gateway keeps serving when the control plane is down. It polls node agents for its
view of the fleet rather than reading the database, so a control-plane outage stops new
deployments but not inference.

### Deployment jobs

Deploying takes minutes and has to survive its worker being killed. Jobs are claimed
with `SELECT ... FOR UPDATE SKIP LOCKED` so several control planes can drain one queue
without blocking each other, plus a lease, because the row lock dies with its
transaction while the work does not. A crashed worker's lease expires and another
worker picks the job up. A slow worker renews it.

### GPU memory tiering

When more models are wanted than fit in VRAM, Kiln pages idle ones out instead of
unloading them.

![How tiering works](docs/img/tiering.svg)

Every model moves through one state machine. An eviction drains first, so a model is
never slept while it still has requests in flight, and a wake never starts while the
model being evicted still holds its memory.

![Model state machine](docs/img/state-machine.svg)

Which model is evicted is a pluggable policy, selected with `-policy`:

| Policy | Evicts the model with the lowest |
|---|---|
| `cold` | last-request time, and kills the process. The no-tiering baseline. |
| `lru` | last-request time |
| `lfu-decay` | exponentially decayed arrival rate |
| `cost-aware` | `rate × wake_cost / GB_freed` |
| `predictive-prewarm` | same as cost-aware, and wakes models whose demand is rising |

Three guards prevent thrashing: a minimum residency time before a model can be
evicted again, hysteresis so freeing memory does not leave the GPU exactly full, and
idle demotion so a sleeper nobody wants gives back its host RAM. Fifty concurrent
requests for one sleeping model produce exactly one wake.

### Routing across replicas

With several replicas of one model, round-robin scatters consecutive turns of a
conversation, so vLLM's prefix cache rarely gets reused. Agent traffic resends a long
system prompt and tool definitions every turn, so most of each prompt is shared.

Kiln hashes each prompt into cumulative leading blocks and scores replicas by how much
of that prefix they already hold, minus a penalty for how busy they are. Four
strategies: `round-robin`, `least-loaded`, `prefix-aware`, and `prefix-aware+load`.
Pure affinity sends every request for a popular prefix to one replica, which is why
the load-penalised variant exists.

## Results

Measured on an RTX 3070 Laptop (8 GB) under WSL2, vLLM 0.32.0, eager mode. Raw
per-request output and the conditions for each run are in
[`docs/results/`](docs/results). The figures are generated from those files by
`python scripts/make_figures.py`.

### Waking a model vs loading it from scratch

![Wake versus cold start](docs/img/results-wake-vs-cold.svg)

| Model | Resident VRAM | Cold start | Wake from CPU RAM | Wake from disk |
|---|---|---|---|---|
| Qwen2.5-0.5B | 2458 MiB | 35.3 s | 0.59 s (59.7×) | 0.71 s (50.0×) |
| Qwen3-0.6B | 2621 MiB | 35.9 s | 0.82 s (43.7×) | 1.05 s (34.3×) |
| Qwen2.5-1.5B | 4506 MiB | 37.4 s | 1.54 s (24.3×) | 1.81 s (20.6×) |

Sleeping frees 92% of a model's resident VRAM and the woken model produces identical
output. Cold start barely changes with model size because it is dominated by process
spawn, CUDA initialisation and memory profiling rather than by reading weights.

Reproduce with `make m0` and `make bench EXP=2`.

### Platform

| | |
|---|---|
| Deploy to `READY`, one API call | 46 s |
| Streaming time-to-first-token vs total | 0.071 s vs 1.107 s |
| Quota enforcement, 3 rpm team | `200 200 200 429 429 429`, `Retry-After: 19` |
| Quota enforcement, 600 rpm team | six 200s, unaffected by the other tenant |
| Deploy job survives `SIGKILL` of its worker | reclaimed and finished after 2 attempts |

Reproduce with `make m1` and `make m2`.

### Routing

![Router comparison](docs/img/results-routing.svg)

Run with `make bench EXP=5`. Every arm replays the same recorded trace from
`bench/traces/`, so the only variable is the router.

The eviction policy comparison is not included. It needs sustained memory pressure,
and on an 8 GB card that means running at the edge of the device, where swap
operations contend with inference and a wake measured at 0.59 s idle took 81 s under
load. The tiering mechanism above is measured; which policy wins under load needs a
GPU with headroom. See [`docs/PROGRESS.md`](docs/PROGRESS.md).

## HTTP API

Inference routes need `Authorization: Bearer <api key>`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat, `stream: true` supported |
| `POST` | `/v1/completions` | OpenAI-compatible legacy completions |
| `GET` | `/v1/models` | Deployed models, state, ready replica count |
| `GET` | `/health` | Liveness. No auth. |
| `GET` | `/metrics` | Prometheus metrics. No auth. |

Admin routes need `Authorization: Bearer <admin token>`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/admin/models` | Deploy a model. Returns a job. |
| `GET` | `/admin/models` | List deployments |
| `GET` | `/admin/models/{id}` | One deployment, its replicas and live job |
| `DELETE` | `/admin/models/{id}` | Tear a deployment down through a job |
| `POST` | `/admin/models/{id}/scale` | Change replica count |
| `POST` | `/admin/models/{id}/pin` | Pin a model so it is never evicted |
| `POST` | `/admin/teams` | Create a team with its quota |
| `GET` | `/admin/teams` | List teams and quotas |
| `POST` | `/admin/teams/{id}/keys` | Mint an API key. Shown once. |
| `DELETE` | `/admin/keys/{id}` | Revoke a key |
| `GET` | `/admin/jobs` | Recent deployment jobs |
| `GET` | `/admin/usage?since=24h` | Token usage per team and model |

Node agent routes are internal, one agent per GPU.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/status` | GPUs, host memory, and every replica on this node |
| `POST` | `/replicas` | Start a replica. Idempotent. |
| `POST` | `/replicas/{id}/sleep` | Sleep to CPU RAM (`level: 1`) or disk (`level: 2`) |
| `POST` | `/replicas/{id}/wake` | Wake, including the weight reload a level-2 wake needs |
| `DELETE` | `/replicas/{id}` | Stop the process and release its GPU memory |

## Monitoring

The gateway and node agent expose Prometheus metrics.
[`dashboards/kiln.json`](dashboards/kiln.json) is a Grafana dashboard with four rows:
client latency, GPU and host memory, scheduler activity, and routing.

| Metric | Meaning |
|---|---|
| `kiln_ttft_seconds` | Time to first token, by model, team and replica |
| `kiln_requests_total` | Requests by status class |
| `kiln_tokens_total` | Prompt and completion tokens, by team |
| `kiln_queue_depth`, `kiln_queue_wait_seconds` | Saturation, and where a wake appears to a client |
| `kiln_quota_rejections_total` | 429s, labelled by which limit was hit |
| `kiln_gpu_memory_bytes`, `kiln_host_memory_bytes` | Device and host memory |
| `kiln_models_by_state` | How many models are resident, asleep, or loading |
| `kiln_swap_duration_seconds` | Transition cost, labelled from-tier and to-tier |
| `kiln_evictions_total`, `kiln_wakes_total`, `kiln_cold_starts_total` | Swap activity |
| `kiln_residency_hits_total`, `kiln_residency_misses_total` | How often a model was already loaded |
| `kiln_thrash_total` | Evictions inside the minimum residency window. Should be zero. |

After `make up`, Grafana is on port 30300 (`admin` / `admin`) and Prometheus on 30090.

## Kubernetes Model

| Object | Kind | Purpose |
|---|---|---|
| `kiln-nodeagent` | DaemonSet | One per GPU node. The only pod that requests `nvidia.com/gpu`. |
| `kiln-gateway` | Deployment, 2 replicas | Stateless request path |
| `kiln-controlplane` | Deployment, 2 replicas | Two on purpose; the job queue is built for it |
| `kiln-postgres` | StatefulSet | Keys and usage outlive a pod restart |
| `kiln-hf-cache` | PVC | Shared weight cache, which makes a disk-tier wake cheap |
| `prometheus`, `grafana` | Deployments | Monitoring |

The node agent has no liveness probe and a 180 second termination grace period. A cold
start can block long enough to look dead to a prober, and restarting mid-load orphans a
vLLM process that still holds the GPU. `/dev/shm` is mounted as an 8 GiB memory-backed
volume, because vLLM's engine core talks to its workers over shared memory and the
Docker default of 64 MB causes a bus error during startup.

## Tests

```bash
make test      # unit tests: no GPU, database or network
make test-all  # adds the PostgreSQL integration suites
```

The default suite is offline and GPU-free. The scheduler and routers are written as
pure functions, so their logic is covered without hardware. The PostgreSQL suites skip
unless `KILN_TEST_DATABASE_URL` is set, and each migrates into its own schema so
packages can run in parallel.

## Troubleshooting

### `cudaGetDeviceCount() Error 500: named symbol not found` inside a pod

GPU passthrough is not reaching the container. Under WSL2 there is no `/dev/nvidia*`;
the GPU is paravirtualised through `/dev/dxg` and the libraries in `/usr/lib/wsl/lib`,
which minikube's node image does not pass into pods. `nvidia-smi` still works inside
the pod, which makes this look like a CUDA version mismatch. Run the node agent
directly on the host instead. The Kubernetes GPU path works on a normal Linux node.

### `No available memory for the cache blocks`

`gpu_memory_utilization` is too low. It caps the weights and the KV cache arena
together, so a 1.5B model in bf16 needs about 0.55 of an 8 GB card, not 0.4.

### `OSError: [Errno 98] Address already in use` when a model restarts

The previous process has not released its port. Kiln takes a fresh port when
respawning a replica, so this means something outside Kiln is holding it.

### A model stays in `STARTING`

Check the per-replica vLLM log; its path is in the error message. A gated Hugging Face
repo is the usual cause, since `meta-llama/*` returns 403 without a token. Set
`HF_TOKEN` or use an openly licensed model.

### GPU memory is full but nothing is deployed

An interrupted run left orphaned vLLM processes:

```bash
bash scripts/gpu_reclaim.sh
```

### Requests return 503 with `model_unavailable`

The gateway logs the reason. Usually the model is not deployed, every eviction
candidate is inside its minimum residency window, or the model is larger than the whole
VRAM budget.

### Shell scripts fail with odd syntax errors on Windows

Line endings. The repo sets `* text eol=lf` in `.gitattributes`. A script checked out
with CRLF makes bash read a trailing `\r` on every line.

## Configuration

Gateway:

| Flag / env | Default | Purpose |
|---|---|---|
| `-addr` / `KILN_GATEWAY_ADDR` | `:8080` | Listen address |
| `-agents` / `KILN_AGENTS` | `http://127.0.0.1:8081` | Node agent URLs, comma-separated |
| `-database-url` / `KILN_DATABASE_URL` | — | Postgres DSN. Empty disables auth, quotas and tiering. |
| `-router` / `KILN_ROUTER` | `round-robin` | `round-robin`, `least-loaded`, `prefix-aware`, `prefix-aware+load` |
| `-policy` / `KILN_POLICY` | — | Eviction policy. Empty disables tiering. |
| `-gpu-bytes` | 8 GiB | Device VRAM the scheduler budgets against |
| `-gpu-reserve-bytes` | 256 MiB | Margin for allocator fragmentation |
| `-min-residency` | `15s` | How long a model must serve before it can be evicted |
| `-drain-deadline` | `10s` | How long to wait for in-flight requests before sleeping a model |
| `-max-queue` | 256 | Per-model queue depth before shedding load |
| `-queue-timeout` | `180s` | How long a request waits for its model to become servable |

Control plane:

| Flag / env | Default | Purpose |
|---|---|---|
| `-addr` / `KILN_CP_ADDR` | `:8082` | Listen address |
| `-database-url` / `KILN_DATABASE_URL` | — | Postgres DSN. Required. |
| `-admin-token` / `KILN_ADMIN_TOKEN` | — | Shared admin token. Empty disables the admin API. |
| `-workers` | 2 | Deployment job workers in this process |
| `-measure-wakes` | `true` | Measure sleep and wake costs at deploy time |

Node agent:

| Flag / env | Default | Purpose |
|---|---|---|
| `-addr` / `KILN_AGENT_ADDR` | `:8081` | Listen address |
| `-python` / `KILN_PYTHON` | `python` | Interpreter that has vLLM installed |
| `-advertise-host` / `KILN_ADVERTISE_HOST` | `127.0.0.1` | Host others use to reach replicas. The pod IP under Kubernetes. |
| `-hf-home` / `HF_HOME` | — | Shared weight cache |
| `-start-timeout` | `15m` | How long a cold start may take |

## Repository Layout

```
cmd/
  gateway/           OpenAI-compatible API, auth, quotas, routing, residency
  controlplane/      Admin API and deployment job workers
  nodeagent/         Owns one GPU, supervises vLLM processes
internal/
  api/               HTTP handlers, SSE proxy, admin API, middleware
  auth/              Hashed API keys, token-bucket quotas
  routing/           Four routers, cumulative prefix hashing
  scheduler/         State machine, VRAM accounting, five eviction policies
  swap/              The tiering executor: drain, evict, wake
  jobs/              Deployment workers using SKIP LOCKED and leases
  nodeagent/         vLLM process supervision, sleep and wake, GPU probe
  store/             PostgreSQL via pgx, embedded migrations
  metrics/           Prometheus instrumentation
bench/               Trace generator, Locust harness, experiment runner
deploy/
  k8s/               Manifests for the stack and monitoring
  docker/            Dockerfiles
dashboards/          Grafana dashboard JSON
docs/
  DESIGN.md          Why it is built this way and where the tradeoffs are
  PROGRESS.md        What was measured, and the bugs found measuring it
  results/           Raw benchmark output and run conditions
scripts/             Setup, end-to-end tests, figure generation
```

## Notes And Limits

- A single GPU is the tested configuration. Placement across nodes sorts agents by
  free VRAM.
- Host RAM is the real ceiling on the CPU-RAM tier. VRAM decides how many models can
  be active; host RAM decides how many can be kept warm.
- The Kubernetes GPU path is untested on WSL2, for the reason under Troubleshooting.
  Every component comes up; GPU workloads need a normal Linux node.
- The eviction policy comparison is not claimed, for the reason under Results.
- The admin API uses a single shared token and is meant to be reachable only from
  inside the cluster.

[`docs/DESIGN.md`](docs/DESIGN.md) covers the design argument and the tradeoffs.
[`docs/PROGRESS.md`](docs/PROGRESS.md) covers what was measured and the bugs that
measuring it exposed.
