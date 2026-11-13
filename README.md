# Kiln

Kiln is a self-hosted inference serving platform for deploying Hugging Face LLMs as
vLLM workers on Kubernetes. It includes a Go API gateway with OpenAI-compatible
routes, a control plane with a Postgres metadata store and crash-safe deployment
jobs, a per-GPU node agent that supervises vLLM processes, API keys with per-team
quotas and usage tracking, and Prometheus/Grafana monitoring.

It is an implemented MVP, intended to run locally on a single GPU with Docker, WSL2
or Linux, and minikube — with a path to multi-node clusters and to Modal for larger
GPUs.

![Kiln](docs/img/logo.svg)

![Kiln system design](docs/img/architecture.svg)

---

## Why This Exists

If you want to run LLMs yourself instead of paying a hosted API per token, the model
server is the easy part. `vllm serve` gets you one model on one GPU. Everything
around it is what actually takes the time:

- **Deploying** a model should be one API call, not a hand-written pod spec.
- **The API** has to be OpenAI-compatible, or every client you already have breaks.
- **Several teams** will share it, so you need API keys, per-team rate limits, and a
  record of who used what.
- **Deployments must not corrupt each other** when two run at once, or when the
  process doing one is killed halfway through.
- **You need to see** latency, throughput, queue depth and GPU memory, or you cannot
  tell a slow model from an overloaded one.
- **GPUs are expensive**, so one GPU should serve as many models and requests as it
  can.

Kiln is those six things. The last one is where it goes deepest: it can serve more
models than fit in VRAM at once by paging idle ones out to CPU RAM or disk and
bringing them back in under a second.

---

## What It Does

Kiln lets you:

- Deploy a Hugging Face model with a single `POST /admin/models` call.
- Send OpenAI-compatible `/v1/chat/completions` and `/v1/completions` requests, with
  server-sent-event streaming.
- List deployed models and their state through `/v1/models`.
- Create teams, issue API keys, and set per-team limits on requests per minute,
  tokens per day, and concurrent requests.
- Track token usage per team and per model, including for streaming responses.
- Scale a model to several replicas and route across them with a choice of four
  routing strategies.
- Run deployment jobs safely from several control-plane replicas at once, and
  recover a job whose worker was killed mid-deploy.
- Serve more models than fit in GPU memory, by sleeping idle ones to CPU RAM or disk.
- Watch latency, throughput, queue depth, GPU memory and swap activity in Grafana.

The Go gateway is the request path. The control plane owns deployments. One node
agent per GPU supervises the vLLM processes that do the actual inference.

---

## Prerequisites

- An NVIDIA GPU with recent drivers
- Linux, or Windows with WSL2 (vLLM has no native Windows build)
- Python 3.12
- Go 1.24 or newer
- PostgreSQL 16+
- Docker, and `minikube` + `kubectl` for the Kubernetes path

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

Check the GPU is visible before going further:

```bash
nvidia-smi
```

---

## Quick Start

```bash
git clone https://github.com/FloareDor/kiln && cd kiln
make setup     # pins vLLM 0.32.0 in a venv, installs the Go toolchain and Postgres
make build     # builds gateway, controlplane and nodeagent into bin/
make m0        # verifies that vLLM sleep mode frees and restores GPU memory
```

`make setup` does the full local setup:

- Creates a Python 3.12 virtualenv and installs a pinned vLLM into it.
- Installs a modern Go toolchain.
- Installs PostgreSQL, creates the `kiln` and `kiln_test` databases, and prints the
  connection strings.

`make m0` is a gate, not a demo. It loads a small model, puts it to sleep, wakes it,
and checks both that GPU memory actually came back and that the model still produces
the same text. Run it before trusting anything else.

### Start the stack locally

```bash
make db        # starts Postgres and applies migrations
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
echo "$KEY"   # sk-kiln-... shown once and never stored in plaintext
```

### Deploy a model

```bash
curl -sX POST localhost:8082/admin/models \
  -H "Authorization: Bearer secret" \
  -d '{"id":"qwen","hf_repo":"Qwen/Qwen2.5-0.5B-Instruct","gpu_memory_utilization":0.35}'
```

This returns immediately with a job id. A worker downloads the weights, starts vLLM,
measures how much GPU memory the model takes and how long it takes to wake, then
marks it `READY`. Watch it with:

```bash
curl -s localhost:8082/admin/models/qwen -H "Authorization: Bearer secret" | jq
```

### Send a request

Any OpenAI client works — only the base URL changes:

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

### Or on Kubernetes

```bash
make cluster   # minikube with GPU passthrough and the NVIDIA device plugin
make up        # builds images, loads them, deploys everything + Prometheus/Grafana
```

---

## Make Commands

| Target | Purpose |
|---|---|
| `make setup` | One-time: installs the pinned vLLM, the Go toolchain and PostgreSQL. |
| `make build` | Builds `gateway`, `controlplane` and `nodeagent` into `bin/`. |
| `make db` | Starts PostgreSQL and applies migrations. |
| `make test` | Unit tests only — no GPU, no database, no network. |
| `make test-all` | Adds the PostgreSQL integration suites. |
| `make cover` | Test coverage summary. |
| `make vet` | Runs `go vet`. |
| `make m0` | Verifies vLLM sleep mode frees and restores GPU memory. |
| `make m1` | End-to-end: the OpenAI SDK streams from the platform. |
| `make m2` | End-to-end: one-call deploy, per-team quotas, crash-safe jobs. |
| `make images` | Builds the three container images. |
| `make cluster` | Starts a local GPU-enabled minikube cluster. |
| `make up` | Deploys the whole stack plus Prometheus and Grafana. |
| `make down` | Removes the stack. |
| `make logs` | Tails node agent logs. |
| `make traces` | Regenerates the fixed benchmark traces. |
| `make bench EXP=5` | Runs a benchmark experiment. |
| `make clean` | Removes build output. |

---

## How It Works

Three processes, each with one job.

**The node agent** owns a GPU and runs vLLM as child processes. This is the decision
everything else follows from: Kubernetes gives a pod exclusive use of a GPU for the
pod's *entire lifetime*, with no way to release it and keep running. One Deployment
per model would therefore pin exactly one model to each GPU forever. A long-lived
supervisor that starts, sleeps, wakes and stops models underneath it is what makes
sharing a GPU possible at all. The same binary runs standalone under plain Docker,
with no Kubernetes, which is how it runs on Modal.

**The control plane** owns deployments: what models exist, what they cost, where they
are placed. It runs the deployment job workers.

**The gateway** is the request path. It authenticates, applies quotas, queues, routes
to a replica, and streams the response back. It also owns GPU *residency* — which
models are in VRAM right now — because deciding that needs the in-flight request
count, and the gateway is the only component that knows it exactly and instantly.

A useful consequence: the gateway keeps serving when the control plane is down. It
polls node agents for its view of the fleet rather than reading the database, so a
control-plane outage stops new deployments but does not stop inference.

### Deployment jobs

Deploying takes minutes and must survive the worker doing it being killed. Jobs are
claimed with `SELECT ... FOR UPDATE SKIP LOCKED`, so several control planes can drain
one queue without blocking each other, plus a **lease** — because the row lock dies
with its transaction while the work takes minutes. A crashed worker's lease simply
expires and another worker picks the job up; a slow worker renews it.

One detail that matters: the attempt counter increments when a job is *claimed*, not
when it fails. A worker killed mid-job never reports a failure, so counting only
reported failures would let a job that reliably crashes its worker retry forever.

### GPU memory tiering

When more models are wanted than fit in VRAM, Kiln pages the idle ones out instead of
unloading them.

![How tiering works](docs/img/tiering.svg)

Every model moves through one state machine. An eviction always drains first — a
model is never put to sleep while it still has requests in flight — and a wake never
begins while a victim still holds its memory.

![Model state machine](docs/img/state-machine.svg)

Which model gets evicted is a pluggable policy, chosen with `-policy`:

| Policy | Evicts the model with the lowest… |
|---|---|
| `cold` | last-request time, and kills the process — the no-tiering baseline |
| `lru` | last-request time |
| `lfu-decay` | exponentially decayed arrival rate |
| `cost-aware` | `rate × wake_cost / GB_freed` |
| `predictive-prewarm` | same, plus wakes models whose demand is rising |

Three guards stop it thrashing: a **minimum residency** so a just-woken model cannot
be evicted immediately, **hysteresis** so freeing memory does not leave the GPU
exactly full, and **idle demotion** so a sleeper nobody wants gives back its host RAM.

Fifty concurrent requests for one sleeping model produce exactly **one** wake; the
other forty-nine wait on it.

### Routing across replicas

With several replicas of one model, round-robin scatters consecutive turns of a
conversation, so vLLM's prefix cache rarely gets reused. Agent traffic — a long
system prompt plus tool definitions resent every turn — is mostly shared prefix.

Kiln hashes each prompt into cumulative leading blocks and scores replicas by how
much of that prefix they already hold, minus a penalty for how busy they are. Four
strategies: `round-robin`, `least-loaded`, `prefix-aware`, `prefix-aware+load`.

> [!NOTE]
> Pure prefix affinity creates hot spots: every request for a popular prefix lands on
> one replica while its peers idle. That is why `prefix-aware+load` exists, and why
> both are kept — the failure mode is worth showing rather than asserting.

---

## Results

Measured on an **RTX 3070 Laptop (8 GB)** under WSL2, vLLM 0.32.0, eager mode. Raw
per-request rows and the exact conditions for each run are in
[`docs/results/`](docs/results). Figures are regenerated from those files by
`python scripts/make_figures.py`, so they cannot drift from the measurement.

### Waking a model vs loading it from scratch

![Wake versus cold start](docs/img/results-wake-vs-cold.svg)

| model | resident VRAM | cold start | wake from CPU RAM | wake from disk |
|---|---|---|---|---|
| Qwen2.5-0.5B | 2458 MiB | 35.3 s | **0.59 s** (59.7×) | **0.71 s** (50.0×) |
| Qwen3-0.6B | 2621 MiB | 35.9 s | **0.82 s** (43.7×) | **1.05 s** (34.3×) |
| Qwen2.5-1.5B | 4506 MiB | 37.4 s | **1.54 s** (24.3×) | **1.81 s** (20.6×) |

Sleeping frees **92%** of a model's resident VRAM, and the woken model produces
byte-identical output.

Cold start barely changes with model size because it is dominated by process spawn,
CUDA initialisation and memory profiling rather than by reading weights. Wake cost
does scale with the weights, so the advantage narrows as models grow — from 60× to
24× — while staying more than an order of magnitude either way.

### Platform

| | |
|---|---|
| Deploy to `READY`, one API call | 46 s |
| Streaming time-to-first-token vs total | **0.071 s** vs 1.107 s |
| Quota enforcement, 3 rpm team | `200 200 200 429 429 429`, `Retry-After: 19` |
| Quota enforcement, 600 rpm team | six 200s, unaffected by the other tenant |
| Deploy job survives `SIGKILL` of its worker | reclaimed from expired lease, done after 2 attempts |

Reproduce with `make m1` and `make m2`.

### Routing

![Router comparison](docs/img/results-routing.svg)

Run with `make bench EXP=5`. Every arm replays the identical recorded trace from
`bench/traces/`, so the only variable is the router.

> [!NOTE]
> **The eviction policy ranking is deliberately not claimed.** Comparing LRU against
> cost-aware needs sustained memory pressure, and on an 8 GB card that means running
> at the edge of the device — where swap operations contend with inference and a wake
> measured at 0.59 s idle took **81 s** under load. Throughput collapses, an arm
> records ~40 requests, and most of the latency measured is contention rather than
> policy. Backing the load off removes the pressure the experiment is about.
>
> What *is* claimed: the tiering mechanism above, measured on hardware, and the five
> policy implementations, covered by unit tests that pin down the case each exists
> for. Which policy wins under load needs a GPU with headroom. See
> [`docs/PROGRESS.md`](docs/PROGRESS.md) for the full reasoning.

---

## HTTP API

### Inference — requires `Authorization: Bearer <api key>`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat, `stream: true` supported |
| `POST` | `/v1/completions` | OpenAI-compatible legacy completions |
| `GET` | `/v1/models` | Deployed models, their state and ready replica count |
| `GET` | `/health` | Liveness, no auth |
| `GET` | `/metrics` | Prometheus metrics, no auth |

### Admin — requires `Authorization: Bearer <admin token>`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/admin/models` | Deploy a model; returns a job |
| `GET` | `/admin/models` | List deployments |
| `GET` | `/admin/models/{id}` | One deployment, its replicas and live job |
| `DELETE` | `/admin/models/{id}` | Tear a deployment down through a job |
| `POST` | `/admin/models/{id}/scale` | Change replica count |
| `POST` | `/admin/models/{id}/pin` | Pin a model so it is never evicted |
| `POST` | `/admin/teams` | Create a team with its quota |
| `GET` | `/admin/teams` | List teams and quotas |
| `POST` | `/admin/teams/{id}/keys` | Mint an API key, shown once |
| `DELETE` | `/admin/keys/{id}` | Revoke a key |
| `GET` | `/admin/jobs` | Recent deployment jobs |
| `GET` | `/admin/usage?since=24h` | Token usage per team and model |

### Node agent — internal, one per GPU

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/status` | GPUs, host memory and every replica on this node |
| `POST` | `/replicas` | Start a replica, idempotent |
| `POST` | `/replicas/{id}/sleep` | Sleep to CPU RAM (`level: 1`) or disk (`level: 2`) |
| `POST` | `/replicas/{id}/wake` | Wake, including the weight reload a level-2 wake needs |
| `DELETE` | `/replicas/{id}` | Stop the process and release its GPU memory |

---

## Monitoring

The gateway and node agent both expose Prometheus metrics, and
[`dashboards/kiln.json`](dashboards/kiln.json) is a Grafana dashboard in four rows:
what clients experience, what the GPU is doing, what the scheduler decided, and what
routing achieved — ordered so a latency spike in the first row is explainable by the
next two without leaving the page.

| Metric | What it tells you |
|---|---|
| `kiln_ttft_seconds` | Time to first token, by model, team and replica |
| `kiln_requests_total` | Requests by status class |
| `kiln_tokens_total` | Prompt and completion tokens, by team |
| `kiln_queue_depth`, `kiln_queue_wait_seconds` | Saturation, and where a wake shows up to a client |
| `kiln_quota_rejections_total` | 429s, labelled by which limit fired |
| `kiln_gpu_memory_bytes`, `kiln_host_memory_bytes` | Device and host memory |
| `kiln_models_by_state` | How many models are resident, asleep or loading |
| `kiln_swap_duration_seconds` | Transition cost, labelled from-tier and to-tier |
| `kiln_evictions_total`, `kiln_wakes_total`, `kiln_cold_starts_total` | Swap activity |
| `kiln_residency_hits_total` / `_misses_total` | How often a request found its model already loaded |
| `kiln_thrash_total` | Evictions inside the minimum residency window; should be zero |

After `make up`, Grafana is on port 30300 (`admin`/`admin`) and Prometheus on 30090.

---

## Kubernetes Model

| Object | Kind | Why |
|---|---|---|
| `kiln-nodeagent` | DaemonSet | One per GPU node. **The only pod that requests `nvidia.com/gpu`.** |
| `kiln-gateway` | Deployment, 2 replicas | Stateless request path |
| `kiln-controlplane` | Deployment, 2 replicas | Two on purpose — the job queue is built for it |
| `kiln-postgres` | StatefulSet | Keys and usage must outlive a pod restart |
| `kiln-hf-cache` | PVC | Shared weight cache; what makes a disk-tier wake cheap |
| `prometheus`, `grafana` | Deployments | Monitoring |

Two details worth knowing. The node agent has **no liveness probe** and a 180-second
termination grace period: a cold start can block long enough to look dead to a prober,
and restarting mid-load orphans a vLLM process that still holds the GPU. And `/dev/shm`
is mounted as an 8 GiB memory-backed volume, because vLLM's engine core talks to its
workers over shared memory and Docker's default 64 MB produces a bus error during
startup that reads as an unrelated CUDA failure.

---

## Tests

```bash
make test      # unit tests: no GPU, no database, no network
make test-all  # adds the PostgreSQL integration suites
```

The default suite is offline and GPU-free, because a suite that needs a GPU is a
suite that stops being run. The scheduler and the routers are written as pure
functions — state in, decisions out, no I/O and no clock — so the interesting logic
is covered without hardware.

The PostgreSQL suites skip unless `KILN_TEST_DATABASE_URL` is set, and each migrates
into its own schema so `go test ./...` can run packages in parallel without them
truncating each other's tables.

---

## Troubleshooting

### `cudaGetDeviceCount() Error 500: named symbol not found` inside a pod

GPU passthrough is not reaching the container. Under WSL2 there is no `/dev/nvidia*`
at all — the GPU is paravirtualised through `/dev/dxg` and the libraries in
`/usr/lib/wsl/lib`, which minikube's node image does not propagate into pods.
`nvidia-smi` keeps working inside the pod, which makes this look like a CUDA version
mismatch rather than a missing passthrough. Run the node agent directly on the host
instead; the Kubernetes GPU path works on a normal Linux node.

### `No available memory for the cache blocks`

`gpu_memory_utilization` is too low for the model. It caps weights **and** the KV
cache arena together, so a 1.5B model in bf16 needs roughly 0.55 of an 8 GB card, not
0.4. Note it is also a ceiling on *device-wide* usage, not this process's own
allowance — so the second replica on a GPU needs a higher value than the first. Kiln
translates this for you: the value you give is the share this replica should get, and
the agent adds whatever is already resident.

### `OSError: [Errno 98] Address already in use` when a model restarts

The previous process has not released its port yet. Kiln takes a fresh port when
respawning a replica, so if you see this, something outside Kiln is holding the port.

### A model stays in `STARTING` forever

Check the per-replica vLLM log, whose path is in the error message and in
`docs/results/logs/vllm/` during benchmarks. A gated Hugging Face repo is the common
cause — `meta-llama/*` returns 403 without a token. Set `HF_TOKEN` or use an openly
licensed model.

### GPU memory is full but nothing is deployed

An interrupted run left orphaned vLLM processes. They are spawned into their own
process group so the agent can signal the whole tree, but if the agent is killed
abruptly the children survive and keep their allocation:

```bash
bash scripts/gpu_reclaim.sh
```

### Requests return 503 with `model_unavailable`

The gateway logs the reason. Common ones: the model is not deployed, every candidate
for eviction is inside its minimum residency window (wait and retry — this clears on
its own), or the model is larger than the whole VRAM budget and never will fit.

### Shell scripts fail with odd syntax errors on Windows

Line endings. The repo sets `* text eol=lf` in `.gitattributes`; if a script was
checked out with CRLF, bash reads a trailing `\r` on every line and reports things
like `set: pipefail: invalid option name`.

---

## Configuration

### Gateway

| Flag / env | Default | Purpose |
|---|---|---|
| `-addr` / `KILN_GATEWAY_ADDR` | `:8080` | Listen address |
| `-agents` / `KILN_AGENTS` | `http://127.0.0.1:8081` | Node agent URLs, comma-separated |
| `-database-url` / `KILN_DATABASE_URL` | — | Postgres DSN; empty disables auth, quotas and tiering |
| `-router` / `KILN_ROUTER` | `round-robin` | `round-robin`, `least-loaded`, `prefix-aware`, `prefix-aware+load` |
| `-policy` / `KILN_POLICY` | — | Eviction policy; empty disables tiering |
| `-gpu-bytes` | 8 GiB | Device VRAM the scheduler budgets against |
| `-gpu-reserve-bytes` | 256 MiB | Margin for allocator fragmentation |
| `-min-residency` | `15s` | How long a model must serve before it may be evicted |
| `-drain-deadline` | `10s` | How long to wait for in-flight requests before sleeping a victim |
| `-max-queue` | 256 | Per-model queue depth before shedding load |
| `-queue-timeout` | `180s` | How long a request waits for its model to become servable |

### Control plane

| Flag / env | Default | Purpose |
|---|---|---|
| `-addr` / `KILN_CP_ADDR` | `:8082` | Listen address |
| `-database-url` / `KILN_DATABASE_URL` | — | Postgres DSN, required |
| `-admin-token` / `KILN_ADMIN_TOKEN` | — | Shared admin token; empty disables the admin API |
| `-workers` | 2 | Deployment job workers in this process |
| `-measure-wakes` | `true` | Measure sleep/wake costs at deploy time |

### Node agent

| Flag / env | Default | Purpose |
|---|---|---|
| `-addr` / `KILN_AGENT_ADDR` | `:8081` | Listen address |
| `-python` / `KILN_PYTHON` | `python` | Interpreter that has vLLM installed |
| `-advertise-host` / `KILN_ADVERTISE_HOST` | `127.0.0.1` | Host others use to reach replicas; the pod IP under Kubernetes |
| `-hf-home` / `HF_HOME` | — | Shared weight cache |
| `-start-timeout` | `15m` | How long a cold start may take |

---

## Repository Layout

```
cmd/
  gateway/           OpenAI-compatible API, auth, quotas, routing, residency
  controlplane/      Admin API and deployment job workers
  nodeagent/         Owns one GPU, supervises vLLM processes
internal/
  api/               HTTP handlers, SSE proxy, admin API, middleware
  auth/              Hashed API keys, token-bucket quotas          (pure)
  routing/           Four routers, cumulative prefix hashing       (pure)
  scheduler/         State machine, VRAM accounting, five policies (pure)
  swap/              The tiering executor: drain, evict, wake
  jobs/              Deployment workers, SKIP LOCKED + leases
  nodeagent/         vLLM process supervision, sleep/wake, GPU probe
  store/             PostgreSQL via pgx, embedded migrations
  metrics/           Prometheus instrumentation
bench/               Trace generator, Locust harness, experiment runner
deploy/
  k8s/               Manifests for the whole stack plus monitoring
  docker/            Dockerfiles
dashboards/          Grafana dashboard JSON
docs/
  DESIGN.md          Why it is built this way, and where the tradeoffs are
  PROGRESS.md        What was measured, and the bugs found measuring it
  results/           Raw benchmark output and conditions
scripts/             Setup, end-to-end tests, figure generation
```

---

## Notes And Limits

- **Single GPU is the tested configuration.** Placement across several nodes sorts
  agents by free VRAM, which is correct but unambitious.
- **Host RAM is the real ceiling on the CPU-RAM tier.** VRAM decides how many models
  can be *active*; host RAM decides how many can be kept *warm*, and exceeding it
  gets the node OOM-killed rather than merely slowed.
- **The Kubernetes GPU path is untested on WSL2** for the reason in Troubleshooting.
  The manifests are correct and every component comes up; GPU workloads need a normal
  Linux node.
- **The eviction policy comparison is not claimed**, for the reason in Results.
- **The admin API uses a single shared token.** It is meant to be reachable only from
  inside the cluster.

Further reading: [`docs/DESIGN.md`](docs/DESIGN.md) for the design argument and where
the OS memory-hierarchy analogy breaks down, and [`docs/PROGRESS.md`](docs/PROGRESS.md)
for what was measured and the bugs that measuring it exposed.
