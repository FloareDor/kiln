# Kiln — milestone log

Measured numbers and what they mean. `context/progress.md` is the short map of where
the project stands; this file is the long-form record, one section per milestone.

---

## M0 — sanity check: does vLLM sleep mode actually free GPU memory?

**Status: done. Gate PASSED, with one significant caveat found.**

Everything in Kiln's headline feature rests on sleep mode genuinely returning VRAM to
the driver and genuinely restoring a working model. Several vLLM releases have shipped
a sleep mode that reported success while leaving the arena mapped, so this was run
before any scheduler code was written.

### Setup

| | |
|---|---|
| vLLM | `0.32.0` (pinned) |
| torch | `2.13.0+cu130` |
| GPU | RTX 3070 Laptop, 8192 MiB, driver 610.47, sm_86 |
| Host | WSL2 Ubuntu, 16 vCPU, 31 GB RAM |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, `--gpu-memory-utilization 0.35`, `--max-model-len 2048`, `--enforce-eager` |
| Script | `scripts/verify_sleep_mode.py` → `docs/results/m0_sleep_mode.json` |

Graph capture is disabled (`--enforce-eager`) so the cold-start baseline measures
weight loading rather than CUDA-graph capture, which would otherwise dominate and
flatter sleep mode.

### Results

| metric | value |
|---|---|
| resident VRAM when ACTIVE | **2926 MiB** |
| freed by sleep L1 (weights → pinned host RAM) | **2695 MiB (92.1%)** |
| freed by sleep L2 (weights discarded) | **2695 MiB (92.1%)** |
| cold start, cold page cache | 38.90 s |
| cold start, warm page cache | **34.34 s** ← the honest baseline |
| sleep L1 / wake L1 | 1.47 s / **0.88 s** |
| sleep L2 / wake L2 (with weight reload) | 0.05 s / **1.87 s** |
| wake L1 speedup vs cold start | **39.0×** |
| wake L2 speedup vs cold start | **18.4×** |
| output identical to pre-sleep reference | L1 ✅ L2 ✅ |

Sleep mode works and it is worth building on. Paging a model out and back costs under
a second at L1 and under two at L2, against 34 seconds to reload it from scratch.
Roughly 231 MiB stays resident while asleep — the CUDA context and vLLM's own
bookkeeping — which is the fixed overhead per sleeping model that the scheduler's
budget has to account for.

### Finding 1: level-2 wake is silently broken without a weight reload

`POST /wake_up` after a level-2 sleep returns `200`, restores the memory, and then
**generates garbage** (`'!!!!!!!!!!!!'` against a reference of `' Paris. It was
founded in 789 AD by'`). It also returns in 0.45 s, which is far too fast to have
re-read a gigabyte of weights.

The cause is visible in `vllm/v1/worker/gpu_worker.py`: level-2 sleep discards the
weights, and `wake_up()` only replays `_sleep_saved_buffers` and any parameters
matched by `sleep_preserve_parameter_names`. The weights themselves are never
reloaded. The correct protocol is:

```
POST /wake_up
POST /collective_rpc  {"method": "reload_weights"}
```

With the reload, output matches the reference exactly and the wake costs 1.87 s.

This is the kind of bug that would have quietly destroyed the whole project: the
benchmark would have shown L2 wakes at 0.45 s — four times *faster* than L1 — and the
results table would have recommended the broken tier. The verification script now
compares generated text against a pre-sleep reference on every wake, so a
fast-but-wrong wake fails the gate rather than winning the benchmark.

### Finding 2: NVML per-process accounting is unavailable under WSL2

`nvmlDeviceGetComputeRunningProcesses` returns an empty list under WSL2's NVML shim,
so the first version of this gate compared 0 MiB freed against 0 MiB resident,
concluded `0 >= 0.5 × 0`, and printed **PASS** having measured nothing at all.

The script now probes for that at startup, falls back to device-wide memory minus a
measured idle baseline (1318 MiB here: the Windows compositor), states which method it
used in its output, and refuses to grade a run whose resident footprint came out as
zero. The node agent inherits the same fallback.

### Decisions this settles

- **Pin vLLM `0.32.0`.** Sleep mode verified working on it, at this exact flag set.
- **Both tiers are worth having.** L1 is ~2× faster to wake than L2 but costs pinned
  host RAM; L2 is free of host RAM and still 18× better than a cold start. That is
  exactly the trade a tiering policy should get to make, which justifies
  `ChooseTier`.
- **Count ~231 MiB of fixed overhead per sleeping model** in the VRAM budget.
- **Cold-start baselines use a warm page cache.** The first-ever load pays a disk read
  no subsequent cold start pays; using it would overstate sleep mode's advantage by
  about 13%.
- **`--no-enable-log-requests`**, not `--disable-log-requests`, which 0.32.0 removed.
- Sleep accepts `mode=wait`, which blocks for in-flight requests. The node agent uses
  it as a second line of defence behind the scheduler's own draining.

---

## M1 - gateway + node agent, one model, streaming

**Status: done. All acceptance checks pass on the local RTX 3070.**

Done when the stock OpenAI Python SDK streams from the platform with no Kiln-specific
client code. `scripts/e2e_m1.py` brings up a node agent and a gateway, deploys a model
through the agent's API, then drives everything through `openai.OpenAI(base_url=...)`.

### Results

| check | result |
|---|---|
| cold start to ACTIVE (Qwen2.5-0.5B, eager) | **42.9 s** |
| `GET /v1/models` lists the model and its tier state | PASS |
| non-streaming chat completion returns text + usage | **1.16 s**, 34 tokens |
| streaming: TTFT vs total | **0.071 s vs 1.107 s**, 37 chunks |
| sleep L1 then wake through the agent | 1.27 s / **0.59 s** |
| greedy output identical after the wake | PASS |

The TTFT number is the one that matters structurally. At 0.071 s against a 1.107 s
total, the gateway is demonstrably forwarding SSE frames as they arrive rather than
buffering the response. If it buffered, TTFT would equal total and **every latency
number this project goes on to report would be an artefact of the gateway rather than
a property of the model**. The test asserts `ttft < 0.8 x total`, so a future change
that reintroduces buffering fails rather than silently inflating the results.

### Finding: replica ids go into URL paths, so they must be URL-safe

The first run failed at the sleep step with a 405. The replica id was `qwen-0.5b#0`,
and `#` begins a URL fragment, so `POST /replicas/qwen-0.5b#0/sleep` was delivered to
the server as `POST /replicas/qwen-0.5b` - a path with no handler. The fix is both
halves: the naming convention now uses `--` as the separator, and `agentclient`
`url.PathEscape`s every id it puts in a path, because an id is caller-supplied data
and an unescaped `/` would retarget the request entirely.

### Design decisions

- **The gateway polls node agents rather than being pushed to.** A serving platform
  whose data path depends on its control path has outages its users cannot explain.
  Polling also makes recovery after a partition automatic. One unreachable agent drops
  only its own replicas from the fleet view and leaves every other agent serving.
- **Only `ACTIVE` is routable.** `STARTING`, `WAKING`, `DRAINING`, `SLEEPING_*` and
  `FAILED` all mean "no traffic", for different reasons the router need not
  distinguish.
- **`stream_options.include_usage` is injected into every streaming request.** vLLM
  only emits a usage block when asked. Without it the gateway cannot count tokens for
  streaming responses - which is most real traffic - and per-team quotas would
  silently stop working on exactly the requests that matter.
- **The node agent kills process groups, not processes.** vLLM runs its engine core in
  a child; killing only the parent leaves that child holding the GPU, and the next
  model fails to place against memory nothing appears to own.
- **A replica is ACTIVE only once it has generated a token.** vLLM reports healthy
  before the model is usable on some paths, and the level-2 wake bug from M0 produces
  a replica that is healthy and wrong.
- **Ensure runs before routing.** It is the hook the swap scheduler takes over in M5;
  routing first would mean a request for a sleeping model 503s instead of waiting for
  a wake, making tiering unreachable from the request path.

---

## M2 - Postgres, auth, quotas, concurrency-safe deploy jobs

**Status: done. All 14 acceptance checks pass (`scripts/e2e_m2.py`).**

Three claims, each checked end to end against a real Postgres and the local GPU.

### 1. Deploying a model is one API call

`POST /admin/models {"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct", ...}` returns `202` with
a job id. A worker claims the job, downloads the weights, starts vLLM, measures the
footprint and both wake costs, and marks the model `READY`.

| | |
|---|---|
| time to READY | **46 s** |
| measured VRAM footprint | 2867 MiB |
| measured cold start | 41.4 s |
| measured wake L1 / L2 | **587 ms / 710 ms** |

Those measurements are the point of doing it this way. The cost-aware eviction policy
in M5 scores victims by `rate x wake-cost / GB`, and if the wake costs were guessed
rather than measured, the policy would be confidently wrong on this exact GPU.

### 2. Two teams with different quotas are throttled correctly

| team | limit | observed status codes | `Retry-After` |
|---|---|---|---|
| `tight` | 3 rpm | `200 200 200 429 429 429` | 19 |
| `generous` | 600 rpm | `200 200 200 200 200 200` | - |

No key and an invalid key both return `401`. Usage is attributed per team and per
model (`generous`: 6 requests / 180 prompt / 24 completion tokens; `tight`: 3 / 90 /
12), including for streaming requests.

### 3. A job survives a worker crash

The control plane is `SIGKILL`ed mid-deploy - no failure reported, no completion, no
lease renewal, nothing. A replacement control plane starts, finds the lapsed lease,
and finishes the job: **DONE after 2 attempts**.

This is the case a row lock alone cannot handle. `FOR UPDATE SKIP LOCKED` is what lets
several control planes drain one queue without serialising, but the lock dies with its
transaction while the work takes minutes. The claim is therefore also a lease
(`locked_by` + `lease_until`): a crashed worker's lease simply expires, while a merely
slow worker renews it. `attempts` increments at claim time rather than at failure time,
because a worker killed mid-job never reports a failure - counting only reported
failures would let a job that reliably crashes its worker retry forever.

### Three bugs the tests caught

**`RecordMeasurements` would have failed for every real model.** pgx cannot infer a
parameter's type when it appears only inside a `CASE` expression, so it defaulted to
`int4` - and every footprint worth storing exceeds 2 GiB. The fix is explicit
`::bigint` casts. This failed on the first real 3 GB write, not in any unit test with
small numbers, which is exactly why the store suite runs against a real Postgres
rather than a fake.

**The "idempotent" deploy retry silently rewrote the deployment.** The handler upserted
the model row first and deduplicated the *job* afterwards, so a bare retry carrying a
shorter body reset `max_model_len` to the handler default and `enforce_eager` to
`false`. The already-running deploy then attempted `torch.compile`, shelled out to
`nvcc`, and died with `PermissionError: [Errno 13]`. Idempotency has to cover the model
row, not just the job: the in-flight check now happens before any write.

**Nil slices serialised as `null`.** A Go `var out []T` that matches no rows marshals to
`null`, not `[]`, forcing every client to treat "no rows" as a different type from
"some rows".

### Design decisions

- **Quota enforcement is in-memory; durable counters are in Postgres.** A token-bucket
  check that needed a database round trip would add milliseconds to every request and
  would fail open or closed during a database blip. The limiter is seeded from the
  database at startup, so a restart does not hand every tenant a fresh daily budget.
- **RPM uses a token bucket, not a fixed window.** A fixed window lets a team spend a
  full minute's budget at the end of one window and again at the start of the next -
  twice the nominal rate in an instant, exactly when a GPU can least absorb it.
- **Tokens are counted after the fact.** A team at 99% of its budget can start a request
  that pushes it over. Pre-reserving a worst-case token count would reject requests
  that would have fit, which is the worse error.
- **API keys are SHA-256, not bcrypt.** The usual argument for a slow hash is a
  low-entropy human password; these are 192 bits of CSPRNG output, so there is no
  dictionary to attack and a slow hash buys nothing but latency on every request. A
  stored plaintext prefix turns verification into one indexed lookup plus one
  constant-time compare.
- **Usage writes are buffered and batched.** A synchronous insert per request would put
  the database on the gateway's latency path. A crash loses at most one flush interval
  of rows; the in-memory limiter is updated immediately, so quota enforcement never
  lags even when the write does.
- **Deletes go through a job.** An HTTP handler cannot stop vLLM processes on remote
  nodes within one request, and removing the row without stopping them would orphan
  processes holding GPU memory that nothing accounts for.

### Environment finding: no CUDA toolkit under WSL

`torch.compile` (inductor) shells out to `nvcc`, which is not installed in WSL, so all
local work runs with `--enforce-eager`. M0 had already chosen eager deliberately, to
keep cold-start-versus-wake an honest comparison rather than a CUDA-graph comparison.
Every benchmark arm runs eager, so the comparisons remain valid; the Modal runs in M7
can compile.

---

## M3 - multiple replicas and Kubernetes

**Status: stack deployed and healthy on a GPU-enabled cluster. Acceptance run in
`scripts/e2e_m3.py`.**

`make cluster` brings up minikube with `--gpus all` plus the NVIDIA device plugin, and
`make up` builds the three images, streams them into the node and applies the
manifests. Eight pods: two gateways, two control planes, one node-agent DaemonSet,
Postgres, Prometheus and Grafana.

### The architectural claim, checked rather than asserted

The acceptance test enumerates every container's resource limits and asserts that
**exactly one pod requests `nvidia.com/gpu`, and it is the node agent**. That is the
whole design in one assertion: Kubernetes hands a pod a GPU for the pod's entire
lifetime, so if anything else held one, that model would own its device forever and
tiering would be unexpressible.

### Two bugs found by deploying for real

**The node agent crash-looped with `exec: "-addr": executable file not found`.** The
Dockerfile cleared `ENTRYPOINT` so the agent could be PID 1 instead of the vLLM base
image's own entrypoint. But a Kubernetes `args:` list *replaces* `CMD` and *appends to*
`ENTRYPOINT`, so with an empty entrypoint the kubelet tried to execute `-addr` as a
program. Fixed on both sides: `ENTRYPOINT ["/usr/local/bin/nodeagent"]` in the image,
and an explicit `command:` in the manifest so the pod is immune to a surprise
entrypoint in whatever base image it is rebuilt on.

**`minikube image load` is unusable on Windows 11.** It shells out to `wmic` to
translate a drive letter into a volume name, and `wmic` was removed from the OS.
Images now stream straight into the node's daemon over SSH
(`docker save | minikube ssh docker load`), which also avoids writing a 23 GB tarball
to disk on the way.

### Finding: GPU workloads do not run under minikube on WSL2

The stack deploys and every component is healthy, but a vLLM process started by the
in-cluster node agent fails with:

```
RuntimeError: Unexpected error from cudaGetDeviceCount().
Error 500: named symbol not found
```

This is an environment limitation, not a defect in Kiln, and it is worth writing down
precisely because the symptom points in the wrong direction.

What the pod actually sees:

| | |
|---|---|
| `nvidia-smi` | **works** — reports the RTX 3070, driver 610.47, CUDA UMD 13.3 |
| `nvidia.com/gpu` capacity | **1** — the device plugin advertises it, the pod is scheduled |
| `/dev/dxg` | present |
| `/dev/nvidia*` | **absent** |
| `libcuda.so.1` | the 188 KB WSL forwarding stub |
| `/usr/local/cuda/compat/libcuda.so.580.95.05` | a full driver library, shipped in the vLLM image |

The cause is the third and fourth rows. On a normal Linux host the NVIDIA container
toolkit injects `/dev/nvidia*` and the real driver library. Under WSL2 there is no
`/dev/nvidia*` at all: GPU access is paravirtualised through `/dev/dxg` and the
forwarding libraries in `/usr/lib/wsl/lib`, which minikube's node image does not
propagate into pods. `nvidia-smi` keeps working because NVML's stub path is satisfied,
which is exactly why the failure looks like a CUDA version mismatch rather than a
missing passthrough. Disabling the image's CUDA compat directory does not help, so the
problem is the passthrough itself and not library precedence.

**Scope of the limitation.** It applies to this development machine's
Windows → WSL2 → Docker Desktop → minikube → pod stack, which is four layers of
virtualisation. It does not apply to a normal Linux GPU node, where `/dev/nvidia*` and
the real `libcuda` are what the device plugin injects, and it does not apply to Modal,
where the node agent runs as a plain container on a real Linux host. The node agent is
built to run standalone without Kubernetes for exactly this reason.

**What this does and does not establish.** Verified here: the manifests are correct,
all eight pods come up, Postgres and the two control planes and two gateways and
Prometheus and Grafana all work, the DaemonSet schedules onto the GPU node, and the
architectural assertion holds — exactly one pod requests `nvidia.com/gpu` and it is
the node agent. Not verified here: a model actually serving from inside the cluster.
That path is covered by M1 and M2 running the same binaries natively against the same
GPU, and will be covered again on Modal in M7.

### Notes

- The gateway and control plane restarted four or five times while Postgres was
  initialising. That is the intended behaviour rather than a defect -- both exit when
  the database is unreachable at startup, and the kubelet's restart *is* the retry
  loop -- but it is worth knowing that a cold `make up` shows a handful of restarts
  before settling.
- The node agent gets `terminationGracePeriodSeconds: 180` and no liveness probe. A
  cold start can block long enough to look dead to a prober, and restarting mid-load
  orphans a vLLM process still holding the GPU, which the next pod then cannot place
  against.
- `/dev/shm` is mounted as an 8 GiB `emptyDir` with `medium: Memory`. vLLM's engine
  core talks to its workers over shared memory, and Docker's default 64 MB produces a
  bus error during startup that reads as an unrelated CUDA failure.

---

## M4 - observability and load testing

**Status: metrics and dashboard done; experiment runs in progress.**

Prometheus instrumentation covers the client view (TTFT, throughput, status classes,
queue depth and wait, quota rejections), the device view (GPU and host memory, models
by state), the tiering view (swap duration by transition, evictions, wakes, cold
starts, residency hits and misses, thrash, drain timeouts) and the routing view
(prefix hits and misses, blocks matched, in-flight per replica).

Two instrumentation decisions worth recording:

- **Latency buckets are hand-chosen, not `prometheus.DefBuckets`.** The default
  buckets top out at 10 s. LLM serving spans four orders of magnitude -- a prefix-cache
  hit answers in 50 ms, a cold start takes 40 s -- so every cold start would land in
  `+Inf` and the p99, which is the number the whole swapping argument turns on, would
  be unmeasurable.
- **Status codes are bucketed into classes.** Raw codes would be fine today but are a
  cardinality trap the moment an upstream starts returning a long tail of them.

`bench/tracegen.py` writes six fixed traces (steady, Zipf at s = 0.6/1.0/1.4, bursty,
agent) with fixed seeds. Every policy and router arm replays the identical sequence, so
a difference in p95 is attributable to the arm rather than to the random draw.

`bench/locustfile.py` replays at recorded arrival times rather than as a closed loop,
because a closed loop's request rate is set by the model's own latency -- which is the
variable under test. It measures TTFT on the first chunk carrying content, not the
first SSE frame: vLLM emits a role-only delta first, and counting it would under-report
TTFT by a whole decode step.

---

## M5 - model swapping

**Status: executor and policies done and tested; experiments pending.**

`internal/swap` turns the scheduler's pure decisions into sleep, wake and start calls
against node agents. 21 tests, no GPU required. The properties they pin down are listed
in the commit for `M5 core`, and the reasoning behind each is in `docs/DESIGN.md`.

The controller lives in the gateway rather than the control plane. A model may only be
slept once its in-flight requests reach zero, and the gateway is the only component
that knows that count exactly and instantly; polling for it from the control plane
would put a network round trip on the eviction path and return a number that is already
stale.

---

## M5 - model swapping: what the benchmark found

**Status: the mechanism works and is measured. Policy comparison running.**

Experiment 2's numbers are above. This section is about what running the policy
comparison exposed, because the benchmark turned out to be a better test of the
swapping code than the unit tests were.

The pattern repeated four times: a run finished, produced plausible-looking numbers,
and was wrong. Each time the giveaway was a counter that should not have been zero.

### 1. Swaps were cancelled by whoever triggered them

The first full run reported **zero evictions on a fleet it had just confirmed was
oversubscribed**. A swap decision log, added to find out why, showed:

```
swap step failed  step=wake q06  err=... context canceled
```

The swap ran on the triggering HTTP request's context. But a swap is *shared* work
with real GPU side effects - fifty requests coalesce onto one wake - so binding it to
one client's lifecycle meant that client timing out aborted the wake part-way, failed
everyone waiting on it, and left the model neither asleep nor resident. A benchmark
run merely *ending* cancelled every swap in flight.

Swaps now run on a context detached from the caller's. The caller only waits; giving
up stops the wait, not the swap, and the model still becomes resident for whoever asks
next. Same workload, LRU on `zipf_s10`:

| | before | after |
|---|---|---|
| p95 TTFT | 9.66 s | **3.62 s** |
| p99 TTFT | 9.66 s | **4.13 s** |
| residency hit ratio | 0.42 | **0.78** |
| thrash events | 10 | **0** |

### 2. ErrWouldThrash reached the client as a 503

`ErrWouldThrash` means enough VRAM exists but every candidate is inside its
minimum-residency window - a condition that clears on its own. `DESIGN.md` said the
gateway queues such a request. The controller returned the error straight through.
That turns a momentarily contended GPU into an outage and makes the whole
`ErrWouldThrash` / `ErrCannotFit` distinction pointless. `Ensure` now retries with
backoff, bounded by the caller's deadline, and reports *what is blocking it* rather
than a bare timeout.

### 3. The cold baseline could stop a model but never restart it

The ten-arm run logged **7004 failed swap steps out of 7009**:

```
start q05a  err=... 500: nodeagent: unknown replica
```

The `cold` policy evicts by stopping a victim's process outright - that is what makes
it the no-tiering baseline - and the agent forgets a stopped replica entirely. The
controller's cold-start path then asked the agent to *wake* a replica that no longer
existed. The model became permanently unreachable, and because every arm shares one
agent, the first cold-policy eviction poisoned all nine arms after it.

A cold start now rebuilds the vLLM spec from the registry and starts a fresh process,
using the deployment's own spec so a restarted replica is configured identically to
the one that was stopped.

**The fake agent in the unit tests was hiding this.** It accepted a wake for any
replica, including ones it had been told to stop, so the suite stayed green while the
real agent returned 500. The fake now forgets stopped replicas and rejects unknown
ones - which is the general lesson: a fake that is more permissive than the real thing
does not test the interesting cases, it conceals them.

### 4. Minimum residency was mis-calibrated, and that is a real constraint

At 15 s, with a trace at 2 req/s across three models, every model was almost always
"freshly resident". Nothing was ever evictable, and the scheduler stopped making
progress: 52 thrash events and zero evictions in two minutes.

The window is bounded **below by the wake cost** (it exists so an eviction's cost is
not immediately wasted) and **above by the inter-request interval for one model** (set
longer, and nothing is ever evictable). A fleet where those two bounds cross has no
workable setting. At 5 s the same workload performed 3 evictions and 6 wakes with no
failed steps.

### 5. The traces did not match the fleet

Generated for five models, replayed against three. Nearly half of every trace addressed
models that did not exist: those requests failed instantly, never triggered a swap, and
dragged measured latency *down* while the eviction counters stayed at zero. A run that
looks finished, with plausible numbers, measuring nothing. `tracegen` now takes the
fleet and the runner refuses to start when a trace addresses models that are not
deployed.

### Fleet calibration for an 8 GB card

Three models totalling 9.6 GB against a 7.2 GB usable budget saturates the device -
p95 reached 80 s, which measures the hardware rather than the policy. The benchmark
fleet is now four models of **mixed** size (1.8 GB, 1.9 GB, 1.8 GB, 4.1 GB = 9.7 GB)
at a lower request rate, so two or three are resident and eviction is frequent without
being pathological.

Mixed sizes are deliberate: a uniform fleet would make `cost-aware`'s per-GB
normalisation indistinguishable from `lfu-decay`, so the experiment could not show the
thing it exists to show.

Healthy run, for reference: p50 78 ms, p95 3.7 s, 9 evictions, **0 thrash events, 0
failed swap steps**.

---

## Experiment 3 (eviction policy comparison): not run. Why.

**The swapping mechanism is measured and works. The policy ranking is not claimed.**
That distinction is deliberate, and this section exists so nobody reads a missing
table as an oversight.

### What the comparison needs, and why this GPU cannot provide it

Comparing LRU against cost-aware against predictive-prewarm requires sustained memory
pressure: more models wanted than fit, with evictions happening continuously. On an
8 GB card with ~1.1 GB already held by the desktop, that means a fleet right at the
edge of the device.

At that edge, two things collide. Swap operations and inference contend for the same
GPU, so with the card busy a wake that takes **0.59 s idle was measured at 81 s**.
Throughput then collapses to about **0.14 req/s**, and a five-minute arm records
roughly 40 requests -- too few for a trustworthy p95, and most of the latency being
measured is contention rather than policy.

Pushing load down to avoid saturation removes the memory pressure that the experiment
is about. There is no setting of the dial that gives both on this hardware.

### What is claimed instead

- **The tiering mechanism, measured on hardware** (experiment 2 above): sleep frees
  92% of a model's resident VRAM, and waking costs 0.59-1.54 s against a 35-37 s cold
  start, 24x to 60x depending on model size.
- **The policy implementations, verified by unit tests**: all five share one fitting
  loop and differ only in a scoring function, and 29 scheduler tests pin down the
  cases each is meant to handle -- LRU falling for a scan that lfu-decay resists,
  cost-aware evicting one large cold model where LRU needs three small warm ones,
  prewarming refusing to evict to make room for a guess.
- **No claim about which policy wins under load.** Measuring that honestly needs a
  GPU with headroom, which is what the Modal run in M7 is for.

### The cost of finding this out

Roughly two hours of GPU time across a dozen runs, and the admission is worth
recording: the infeasibility was derivable up front. A 1.5B model needs 4.1 GB of a
6.3 GB schedulable budget; one model taking two thirds of the device cannot coexist
with anything, so eviction can never succeed. Arithmetic available before the first
run would have said so. Instead it was discovered empirically, one run at a time.

The runs were not wasted -- they found eleven real bugs, listed above, several of
which only appear under concurrent load -- but the experiment itself was never going
to work here.

---

## M6 - prefix-aware routing: measured

**Status: done. All four routers, 81 requests each, same recorded trace.**

Two replicas of one 0.5B model on the RTX 3070, replaying agent traffic -- multi-turn
conversations that resend the same system prompt and tool definitions every turn, so
most of each prompt is a shared prefix.

| Router | p50 TTFT | p95 TTFT | p99 TTFT | Requests per replica |
|---|---|---|---|---|
| `round-robin` | 1029 ms | 2676 ms | 2826 ms | 41 / 40 |
| `least-loaded` | 273 ms | 2827 ms | 2865 ms | 43 / 37 |
| `prefix-aware` | **85 ms** | **160 ms** | 211 ms | 81 / 0 |
| `prefix-aware+load` | 106 ms | 2525 ms | 2825 ms | 52 / 29 |

All arms served 81 requests with zero errors at 0.35 req/s offered load, so the
comparison is like for like.

### What the numbers say

**Prefix affinity is worth 12x on p50 and 17x on p95 against round-robin.** That is
the mechanism working exactly as intended: consecutive turns of one conversation land
on the replica that already holds the system prompt and tool definitions in its KV
cache, so the shared prefill is a cache hit rather than recomputed work. Round-robin
scatters those turns and pays full prefill nearly every time.

**Least-loaded is a much better baseline than round-robin**, and worth having as a
separate arm -- 273 ms against 1029 ms at p50 purely from not sending work to a busy
replica. Beating round-robin is easy; beating least-loaded is the claim that matters,
and prefix-aware does it by a further 3x.

**The hot spot is real, and the numbers show it rather than hide it.** Pure affinity
routed all 81 requests to one replica and left the other completely idle. Its
excellent p95 is partly a consequence of that concentration: every request hit a warm
cache on one machine. But the deployment has no headroom in that state, and at higher
load that replica saturates while its peer does nothing. The unit test
`TestPrefixAwareWithoutLoadPenaltyCreatesAHotSpot` pins this behaviour deliberately.

**`prefix-aware+load` is the configuration to actually run.** It keeps most of the p50
win (106 ms vs 85 ms), spreads work 52/29 instead of 81/0, and pays for it at p95
(2525 ms vs 160 ms) because breaking affinity to balance load costs cache misses. That
is precisely the trade the load penalty exists to make, and seeing both arms side by
side is the reason both were kept.

### Why this experiment worked where the policy comparison did not

Two replicas of a 0.5B model is 3.6 GB on an 8 GB card. No memory pressure, so no
swapping, so swap operations never contend with inference for the device. Throughput
stays normal, every arm records a full sample, and p50 and p95 are both trustworthy.

The eviction policy comparison needed the opposite condition -- sustained memory
pressure -- which on this hardware means saturation, and saturation is what destroyed
that measurement. Same platform, same harness; one experiment fits the hardware and
the other does not.

---

## Appendix - scheduler core (built early)

Built early because it is pure logic that needs no GPU, so it could proceed while
model downloads and installs were running.

`internal/scheduler` holds the state machine, VRAM/host-RAM accounting, the demand
tracker and all five eviction policies. Nothing in the package performs I/O or reads
the clock; a policy takes a `ClusterState` and returns decisions. **29 unit tests
pass** with no GPU attached.

The five policies share one fitting loop and differ only in a scoring function, so a
benchmark gap between two of them is attributable to the score rather than to
incidental implementation differences. Notable cases the tests pin down:

- a `WAKING` model's VRAM counts against the budget (not counting it oversubscribes
  the GPU and OOMs mid-wake);
- `ErrWouldThrash` (wait and retry) is distinguished from `ErrCannotFit` (fail now);
- 50 concurrent requests for one sleeping model produce exactly one wake;
- a drain that is abandoned because the victim became hot again goes straight back to
  `ACTIVE` rather than completing a round trip;
- victim choice is deterministic under Go's randomised map iteration, without which
  the policy comparison in experiment 3 would be noise.
