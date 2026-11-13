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

## M5 (partial) - scheduler core

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
