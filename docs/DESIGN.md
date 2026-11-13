# Kiln — design notes

What was hard, what was decided, and why. The milestone-by-milestone record with
measured numbers is in [PROGRESS.md](PROGRESS.md); this file is the argument.

---

## 1. The central problem

GPUs are expensive and VRAM is the binding constraint. An 8 GB card fits two small
models; an A100 fits one 70B model. A team that wants to serve six models either buys
six GPUs' worth of capacity or loads models on demand and pays a **cold start** — 35
to 40 seconds for a 0.5B model on the development box, minutes for anything real.

Both options are bad, and the second is worse than it looks: a 35-second cold start on
the request path is not a slow request, it is a timeout.

Kiln's answer is that this is a memory hierarchy problem, and memory hierarchies have
a well-understood solution. Treat VRAM as RAM, host memory as swap, and disk as the
page file:

| Tier | OS analogue | vLLM mechanism | Measured wake (0.5B, RTX 3070) |
|---|---|---|---|
| GPU VRAM | resident pages | — | 0 |
| CPU RAM | swap | sleep level 1 | **0.59–0.88 s** |
| Disk | page file | sleep level 2 + weight reload | **0.71–1.87 s** |
| Not loaded | not in the working set | spawn + load + profile | **34–41 s** |

Those are measured, not estimated — `scripts/verify_sleep_mode.py` and the deploy job
produce them. The gap between rows 2–3 and row 4 is the entire justification for the
project: **paging a model out and back costs 1–2% of what reloading it costs.**

### Where the analogy breaks

It is worth being precise about this, because it is the first thing an interviewer
should push on.

- **Pages are uniform; models are not.** A page is 4 KB. Models here range from 2.8 GB
  to 4.3 GB, and on Modal from 14 GB upward. Evicting "one model" frees wildly
  different amounts, which is why the cost-aware policy normalises its score *per GB*
  rather than treating every victim as one unit.
- **Page faults are uniform; wakes are not.** A page fault costs what a page fault
  costs. A wake costs 0.6 s or 1.9 s or 40 s depending on which tier the model is in
  and how big it is. A policy that ignores this will happily evict the expensive thing.
- **Arrivals are predictable in a way page faults are not.** Request traffic has
  structure — diurnal patterns, per-tenant bursts, a model that is being actively
  worked on. That is exploitable, and it is what the predictive-prewarm policy tries
  to exploit. An OS has far less signal.
- **Draining has no OS analogue.** You cannot evict a page that is being written to;
  you also cannot sleep a model that is mid-generation. The OS handles this with pins
  and locks on a microsecond timescale. A generation runs for seconds, so Kiln needs an
  explicit drain with a deadline, and has to decide what to do when the deadline
  expires.

---

## 2. Why a node agent instead of one pod per model

This is the architectural decision everything else follows from.

Kubernetes allocates a GPU to a pod **for the pod's entire lifetime**. There is no
"release the GPU but keep running". So the obvious design — one Deployment per model,
each pod requesting `nvidia.com/gpu: 1` — pins exactly one model to each GPU forever.
Serving six models needs six GPUs. Tiering is not merely hard in that design; it is
unexpressible.

Kiln instead runs **one node agent pod per GPU**, which is the only thing that requests
`nvidia.com/gpu`, and it spawns vLLM as child processes. The GPU is held by a
long-lived supervisor that can start, sleep, wake and stop models underneath it.

The costs of this are real and worth naming:

- **Kubernetes cannot see the models.** They are processes, not pods, so there is no
  `kubectl get pods` view of what is loaded, no per-model resource limits, and no
  automatic restart. Kiln has to provide that itself — which is what the node agent's
  `/status`, the replica table and the state machine are.
- **Blast radius.** A node agent crash takes every model on that GPU with it. One pod
  per model would lose one model.
- **Process supervision is now Kiln's problem**, including the detail that killing the
  parent vLLM process leaves its engine-core child holding the GPU. The agent kills
  process *groups* for exactly this reason.

The benefit is that a GPU can hold several models and page between them in under a
second, which is the entire product.

---

## 3. Why residency lives in the gateway

The control plane owns deployments: what models exist, what they cost, which node they
are placed on. The gateway owns residency: which of them are in VRAM right now.

That split is not arbitrary. **Draining decides it.** A model may only be slept once
its in-flight requests have finished, and the gateway is the only component that knows
that count exactly and instantly — it is the thing that incremented it. Putting the
swap controller in the control plane would mean polling the gateway for the count on
the eviction path: a network round trip to obtain a number that is already stale when
it arrives.

The corollary is that the gateway keeps working when the control plane is down. It
polls node agents for its fleet view rather than reading the database, so a control
plane outage stops new deployments but does not stop serving. A serving platform whose
data path depends on its control path has outages its users cannot explain.

---

## 4. The scheduler is pure

`internal/scheduler` performs no I/O, reads no clock, and opens no sockets. A policy
receives a `ClusterState` and returns decisions; `internal/swap` carries them out.

This is the single decision that most improved the project's testability. The
interesting logic — "which model should be evicted, to which tier, and is it safe
to do so yet" — is exercised by 29 unit tests that run in 10 milliseconds with no GPU
attached. A test suite that needed a GPU would be a test suite that stopped being run.

It also makes the policy comparison honest. All five policies share one fitting loop,
one tier-selection rule and one thrash guard, and differ **only in a scoring
function**. A gap between two arms in experiment 3 is therefore attributable to the
score and not to some incidental difference in how one of them happened to be
implemented.

A related detail that matters more than it looks: Go randomises map iteration order, so
a policy that iterated `state.Models` directly would return *different victims on
identical input*. The A/B comparison would then be partly measuring Go's hashmap seed.
Every policy iterates through `SortedIDs()`, and a test runs each policy 50 times on an
ambiguous fixture to prove the choice is stable.

---

## 5. The eviction policies

| policy | score (lowest evicted first) | what it is blind to |
|---|---|---|
| `cold` | last request time, then **kill the process** | everything; it is the no-tiering baseline |
| `lru` | last request time | size and wake cost |
| `lfu-decay` | exponentially decayed arrival rate | size and wake cost |
| `cost-aware` | `rate × wake_cost / GB_freed` | correlations between models |
| `predictive-prewarm` | same as cost-aware, plus a forward-looking `Tick` | — |

**Why `cold` is implemented carefully.** It is the arm that answers "is any of this
worth building?", so it has to be a fair version of the obvious alternative rather than
a strawman. It picks its victim by LRU, exactly as the `lru` arm does, and differs only
in that it stops the process instead of sleeping it. One variable.

**Why plain LFU is not an arm.** A model that was hot last Tuesday keeps its count
forever and is never evicted. Decay gives recent requests more weight, which behaves
like LFU over a sliding window without storing one — two floats per model instead of a
window, which matters when the scheduler ticks several times a second.

**Why cost-aware normalises per GB.** This is the part that actually wins. LRU will
evict a 6 GB model that takes 40 s to wake in order to admit a 1 GB one, when evicting
two small cheap models would have been strictly better. Dividing by the memory freed
means a victim that frees six times the headroom can justify six times the expected
cost. The unit tests pin the case down directly: on a mixed fleet, cost-aware evicts
one large cold model where LRU needs three small warm ones.

**Why the wake cost is read from the tier the victim would actually land in.** If host
RAM is full, the victim is going to disk, so the disk wake cost is what scoring it
against means. Using the host cost there picks the wrong victim with full confidence.

**Why prewarming is strictly opportunistic.** `predictive-prewarm` only wakes a model
into VRAM that is *already free*. A speculative wake that forces an eviction is a bet
the policy has no business making: it is trading a certain cost (evicting something
that is serving) for an uncertain benefit (a model that might be requested). The
eviction scoring is identical to cost-aware on purpose, so the experiment isolates the
value of prewarming rather than confounding two changes.

---

## 6. Anti-thrash

Two models that each receive steady traffic, on a GPU that fits one, will evict each
other on every alternating request and spend all their time swapping. This is the
failure mode that makes naive tiering worse than no tiering.

Three guards:

- **Minimum residency.** A model that has been serving for less than
  `MinResidency` is not evictable. This is the primary guard and it is what makes the
  pathological alternating case impossible rather than merely unlikely.
- **Hysteresis.** Evictions free a little more than strictly needed. Freeing exactly
  enough leaves the GPU at 100% occupancy, so the next model — even one a few MiB
  larger — triggers another full eviction cycle. Over-freeing converts a stream of
  single-victim evictions into occasional two-victim ones, which is cheaper overall.
- **Idle demotion.** A host-tier sleeper nobody has asked for in several minutes is
  demoted to disk, returning pinned host RAM to the budget.

The most useful thing the residency guard produces is not the guard itself but the
**distinction it forces**. When a swap cannot proceed, there are two different
situations:

- `ErrWouldThrash` — enough VRAM exists in principle, but every candidate is inside its
  residency window. Waiting *will* help. The gateway queues the request.
- `ErrCannotFit` — the model is larger than the whole budget, or everything else is
  pinned. Waiting will never help. The gateway fails fast.

Collapsing these into one error means either failing requests that would have
succeeded, or hanging on requests that never could.

---

## 7. Draining, and what happens at the deadline

Sleeping a model with requests in flight kills those requests. Waiting indefinitely for
them lets one slow generation block an eviction forever, which makes eviction latency
unbounded and the whole system unusable under load.

So: stop routing to the victim (flip it to `DRAINING`, which removes it from the
routable set), wait for in-flight to reach zero, and bound that wait with
`DrainDeadline`. The order matters — flipping the state *after* the wait would mean new
requests keep arriving throughout and the drain never converges.

At the deadline, Kiln sleeps anyway, with vLLM's `mode=wait` rather than `mode=abort`,
which gives the stragglers one more chance to finish inside vLLM. The event is counted
in `kiln_drain_timeouts_total`, not hidden: a rare timeout is acceptable, a sustained
rate means the deadline is too tight for the workload and should be tuned.

---

## 8. Swap coalescing

A model goes to sleep. Fifty requests arrive for it in the next 200 ms. Without
coalescing, fifty swap plans are computed and the GPU is asked to load the same weights
fifty times.

Two layers handle it:

- **At the plan level.** Once the model is in `WAKING`, the planner returns an empty
  plan for it, so later callers do not generate duplicate work.
- **At the call level.** `Controller.Ensure` keeps a map of in-flight swaps; the first
  caller performs the wake and the other forty-nine wait on the same channel.

The second layer exists because the first has a race: between planning and the state
flipping to `WAKING`, a concurrent caller can observe the old state. The test asserts
exactly one wake across fifty concurrent `Ensure` calls, and separately asserts that a
*failed* wake is reported to every waiter — returning success to the piggybackers would
route traffic at a model that is not there.

---

## 9. Concurrency-safe deploy jobs

Several control-plane replicas drain one job queue. Two mechanisms, and both are
necessary:

**`SELECT ... FOR UPDATE SKIP LOCKED`** lets workers claim distinct jobs without
blocking. Each claim takes a row lock and `SKIP LOCKED` makes competitors step over it.
Without `SKIP LOCKED`, N workers serialise behind one row and the queue stops being a
queue.

**A lease (`locked_by` + `lease_until`)** because the row lock dies with its
transaction, while the work takes minutes. A worker that crashes leaves a lease that
expires and the job becomes claimable again; a worker that is merely slow renews it.

One subtlety worth stating: `attempts` increments at **claim** time, not at failure
time. A worker killed mid-job never gets to report a failure, so counting only reported
failures would let a job that reliably crashes its worker retry forever.

And the matching guard: `CompleteJob`, `FailJob` and `RenewLease` are all conditioned
on `locked_by`. A worker that stalls past its lease and then wakes up must discover it
lost the job rather than marking the new owner's in-progress work as done.

---

## 10. Prefix-aware routing

With several replicas of one model, round-robin scatters consecutive turns of a
conversation across replicas, so vLLM's prefix cache rarely gets reused. Agent traffic —
a long system prompt plus tool definitions resent every turn — is mostly shared prefix,
so this is a large amount of wasted prefill.

The gateway hashes the prompt into **cumulative** leading blocks: block *k*'s hash
covers the prompt from the start through block *k*, not block *k* alone. Hashing blocks
independently would match two prompts that share their second block but differ in their
first — prompts that share no KV cache whatsoever.

Prefix matching stops at the first miss, because a KV hit requires an unbroken prefix.
Counting blocks past a gap would overstate affinity and send the request to a replica
that has to recompute the prefill anyway.

**When it hurts, and the fix.** Pure affinity sends every request for a popular prefix
to one replica, which queues while its peers idle. The `prefix-aware+load` variant
subtracts a load penalty, so affinity wins when replicas are comparably busy and loses
when one is saturated. Both are kept as separate arms because the hot-spot effect is
worth *showing* rather than asserting — there is a test that pins the pure-affinity arm
continuing to feed a replica with a queue depth of 100.

The gateway does not tokenise. Shipping a tokeniser per model and running it on every
request would add latency to the hot path for a routing hint. Blocks are 64 characters,
roughly one 16-token vLLM block. The approximation can only cause a cache miss or a
redundant affinity, never a wrong answer — and the prefix hit ratio in the results is
read from vLLM's own counters rather than from this estimate.

---

## 11. Measurement integrity

Several decisions exist only to stop the project from publishing numbers that are
wrong in a flattering direction.

- **The M0 gate checks output, not just timing.** vLLM's level-2 wake returns `200` in
  0.45 s and then generates `'!!!!!!!!!!!!'`, because level-2 sleep discards the
  weights and `wake_up` restores only buffers. The correct protocol adds
  `collective_rpc{"method": "reload_weights"}` and costs 1.87 s. The broken wake is
  **four times faster than the correct one**, so a timing-only benchmark would have put
  a garbage-producing tier at the top of the results table. Every wake is now checked
  against a pre-sleep reference output.
- **The gate refuses to grade what it could not measure.** NVML per-process accounting
  returns nothing under WSL2, so the first version compared 0 freed against 0 resident
  and printed PASS. It now detects that, falls back to device-wide memory minus a
  measured idle baseline, says which method it used, and fails if the footprint came
  out as zero.
- **Cold-start baselines use a warm page cache.** The first-ever load pays a disk read
  that no subsequent cold start pays. Using it would overstate sleep mode's advantage
  by about 13%.
- **TTFT is asserted to be well below total latency.** A buffering proxy would report
  TTFT equal to total, and every latency number in the project would then be a property
  of the gateway rather than of the model. The e2e asserts `ttft < 0.8 × total`.
- **TTFT is only recorded when it was genuinely measured.** A non-streaming request has
  no meaningful time-to-*first*-token; folding its total latency into the histogram
  would silently inflate the headline metric.
- **Percentiles use nearest-rank, not interpolation.** For a p99 over a few hundred
  samples, interpolation invents a value that was never observed.
- **Every arm replays an identical recorded trace.** If each arm generated its own
  traffic, a difference in p95 could be the policy or the random draw, with no way to
  tell.
- **Counters come from the gateway's `/metrics`, not from client-side inference.** A
  client cannot observe an eviction; inferring one from a latency spike is guessing.

---

## 12. What I would do next

- **Cross-node placement.** Placement today sorts agents by free VRAM, which is correct
  but unambitious. The interesting version co-locates models that are never hot
  simultaneously, which needs a correlation estimate the tracker does not keep.
- **Host RAM is the real ceiling.** VRAM decides how many models can be *active*; host
  RAM decides how many can be kept *warm*. Exceeding it gets the node OOM-killed rather
  than merely slowed, so it is a harder limit than VRAM, and the current budget is a
  flag rather than something adaptive.
- **The predictive policy is deliberately crude.** It extrapolates a decayed rate
  linearly over a horizon. Anything fancier would be fitting noise from a handful of
  requests, but with real traffic volume a per-model arrival model would be justified.
- **Tiered KV cache (LMCache).** Right now sleeping drops the KV arena entirely. Keeping
  hot prefixes across a sleep/wake cycle would make wakes cheaper than they already are.
- **Multi-LoRA.** Per-team adapters on a shared base model is a different and
  complementary answer to the same "more models than GPUs" problem — and the two
  compose.
