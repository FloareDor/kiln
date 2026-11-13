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

## M5 (partial) — scheduler core

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
