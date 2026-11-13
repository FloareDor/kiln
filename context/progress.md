# Progress

Kiln is a self-hosted LLM serving platform: deploy a Hugging Face model with one API
call, get an OpenAI-compatible streaming API with auth, quotas and Grafana monitoring,
running on Kubernetes. The feature it goes deep on is **GPU memory tiering** — serving
more models than fit in VRAM by paging idle ones out to CPU RAM (vLLM sleep L1) or disk
(sleep L2), with a pluggable eviction scheduler. Full spec and design rationale live in
`context/plan.md`. This file is updated every tick.

**End goal:** a working, benchmarked repo with a README good enough to carry an
interview — real numbers for sleep-vs-cold swapping, eviction policies and
prefix-aware routing, measured locally on an RTX 3070 and at scale on Modal.

Last updated: 2026-11-13

## Milestones
| # | Milestone | State |
|---|---|---|
| M0 | Sanity check: vLLM sleep/wake frees + restores VRAM | done |
| M1 | Gateway + node agent, one model, streaming | done |
| M2 | Postgres, auth, quotas, deploy jobs | done |
| M3 | Multiple replicas + Kubernetes | in progress |
| M4 | Observability + load testing | metrics done; Locust todo |
| M5 | Model swapping (the main feature) | policy core done; executor todo |
| M6 | Prefix-aware routing | todo |
| M7 | Modal scale run + README polish | todo |

## Done
**M0 - sleep mode verified.** vLLM 0.32.0 pinned. Sleep L1 and L2 each return 92% of a
model's resident VRAM; wakes cost 0.88 s and 1.87 s against a 34.3 s warm-cache cold
start (39x and 18x). Two findings recorded in `docs/PROGRESS.md`: level-2 wake is
silently broken without a `collective_rpc{"method":"reload_weights"}` and produces
garbage four times *faster* than the correct path, and NVML per-process accounting is
unavailable under WSL2 so the first version of the gate passed having measured nothing.

**M1 - gateway + node agent.** The stock OpenAI Python SDK streams from the platform.
Cold start 42.9 s, streaming TTFT 0.071 s against a 1.107 s total, sleep/wake through
the agent at 1.27 s / 0.59 s with byte-identical greedy output afterwards.

**Scheduler core (M5 logic, built early).** `internal/scheduler`: state machine, VRAM
and host-RAM accounting, EWMA demand tracker, all five eviction policies behind one
shared fitting loop. Pure, no I/O, 29 tests, no GPU.

**Routing core (M6 logic, built early).** `internal/routing`: all four routers and
cumulative prefix block hashing. Pure, 18 tests.

**M2 - Postgres, auth, quotas, crash-safe jobs.** All 14 acceptance checks pass.
One-call deploy reaches READY in 46 s and records its own measurements (VRAM 2867 MiB,
cold start 41.4 s, wake L1 587 ms, wake L2 710 ms). A 3 rpm team gets
`200 200 200 429 429 429` with `Retry-After: 19` while a 600 rpm team is unaffected. A
`SIGKILL`ed control plane's job is reclaimed from its expired lease and finished by a
replacement: DONE after 2 attempts. Three real bugs found and fixed, written up in
`docs/PROGRESS.md`.

## Next up
**M3 - multiple replicas + Kubernetes.** Manifests, Dockerfiles and the Makefile are
written (node agent as a DaemonSet owning the GPU; gateway and control plane as
Deployments; Postgres as a StatefulSet; Prometheus + Grafana). What remains: bring up
minikube with GPU passthrough, verify `make up`, and show requests spreading across
replicas. Open risk: 8 GB is tight for two replicas even of a 0.5B model at
`gpu_memory_utilization 0.35`, so the replica experiment may need `0.2` or Modal.

## Decisions
- **Dev environment is WSL2 Ubuntu, not native Windows.** vLLM has no Windows build.
  WSL2 has CUDA passthrough working (driver 610.47, RTX 3070 Laptop 8GB, 31GB host RAM,
  16 cores, 789GB free). Source stays at `C:\CMU\kiln` = `/mnt/c/CMU/kiln`; Go build and
  Python caches go on the WSL native filesystem because `/mnt/c` I/O is slow.
- **Node agent owns the GPU, not one pod per model.** Kubernetes grants a pod exclusive
  use of a GPU for the pod's whole lifetime, so one-pod-per-model pins one model per GPU
  forever. A single node-agent pod per GPU running vLLM as child processes is what makes
  swapping possible at all. The agent must also run standalone under plain Docker so the
  same binary works on Modal.
- **Policy code is pure.** Eviction policies and routers take state in and return
  decisions out — no HTTP, no vLLM calls — so the interesting logic is unit-testable
  offline with no GPU.

## Open questions
- Both tiers earn their place (L1 wakes about 2x faster than L2 but costs pinned host
  RAM), so `ChooseTier` has a real decision to make. Settled by M0.
- How many 0.5B-class models actually fit active on 8 GB at once? Needed to size the
  capacity experiment; measure during M3 when multiple replicas land.
- Modal GPU choice for the headline runs (A10G vs L4 vs A100) and which 7-8B model.
  Decide at M7; the benchmark scripts already target local or Modal via config.
