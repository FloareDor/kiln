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
| M0 | Sanity check: vLLM sleep/wake frees + restores VRAM | in progress |
| M1 | Gateway + node agent, one model, streaming | todo |
| M2 | Postgres, auth, quotas, deploy jobs | todo |
| M3 | Multiple replicas + Kubernetes | todo |
| M4 | Observability + load testing | todo |
| M5 | Model swapping (the main feature) | todo |
| M6 | Prefix-aware routing | todo |
| M7 | Modal scale run + README polish | todo |

## Done
Nothing landed yet. Repo skeleton, git, `.gitignore`, plan and progress files exist.

## Next up
**M0 — sanity check.** Set up a vLLM venv inside WSL2, pin the version, and run
`scripts/verify_sleep_mode.py`: load a small model, measure cold start, call
`sleep(level=1)` / `sleep(level=2)` / `wake_up()`, and record VRAM before/after each
transition via NVML. The whole swapping feature is dead if sleep does not actually free
VRAM on the pinned version, so this gate comes before any scheduler code.

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
- Which exact vLLM version pins cleanly with working sleep mode on CUDA 12.x? Resolved
  by M0; record the version and the measured VRAM deltas here.
- Does sleep L1 on a 3070 Laptop (8GB, shared with the desktop compositor) free enough
  VRAM to be worth it, or is only L2 useful at this size? Decide after M0 numbers.
