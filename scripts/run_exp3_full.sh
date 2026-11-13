#!/usr/bin/env bash
# The headline experiment: five eviction policies across two traces.
#
# Fleet sizing is the whole game here, and getting it wrong produces a run that looks
# finished and measures nothing. The card is 8 GB with ~1.1 GB already held by the
# desktop, and each sleeping model keeps ~230 MiB of CUDA context, so the schedulable
# budget is about 6.3 GB.
#
# The fleet must satisfy two constraints at once:
#   * every individual model, plus at least one other, must fit -- otherwise the
#     scheduler can never admit it and every request for it is a 503;
#   * the total must exceed the budget, or there is no eviction to measure.
#
# Four models at 1.8-2.6 GB totalling ~8.0 GB against ~6.3 GB does both: three fit at
# once, the fourth forces an eviction. An earlier attempt included a 1.5B model at
# 4.1 GB, which alone took two thirds of the budget and could not coexist with
# anything else, so 97% of its requests were rejected outright.
#
# Sizes stay deliberately uneven: a uniform fleet makes cost-aware's per-GB
# normalisation indistinguishable from lfu-decay, so the experiment could not show the
# thing it exists to show.
set -euo pipefail
exec bash "$(dirname "$0")/run_bench.sh" 3 \
  --traces zipf_s10,bursty \
  --duration 5m \
  --models "q05a:Qwen/Qwen2.5-0.5B-Instruct:0.22,q06:Qwen/Qwen3-0.6B:0.32,q05b:Qwen/Qwen2.5-0.5B-Instruct:0.22,q05c:Qwen/Qwen2.5-0.5B-Instruct:0.26" \
  "$@"
