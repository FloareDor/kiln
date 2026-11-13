#!/usr/bin/env bash
# The headline experiment: five eviction policies across two traces.
#
# Fleet and load are calibrated for an 8 GB card with ~1.1 GB already taken by the
# desktop: four models of mixed size totalling ~9.7 GB against a 7.2 GB budget, so
# two or three are resident and the policy has a real choice to make. The sizes are
# deliberately heterogeneous, because a uniform fleet would make cost-aware's
# per-GB normalisation indistinguishable from lfu-decay.
set -euo pipefail
exec bash "$(dirname "$0")/run_bench.sh" 3 \
  --traces zipf_s10,bursty \
  --duration 3m --users 3 \
  --models "q05a:Qwen/Qwen2.5-0.5B-Instruct:0.22,q06:Qwen/Qwen3-0.6B:0.24,q05b:Qwen/Qwen2.5-0.5B-Instruct:0.22,q15:Qwen/Qwen2.5-1.5B-Instruct:0.50" \
  "$@"
