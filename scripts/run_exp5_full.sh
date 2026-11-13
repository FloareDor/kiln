#!/usr/bin/env bash
# Experiment 5: the four routers on shared-prefix agent traffic.
#
# One model, several replicas: the question here is which *replica* serves a
# conversation, not which model, so the fleet is deliberately different from
# experiment 3's. Two replicas of a 0.5B model fit comfortably, which keeps the
# tiering scheduler out of the measurement entirely -- otherwise a swap would
# dominate the TTFT the routers are being compared on.
set -euo pipefail
exec bash "$(dirname "$0")/run_bench.sh" 5 \
  --replicas 2 \
  --duration 4m --users 4 \
  --models "q05a:Qwen/Qwen2.5-0.5B-Instruct:0.26" \
  "$@"
