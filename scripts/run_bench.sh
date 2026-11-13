#!/usr/bin/env bash
# Runs one of the benchmark experiments against the local GPU.
#
#   bash scripts/run_bench.sh 2        sleep vs cold, per model size
#   bash scripts/run_bench.sh 3        the five eviction policies
#   bash scripts/run_bench.sh 5        the four routers on agent traffic
set -euo pipefail
cd /mnt/c/CMU/kiln

for d in /usr/lib/postgresql/*/bin; do PATH="$PATH:$d"; done
export PATH

bash scripts/go.sh build -trimpath -o bin/nodeagent    ./cmd/nodeagent
bash scripts/go.sh build -trimpath -o bin/gateway      ./cmd/gateway
bash scripts/go.sh build -trimpath -o bin/controlplane ./cmd/controlplane

source "${KILN_VENV:-$HOME/.kiln-venv}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL=WARNING
export KILN_DATABASE_URL="${KILN_DATABASE_URL:-postgres://kiln:kiln@127.0.0.1:5433/kiln?sslmode=disable}"

EXP="${1:-2}"
shift || true

# A clean registry per experiment: leftover rows from a previous run would let a
# model appear deployed when its process is gone, and the scheduler would budget
# against memory nothing is holding.
psql "$KILN_DATABASE_URL" -q -c \
  "TRUNCATE deploy_jobs, replicas, models CASCADE" >/dev/null 2>&1 || true

exec python -u bench/run_experiment.py --experiment "$EXP" "$@"
