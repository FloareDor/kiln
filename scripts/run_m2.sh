#!/usr/bin/env bash
# Builds the binaries and runs the M2 acceptance test against the local GPU + Postgres.
set -euo pipefail
cd /mnt/c/CMU/kiln

# Debian keeps psql's binaries off the default PATH when more than one cluster exists.
for d in /usr/lib/postgresql/*/bin; do PATH="$PATH:$d"; done
export PATH

bash scripts/go.sh build -o bin/nodeagent    ./cmd/nodeagent
bash scripts/go.sh build -o bin/gateway      ./cmd/gateway
bash scripts/go.sh build -o bin/controlplane ./cmd/controlplane

source "${KILN_VENV:-$HOME/.kiln-venv}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL=WARNING
export KILN_DATABASE_URL="${KILN_DATABASE_URL:-postgres://kiln:kiln@127.0.0.1:5433/kiln?sslmode=disable}"

exec python scripts/e2e_m2.py "$@"
