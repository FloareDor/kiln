#!/usr/bin/env bash
# Builds the Go binaries and runs an end-to-end acceptance test against the local GPU.
# Usage: bash scripts/run_e2e.sh m1 [extra args...]
set -euo pipefail
cd /mnt/c/CMU/kiln

MILESTONE="${1:-m1}"
shift || true

bash scripts/go.sh build -o bin/nodeagent ./cmd/nodeagent
bash scripts/go.sh build -o bin/gateway   ./cmd/gateway
bash scripts/go.sh build -o bin/controlplane ./cmd/controlplane 2>/dev/null || true

source "${KILN_VENV:-$HOME/.kiln-venv}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL=WARNING
exec python "scripts/e2e_${MILESTONE}.py" "$@"
