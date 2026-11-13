#!/usr/bin/env bash
# M0 gate runner: verifies vLLM sleep mode on the local GPU.
set -euo pipefail
source "${KILN_VENV:-$HOME/.kiln-venv}/bin/activate"
cd /mnt/c/CMU/kiln
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_LOGGING_LEVEL=WARNING
exec python scripts/verify_sleep_mode.py "$@"
