#!/usr/bin/env bash
# Creates the vLLM virtualenv used for local (RTX 3070) development inside WSL2.
#
# The venv deliberately lives on the WSL native filesystem (~/.kiln-venv), not on
# /mnt/c: 9p-mounted Windows paths make multi-gigabyte installs and torch's import
# time pathologically slow. Only the source tree lives on /mnt/c.
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
VENV="${KILN_VENV:-$HOME/.kiln-venv}"

echo "==> pinning vllm==${VLLM_VERSION} into ${VENV}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install "vllm==${VLLM_VERSION}" nvidia-ml-py huggingface_hub openai

python - <<'PY'
import torch, vllm
print("vllm        :", vllm.__version__)
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0))
    print("capability  :", torch.cuda.get_device_capability(0))
    print("total VRAM  : %.2f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))
PY

echo "==> done. activate with: source ${VENV}/bin/activate"
