#!/usr/bin/env bash
# Installs the benchmark-side Python dependencies into the Kiln venv.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source "${KILN_VENV:-$HOME/.kiln-venv}/bin/activate"
uv pip install locust
python - <<'PY'
import locust, requests
print("locust  :", locust.__version__)
print("requests:", requests.__version__)
PY
