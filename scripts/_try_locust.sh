#!/usr/bin/env bash
set -uo pipefail
cd /mnt/c/CMU/kiln
source "$HOME/.kiln-venv/bin/activate"
python -m locust -f bench/locustfile.py TraceUser \
  --host http://127.0.0.1:1 --headless -u 2 -r 1 -t 5s \
  --trace bench/traces/zipf_s10.jsonl --results /tmp/_probe.csv \
  --max-tokens 16 --only-summary 2>&1 | tail -20
