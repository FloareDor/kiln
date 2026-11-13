#!/usr/bin/env bash
# Waits for the Kiln stack to become ready and prints what did not.
set -uo pipefail
CTX="${KILN_PROFILE:-kiln}"
K="kubectl --context $CTX -n kiln"

for i in $(seq 1 90); do
  READY=$($K get pods --no-headers 2>/dev/null | awk '{print $2}' | grep -c '^\([0-9]*\)/\1$' || true)
  TOTAL=$($K get pods --no-headers 2>/dev/null | wc -l)
  if [ "$TOTAL" -gt 0 ] && [ "$READY" -eq "$TOTAL" ]; then
    echo "all $TOTAL pods ready"
    $K get pods -o wide
    exit 0
  fi
  sleep 10
done

echo "=== not everything became ready ==="
$K get pods -o wide
echo
echo "=== events ==="
$K get events --sort-by=.lastTimestamp 2>/dev/null | tail -25
exit 1
