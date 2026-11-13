#!/usr/bin/env bash
set -uo pipefail
echo "=== log files ==="
ls -la /tmp/ 2>/dev/null | grep -i kiln | head -20
echo
echo "=== tiering enabled lines ==="
grep -h "tiering enabled" /tmp/kiln-bench-gw-*.log 2>/dev/null | tail -3
echo
echo "=== swap activity ==="
grep -hE "swap plan|model slept|model woken|swap step failed" /tmp/kiln-bench-gw-*.log 2>/dev/null | tail -10
echo
echo "=== gateway warnings/errors ==="
grep -hE '"level":"(ERROR|WARN)"' /tmp/kiln-bench-gw-*.log 2>/dev/null | tail -6
echo
echo "=== gateway startup ==="
grep -h "gateway listening" /tmp/kiln-bench-gw-*.log 2>/dev/null | tail -2
