#!/usr/bin/env bash
# Blocks until a marker file stops changing or a pattern appears.
f="$1"; pat="$2"; limit="${3:-120}"
for i in $(seq 1 "$limit"); do
  if grep -q "$pat" "$f" 2>/dev/null; then echo FOUND; exit 0; fi
  sleep 10
done
echo TIMEOUT; exit 1
