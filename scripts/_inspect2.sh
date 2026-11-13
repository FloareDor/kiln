#!/usr/bin/env bash
set -uo pipefail
echo "whoami: $(id -un)  tmp: $(ls /tmp | wc -l) entries"
ls /tmp | head -20
echo "--- kiln-logs ---"
ls -la /tmp/kiln-logs 2>/dev/null | head -6
echo "--- any kiln bench logs anywhere ---"
find / -maxdepth 3 -name 'kiln-bench-*' 2>/dev/null | head -10
