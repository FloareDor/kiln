#!/usr/bin/env bash
# Waits for the Docker daemon to accept connections.
set -uo pipefail
for i in $(seq 1 120); do
  if docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
    echo "docker ready: $(docker info --format '{{.ServerVersion}} / {{.OSType}}')"
    exit 0
  fi
  sleep 5
done
echo "docker did not become ready within 10 minutes" >&2
exit 1
