#!/usr/bin/env bash
# Thin wrapper so every `go` invocation uses the WSL toolchain with its caches on the
# native filesystem. Building with GOCACHE on /mnt/c is roughly an order of magnitude
# slower because of 9p round-trips per object file.
set -euo pipefail
export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin
export GOCACHE="${GOCACHE:-$HOME/.cache/kiln-go-build}"
export GOMODCACHE="${GOMODCACHE:-$HOME/.cache/kiln-go-mod}"
export GOFLAGS="${GOFLAGS:--mod=mod}"
cd /mnt/c/CMU/kiln
exec go "$@"
