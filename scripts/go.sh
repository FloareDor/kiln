#!/usr/bin/env bash
# Thin wrapper so every `go` invocation uses the WSL toolchain with its caches on the
# native filesystem. Building with GOCACHE on /mnt/c is roughly an order of magnitude
# slower because of 9p round-trips per object file.
#
# PATH and HOME are both reset first. When this is invoked as
# `wsl.exe -- bash scripts/go.sh`, the Windows shell's environment leaks in: HOME
# becomes a /c/Users/... path that Go rejects as relative, and PATH is full of Windows
# directories that shadow the Linux tools used below.
set -euo pipefail
export PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOME="$(eval echo "~$(id -un)")"
export HOME
export PATH="$PATH:$HOME/go/bin"
export GOCACHE="$HOME/.cache/kiln-go-build"
export GOMODCACHE="$HOME/.cache/kiln-go-mod"
export GOFLAGS="${GOFLAGS:--mod=mod}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec go "$@"
