#!/usr/bin/env bash
# Installs a modern Go toolchain inside WSL2.
#
# All Go work for Kiln happens in WSL rather than on the Windows host: the node agent
# spawns Linux vLLM processes, the containers are Linux, and keeping one toolchain
# means `go test` and the shipped binaries are built the same way.
set -euo pipefail

GO_VERSION="${GO_VERSION:-$(curl -s https://go.dev/VERSION?m=text | head -1)}"
ARCH="linux-amd64"
TARBALL="${GO_VERSION}.${ARCH}.tar.gz"

if command -v go >/dev/null 2>&1 && [ "$(go version | awk '{print $3}')" = "$GO_VERSION" ]; then
  echo "==> ${GO_VERSION} already installed"
  exit 0
fi

echo "==> installing ${GO_VERSION}"
curl -fsSLo "/tmp/${TARBALL}" "https://go.dev/dl/${TARBALL}"
rm -rf /usr/local/go
tar -C /usr/local -xzf "/tmp/${TARBALL}"
rm -f "/tmp/${TARBALL}"

grep -q '/usr/local/go/bin' "$HOME/.bashrc" 2>/dev/null || \
  echo 'export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin' >> "$HOME/.bashrc"

export PATH=$PATH:/usr/local/go/bin
go version
