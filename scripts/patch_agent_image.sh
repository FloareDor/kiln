#!/usr/bin/env bash
# Rebuilds only the node agent's Go binary into the cluster's existing image.
#
# A full `docker save | minikube ssh docker load` of the node agent means streaming
# ~23 GB because the vLLM base dominates it. Only the final layer ever changes during
# development, so copy the new binary into the node and layer it on there instead.
# Seconds rather than half an hour.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${KILN_PROFILE:-kiln}"
MK="minikube -p $PROFILE ssh --native-ssh=false --"

echo "==> building nodeagent for linux/amd64"
# This script drives docker and kubectl, so it runs on the Windows side, while the Go
# toolchain lives in WSL. Dispatch the build across when we are not already in WSL.
# `uname -r` carries "microsoft" only inside WSL, which is the reliable test: checking
# for /mnt/c gets it backwards, because that path exists in WSL and not in Git Bash.
if uname -r | grep -qi microsoft; then
  bash scripts/go.sh build -trimpath -ldflags="-s -w" -o bin/nodeagent ./cmd/nodeagent
else
  wsl.exe -d Ubuntu -- bash /mnt/c/CMU/kiln/scripts/go.sh build -trimpath     -ldflags="-s -w" -o bin/nodeagent ./cmd/nodeagent
fi

echo "==> copying into the node"
# minikube cp lands the file owned by root, and the ssh user is not root, so the
# chmod and the build directory both need sudo.
minikube -p "$PROFILE" cp bin/nodeagent /tmp/nodeagent
$MK sudo chmod +x /tmp/nodeagent

echo "==> layering onto kiln/nodeagent:dev inside the node"
$MK "mkdir -p /tmp/agentbuild && cp /tmp/nodeagent /tmp/agentbuild/nodeagent && \
     printf 'FROM kiln/nodeagent:dev\nCOPY nodeagent /usr/local/bin/nodeagent\n' > /tmp/agentbuild/Dockerfile && \
     docker build -q -t kiln/nodeagent:dev /tmp/agentbuild"

echo "==> restarting the DaemonSet"
kubectl --context "$PROFILE" -n kiln rollout restart ds/kiln-nodeagent
kubectl --context "$PROFILE" -n kiln rollout status ds/kiln-nodeagent --timeout=300s
