#!/usr/bin/env bash
# Builds Kiln's container images and loads them into the local cluster.
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM_VERSION="${VLLM_VERSION:-0.32.0}"
PROFILE="${KILN_PROFILE:-kiln}"

echo "==> gateway"
docker build -q -f deploy/docker/Dockerfile.services --build-arg TARGET=gateway      -t kiln/gateway:dev .
echo "==> controlplane"
docker build -q -f deploy/docker/Dockerfile.services --build-arg TARGET=controlplane -t kiln/controlplane:dev .
echo "==> nodeagent (vLLM base image is large; this is the slow one)"
docker build -f deploy/docker/Dockerfile.nodeagent --build-arg VLLM_VERSION="$VLLM_VERSION" -t kiln/nodeagent:dev .

echo "==> loading into minikube profile '$PROFILE'"
minikube -p "$PROFILE" image load kiln/gateway:dev
minikube -p "$PROFILE" image load kiln/controlplane:dev
minikube -p "$PROFILE" image load kiln/nodeagent:dev
echo "==> done"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep '^kiln/'
