#!/usr/bin/env bash
# Loads locally built images into the minikube node.
#
# `minikube image load` is unusable on Windows 11: it shells out to wmic to translate
# a drive letter into a volume name, and wmic was removed from the OS. Streaming the
# image straight into the node's docker daemon over SSH sidesteps the whole Windows
# path-handling code path and needs no temporary 16GB tarball on disk.
set -euo pipefail
PROFILE="${KILN_PROFILE:-kiln}"

for img in "$@"; do
  echo "==> $img"
  docker save "$img" | minikube -p "$PROFILE" ssh --native-ssh=false -- docker load
done

echo "==> images present on the node:"
minikube -p "$PROFILE" ssh --native-ssh=false -- docker images --format "'{{.Repository}}:{{.Tag}} {{.Size}}'" | grep kiln || true
