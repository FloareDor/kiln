#!/usr/bin/env bash
# Brings up a local GPU-enabled Kubernetes cluster.
#
# minikube with the docker driver rather than kind: kind needs NVIDIA's nvkind
# wrapper to expose a GPU, while minikube supports --gpus all directly on a Docker
# Desktop backend that already has the container toolkit wired up.
set -uo pipefail

PROFILE="${KILN_PROFILE:-kiln}"

if minikube -p "$PROFILE" status >/dev/null 2>&1; then
  echo "==> cluster '$PROFILE' already running"
else
  echo "==> starting minikube profile '$PROFILE' with GPU passthrough"
  minikube start -p "$PROFILE" \
    --driver docker --container-runtime docker --gpus all \
    --cpus 6 --memory 12288 --disk-size 60g || exit 1
fi

echo "==> installing the NVIDIA device plugin"
kubectl --context "$PROFILE" apply -f \
  https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml

# The DaemonSet's nodeSelector keys off this label. On a mixed cluster the GPU
# operator sets it; on single-node minikube nothing does, so set it here.
kubectl --context "$PROFILE" label node --all nvidia.com/gpu.present=true --overwrite

echo "==> waiting for the node to advertise nvidia.com/gpu"
for i in $(seq 1 60); do
  CAP=$(kubectl --context "$PROFILE" get nodes -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}' 2>/dev/null)
  if [ -n "$CAP" ] && [ "$CAP" != "0" ]; then
    echo "==> node advertises nvidia.com/gpu: $CAP"
    kubectl --context "$PROFILE" get nodes -o wide
    exit 0
  fi
  sleep 5
done

echo "node never advertised a GPU; device plugin logs:" >&2
kubectl --context "$PROFILE" -n kube-system logs -l name=nvidia-device-plugin-ds --tail=30 2>&1 | tail -30 >&2
exit 1
