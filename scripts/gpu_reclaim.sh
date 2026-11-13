#!/usr/bin/env bash
# Kills anything Kiln left holding the GPU.
#
# A vLLM process is spawned into its own process group so the agent can signal the
# whole tree. If the agent itself is killed abruptly -- an interrupted benchmark, a
# crashed container -- those children survive and keep their VRAM allocation. The
# next run then sees a device that is mostly full with no owner it can account for,
# and every model fails to allocate a KV cache.
set -uo pipefail
echo "before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
ps -eo pid,etimes,args | grep -E 'vllm\.entrypoints|bin/(gateway|controlplane|nodeagent)' | grep -v grep || echo "  (none)"
pkill -f 'bin/gateway' 2>/dev/null || true
pkill -f 'bin/controlplane' 2>/dev/null || true
pkill -f 'bin/nodeagent' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
for i in $(seq 1 20); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$U" -lt 2000 ] && break
  sleep 1
done
echo "after:  $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
