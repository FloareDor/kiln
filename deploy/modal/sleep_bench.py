"""Run Kiln's sleep-mode benchmark on a Modal GPU.

The local numbers come from an 8 GB laptop card, which caps the model size at about
1.5B and leaves no compute headroom. This runs the *same* measurement script against a
7-8B model on a datacentre GPU, so the two sets of numbers are directly comparable:
same vLLM version, same flags, same correctness check against a pre-sleep reference
string.

The point is the ratio at a realistic model size. A 7B model takes minutes to load
from scratch, so if sleep mode still wakes it in seconds, the case for tiering is much
stronger than it is at 0.5B — and if it does not, that is worth knowing too.

    modal run --detach deploy/modal/sleep_bench.py
    modal run --detach deploy/modal/sleep_bench.py --model Qwen/Qwen2.5-7B-Instruct --gpu A100-40GB

Results are written to a Modal volume and printed at the end; copy them into
docs/results/ with:

    modal volume get kiln-results modal_sleep_mode.json docs/results/
"""

from __future__ import annotations

import modal

VLLM_VERSION = "0.32.0"

# Build on vLLM's own image so the CUDA stack matches the version being measured.
# Installing vLLM onto a bare CUDA base works but risks a different driver/runtime
# pairing than the local run, which would make the comparison meaningless.
image = (
    modal.Image.from_registry(
        f"vllm/vllm-openai:v{VLLM_VERSION}",
        add_python="3.12",
    )
    .entrypoint([])  # the base image launches vLLM directly; we drive it ourselves
    .pip_install("nvidia-ml-py", "requests", "huggingface_hub")
    .env({
        # Gates /sleep, /wake_up and /collective_rpc. Without it those endpoints 404
        # and there is nothing to measure.
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "HF_HOME": "/cache/huggingface",
    })
    # The measurement script itself, unchanged from the one used locally.
    .add_local_file(
        __file__.replace("deploy/modal/sleep_bench.py", "scripts/verify_sleep_mode.py"),
        "/root/verify_sleep_mode.py",
        copy=True,
    )
)

app = modal.App("kiln-sleep-bench", image=image)

# Weights persist between runs. A 7B model is ~15 GB, and re-downloading it on every
# run would dominate both the wall clock and the cold-start measurement.
hf_cache = modal.Volume.from_name("kiln-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("kiln-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    volumes={"/cache/huggingface": hf_cache, "/results": results},
    timeout=60 * 60,
    # A 7B download plus several model loads needs room; the default is too small.
    memory=32 * 1024,
)
def bench(model: str = "Qwen/Qwen2.5-7B-Instruct", gpu_util: float = 0.85,
          max_model_len: int = 2048) -> dict:
    """Run the sleep-mode verification and return its verdict."""
    import json
    import subprocess
    import sys

    cmd = [
        sys.executable, "/root/verify_sleep_mode.py",
        "--model", model,
        "--gpu-util", str(gpu_util),
        "--max-model-len", str(max_model_len),
        "--out", "/results/modal_sleep_mode.json",
    ]
    print(f"running: {' '.join(cmd)}\n", flush=True)

    # Stream the child's output rather than capturing it: the run takes minutes and a
    # silent terminal makes a hung load indistinguishable from a slow one.
    proc = subprocess.run(cmd, check=False)

    out = {}
    try:
        with open("/results/modal_sleep_mode.json", encoding="utf-8") as fh:
            out = json.load(fh)
    except FileNotFoundError:
        pass

    results.commit()
    hf_cache.commit()

    if proc.returncode != 0:
        print(f"\nverification exited {proc.returncode} — the gate did not pass")
    return out.get("verdict", {})


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-7B-Instruct", gpu_util: float = 0.85,
         max_model_len: int = 2048) -> None:
    verdict = bench.remote(model=model, gpu_util=gpu_util, max_model_len=max_model_len)
    if not verdict:
        print("no verdict returned; check the Modal logs")
        return

    print("\n" + "=" * 72)
    print(f"{model} on Modal")
    print("-" * 72)
    for key in ("measurement", "resident_mib", "freed_by_l1_mib", "freed_by_l1_pct",
                "cold_start_warm_cache_s", "wake_l1_s", "wake_l2_with_reload_s",
                "speedup_l1_vs_cold", "speedup_l2_vs_cold", "PASS"):
        if key in verdict:
            print(f"  {key:<28} {verdict[key]}")
    print("=" * 72)
    print("\nfetch the full record with:")
    print("  modal volume get kiln-results modal_sleep_mode.json docs/results/")
