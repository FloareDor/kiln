#!/usr/bin/env python
"""M0 gate: prove vLLM sleep mode actually frees and restores GPU memory.

Kiln's headline feature assumes three things about the pinned vLLM version:

  1. ``POST /sleep?level=1`` releases the model's VRAM back to the driver (not merely
     to vLLM's own allocator), with the weights parked in pinned host memory.
  2. ``POST /sleep?level=2`` releases at least as much, discarding the weights.
  3. ``POST /wake_up`` restores a *working* model, meaningfully faster than a cold
     start.

Several vLLM releases have shipped a sleep mode that reported success while leaving
the arena mapped, so this runs before any scheduler code is trusted and its numbers
go into docs/PROGRESS.md.

Two measurement traps this script exists to avoid:

  * **Vacuous pass.** NVML per-process accounting
    (``nvmlDeviceGetComputeRunningProcesses``) returns an empty list under WSL2, so a
    gate written against per-process VRAM silently compares 0 against 0 and passes.
    The probe detects that and falls back to device-wide memory minus an idle
    baseline, and refuses to report a pass it could not measure.
  * **Timing a broken wake.** Level-2 sleep *discards* the weights; vLLM's
    ``wake_up`` restores the allocator and the model buffers but not the weights
    themselves (see ``GPUWorker.wake_up`` — it replays ``_sleep_saved_buffers`` and
    ``sleep_preserve_parameter_names`` only). A naive level-2 wake therefore returns
    in milliseconds and generates garbage. The real protocol is wake_up followed by
    ``collective_rpc{"method": "reload_weights"}``. This script measures both and
    checks the generated text against a pre-sleep reference, so a fast-but-wrong wake
    fails the gate instead of flattering the results.

Usage:
    python scripts/verify_sleep_mode.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import requests

try:
    import pynvml
except ImportError:  # nvidia-ml-py installs under the pynvml name
    pynvml = None

MIB = 1024 * 1024
PROMPT = "The capital of France is"


# --------------------------------------------------------------------------- NVML


class GPUProbe:
    """Device-wide and (where available) per-process VRAM for one device."""

    def __init__(self, index: int = 0) -> None:
        if pynvml is None:
            raise RuntimeError("nvidia-ml-py is not installed")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name
        self.total_mib = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total // MIB
        self.per_process_supported = self._probe_per_process()
        self.idle_baseline_mib = self.device_used_mib()

    def _probe_per_process(self) -> bool:
        """WSL2's NVML shim does not implement process enumeration. Detect it once so
        the gate can say 'measured device-wide' instead of silently reading zeros."""
        try:
            pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except Exception:
            return False
        # An empty list at startup is expected and tells us nothing; the real check
        # happens once a vLLM process is known to be resident.
        return True

    def device_used_mib(self) -> int:
        return pynvml.nvmlDeviceGetMemoryInfo(self.handle).used // MIB

    def proc_used_mib(self, pids: set[int]) -> int:
        if not pids:
            return 0
        total = 0
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except Exception:
            return 0
        for proc in procs:
            if proc.pid in pids and proc.usedGpuMemory:
                total += proc.usedGpuMemory // MIB
        return total

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def process_tree(root_pid: int) -> set[int]:
    """All pids descended from root_pid, inclusive, read from /proc. vLLM runs its
    engine core in a child process, so a single pid undercounts badly."""
    children: dict[int, list[int]] = {}
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))

    out, stack = set(), [root_pid]
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack.extend(children.get(pid, []))
    return out


# ------------------------------------------------------------------------- server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Sample:
    label: str
    device_used_mib: int
    vllm_mib: int  # device_used minus the idle baseline: the server's own footprint
    proc_used_mib: int  # NVML per-process, 0 where unsupported
    elapsed_s: float = 0.0
    note: str = ""


@dataclass
class Report:
    vllm_version: str
    torch_version: str
    gpu: str
    gpu_total_mib: int
    idle_baseline_mib: int
    per_process_nvml: bool
    model: str
    gpu_memory_utilization: float
    max_model_len: int
    samples: list[Sample] = field(default_factory=list)
    verdict: dict = field(default_factory=dict)


class VLLMServer:
    """Spawns `vllm serve` with sleep mode on and drives its dev endpoints."""

    def __init__(self, model: str, port: int, gpu_util: float, max_len: int, python: str) -> None:
        self.model, self.port = model, port
        self.base = f"http://127.0.0.1:{port}"
        self.gpu_util, self.max_len, self.python = gpu_util, max_len, python
        self.proc: subprocess.Popen | None = None

    def start(self) -> float:
        env = dict(os.environ)
        # Gates /sleep, /wake_up, /is_sleeping and /collective_rpc. Without it those
        # endpoints 404 and the entire feature is unreachable over HTTP.
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        cmd = [
            self.python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model, "--port", str(self.port),
            "--enable-sleep-mode",
            "--gpu-memory-utilization", str(self.gpu_util),
            "--max-model-len", str(self.max_len),
            # Graph capture adds ~20s to a cold start and is not the thing under
            # test; disabling it keeps cold-vs-wake an honest comparison rather than
            # a CUDA-graph comparison.
            "--enforce-eager",
            # 0.32.0 replaced --disable-log-requests with this negated pair.
            "--no-enable-log-requests",
        ]
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        self.await_ready(timeout=900)
        return time.perf_counter() - t0

    def await_ready(self, timeout: float = 300) -> None:
        deadline, last = time.time() + timeout, None
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited early with code {self.proc.returncode}")
            try:
                if requests.get(f"{self.base}/health", timeout=2).status_code == 200:
                    self.generate(1)
                    return
            except Exception as exc:
                last = exc
            time.sleep(0.5)
        raise TimeoutError(f"server not ready within {timeout}s (last: {last})")

    def generate(self, max_tokens: int = 8) -> str:
        r = requests.post(f"{self.base}/v1/completions",
                          json={"model": self.model, "prompt": PROMPT,
                                "max_tokens": max_tokens, "temperature": 0.0, "seed": 0},
                          timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def sleep(self, level: int, mode: str = "wait") -> None:
        requests.post(f"{self.base}/sleep", params={"level": level, "mode": mode},
                      timeout=600).raise_for_status()

    def wake(self) -> None:
        requests.post(f"{self.base}/wake_up", timeout=600).raise_for_status()

    def reload_weights(self) -> None:
        """Re-read the weights into the GPU after a level-2 sleep discarded them."""
        r = requests.post(f"{self.base}/collective_rpc",
                          json={"method": "reload_weights"}, timeout=900)
        r.raise_for_status()

    def is_sleeping(self) -> bool:
        return bool(requests.get(f"{self.base}/is_sleeping", timeout=10).json()["is_sleeping"])

    def pids(self) -> set[int]:
        return process_tree(self.proc.pid) if self.proc else set()

    def stop(self) -> None:
        if not self.proc:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc.wait(timeout=30)
        self.proc = None


# ---------------------------------------------------------------------- the test


def run(args: argparse.Namespace) -> Report:
    import torch
    import vllm

    probe = GPUProbe(args.device)
    report = Report(
        vllm_version=vllm.__version__, torch_version=torch.__version__,
        gpu=probe.name, gpu_total_mib=probe.total_mib,
        idle_baseline_mib=probe.idle_baseline_mib,
        per_process_nvml=probe.per_process_supported,
        model=args.model, gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
    )

    def sample(label: str, pids: set[int], elapsed: float = 0.0, note: str = "") -> Sample:
        dev = probe.device_used_mib()
        s = Sample(label, dev, max(dev - probe.idle_baseline_mib, 0),
                   probe.proc_used_mib(pids), round(elapsed, 2), note)
        report.samples.append(s)
        print(f"  {label:<30} device={s.device_used_mib:>5}  vllm={s.vllm_mib:>5} MiB"
              f"  {('%.2fs' % elapsed) if elapsed else '':>8}  {note}")
        return s

    sample("baseline (no server)", set())

    server = VLLMServer(args.model, free_port(), args.gpu_util, args.max_model_len, sys.executable)
    reference = ""
    checks: dict[str, bool] = {}
    try:
        print(f"\n[1/6] cold start {args.model}")
        cold = server.start()
        pids = server.pids()
        reference = server.generate(args.check_tokens)
        sample("ACTIVE (cold start)", pids, cold, f"ref={reference!r}")
        if probe.per_process_supported and probe.proc_used_mib(pids) == 0:
            # NVML answered but attributed nothing to a process we know is resident.
            report.per_process_nvml = False

        print("\n[2/6] sleep level 1 (weights -> pinned host memory)")
        t0 = time.perf_counter()
        server.sleep(1, mode="wait")
        sample("SLEEPING_L1", pids, time.perf_counter() - t0,
               f"is_sleeping={server.is_sleeping()}")

        print("\n[3/6] wake from level 1")
        t0 = time.perf_counter()
        server.wake()
        got = server.generate(args.check_tokens)
        l1_wake = time.perf_counter() - t0
        checks["l1_output_matches"] = got == reference
        sample("ACTIVE (woke from L1)", pids, l1_wake,
               f"out={got!r} {'OK' if checks['l1_output_matches'] else 'MISMATCH'}")

        print("\n[4/6] sleep level 2 (weights discarded)")
        t0 = time.perf_counter()
        server.sleep(2, mode="wait")
        sample("SLEEPING_L2", pids, time.perf_counter() - t0,
               f"is_sleeping={server.is_sleeping()}")

        print("\n[5/6] wake from level 2, WITHOUT reloading weights (expected to be wrong)")
        t0 = time.perf_counter()
        server.wake()
        naive = server.generate(args.check_tokens)
        naive_wake = time.perf_counter() - t0
        checks["l2_naive_output_matches"] = naive == reference
        sample("ACTIVE (L2 wake, no reload)", pids, naive_wake,
               f"out={naive!r} {'OK' if checks['l2_naive_output_matches'] else 'GARBAGE (expected)'}")

        print("\n[6/6] reload weights -> the correct level-2 wake")
        t0 = time.perf_counter()
        server.reload_weights()
        fixed = server.generate(args.check_tokens)
        reload_s = time.perf_counter() - t0
        checks["l2_reload_output_matches"] = fixed == reference
        sample("ACTIVE (L2 wake + reload)", pids, naive_wake + reload_s,
               f"out={fixed!r} {'OK' if checks['l2_reload_output_matches'] else 'MISMATCH'}")
    finally:
        server.stop()
        time.sleep(3)

    sample("after shutdown", set())

    # A second cold start, measured after the first, so the baseline is against a warm
    # page cache. Comparing a wake against a cold start that also paid for the
    # first-ever disk read of the weights would flatter sleep mode unfairly.
    print("\n[bonus] second cold start (warm page cache) for a fair baseline")
    server2 = VLLMServer(args.model, free_port(), args.gpu_util, args.max_model_len, sys.executable)
    try:
        sample("ACTIVE (2nd cold start)", server2.pids(), server2.start(), "warm page cache")
    finally:
        server2.stop()

    by = {s.label: s for s in report.samples}
    resident = by["ACTIVE (cold start)"].vllm_mib
    freed_l1 = resident - by["SLEEPING_L1"].vllm_mib
    freed_l2 = resident - by["SLEEPING_L2"].vllm_mib
    cold_warm = by["ACTIVE (2nd cold start)"].elapsed_s
    wake_l1 = by["ACTIVE (woke from L1)"].elapsed_s
    wake_l2 = by["ACTIVE (L2 wake + reload)"].elapsed_s

    measured = resident > 0  # refuse to grade what we could not measure
    report.verdict = {
        "measurement": "nvml per-process" if report.per_process_nvml else "device-wide minus idle baseline",
        "resident_mib": resident,
        "freed_by_l1_mib": freed_l1,
        "freed_by_l2_mib": freed_l2,
        "freed_by_l1_pct": round(100 * freed_l1 / resident, 1) if measured else None,
        "freed_by_l2_pct": round(100 * freed_l2 / resident, 1) if measured else None,
        "cold_start_s": by["ACTIVE (cold start)"].elapsed_s,
        "cold_start_warm_cache_s": cold_warm,
        "sleep_l1_s": by["SLEEPING_L1"].elapsed_s,
        "wake_l1_s": wake_l1,
        "sleep_l2_s": by["SLEEPING_L2"].elapsed_s,
        "wake_l2_naive_s": by["ACTIVE (L2 wake, no reload)"].elapsed_s,
        "wake_l2_with_reload_s": wake_l2,
        "speedup_l1_vs_cold": round(cold_warm / max(wake_l1, 1e-6), 2),
        "speedup_l2_vs_cold": round(cold_warm / max(wake_l2, 1e-6), 2),
        "checks": checks,
        "reference_output": reference,
        # The gate: memory must actually be measured and actually come back, and both
        # wakes must reproduce the pre-sleep output exactly.
        "PASS": bool(
            measured
            and freed_l1 >= 0.5 * resident
            and freed_l2 >= 0.5 * resident
            and checks.get("l1_output_matches")
            and checks.get("l2_reload_output_matches")
        ),
    }
    probe.shutdown()
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--gpu-util", type=float, default=0.35,
                   help="fraction of total VRAM vLLM may claim; small so two models fit on 8GB")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--check-tokens", type=int, default=12,
                   help="greedy tokens compared before and after each wake")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", default="docs/results/m0_sleep_mode.json")
    args = p.parse_args()

    print("=== Kiln M0: sleep mode verification ===\n")
    report = run(args)
    v = report.verdict

    print("\n" + "=" * 86)
    print(f"{'metric':<40}{'value':>46}")
    print("-" * 86)
    for k, val in [
        ("vLLM / torch", f"{report.vllm_version} / {report.torch_version}"),
        ("GPU", f"{report.gpu} ({report.gpu_total_mib} MiB)"),
        ("VRAM measurement", v["measurement"]),
        ("model", report.model),
        ("resident VRAM when ACTIVE", f"{v['resident_mib']} MiB"),
        ("freed by sleep L1", f"{v['freed_by_l1_mib']} MiB ({v['freed_by_l1_pct']}%)"),
        ("freed by sleep L2", f"{v['freed_by_l2_mib']} MiB ({v['freed_by_l2_pct']}%)"),
        ("cold start (cold cache)", f"{v['cold_start_s']} s"),
        ("cold start (warm cache)", f"{v['cold_start_warm_cache_s']} s"),
        ("sleep L1 / wake L1", f"{v['sleep_l1_s']} s / {v['wake_l1_s']} s"),
        ("sleep L2 / wake L2 (+reload)", f"{v['sleep_l2_s']} s / {v['wake_l2_with_reload_s']} s"),
        ("L2 wake without reload", f"{v['wake_l2_naive_s']} s -> WRONG OUTPUT"),
        ("wake L1 speedup vs cold", f"{v['speedup_l1_vs_cold']}x"),
        ("wake L2 speedup vs cold", f"{v['speedup_l2_vs_cold']}x"),
        ("output correct after L1 wake", v["checks"].get("l1_output_matches")),
        ("output correct after L2 wake", v["checks"].get("l2_reload_output_matches")),
        ("GATE", "PASS" if v["PASS"] else "FAIL"),
    ]:
        print(f"{k:<40}{str(val):>46}")
    print("=" * 86)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2))
    print(f"\nwrote {out}")
    return 0 if v["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
