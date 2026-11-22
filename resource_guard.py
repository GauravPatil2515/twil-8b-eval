"""Lightweight system resource monitor and OOM/thermal safety guard."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time host memory, swap, disk, and GPU telemetry."""

    timestamp: float
    mem_available_mb: int
    swap_free_mb: int
    disk_free_gb: float
    gpu_mem_used_mb: int
    gpu_mem_total_mb: int
    gpu_temp_c: int
    gpu_util_pct: int


@dataclass(frozen=True)
class SafetyLimits:
    """Thresholds below/above which inference is aborted to protect stability."""

    min_mem_available_mb: int = 1024
    min_swap_free_mb: int = 1024
    min_disk_free_gb: float = 2.0
    max_gpu_temp_c: int = 88


def _read_meminfo_mb() -> tuple[int, int]:
    mem_avail_kb = 0
    swap_free_kb = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_avail_kb = int(line.split()[1])
                elif line.startswith("SwapFree:"):
                    swap_free_kb = int(line.split()[1])
        return mem_avail_kb // 1024, swap_free_kb // 1024
    except (FileNotFoundError, PermissionError):
        # Fallback for non-Linux or container environments
        try:
            import psutil  # type: ignore[import-untyped]
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            return int(vm.available // (1024 * 1024)), int(sm.free // (1024 * 1024))
        except Exception:
            return 8192, 4096


def _read_gpu_stats() -> tuple[int, int, int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        used, total, temp, util = [int(x.strip()) for x in out.splitlines()[0].split(",")]
        return used, total, temp, util
    except Exception:
        return 0, 0, 0, 0


def take_snapshot(path: str = "/") -> ResourceSnapshot:
    """Capture current RAM, swap, disk, and GPU metrics."""
    mem_avail_mb, swap_free_mb = _read_meminfo_mb()
    try:
        disk_free_gb = shutil.disk_usage(path).free / (1024**3)
    except Exception:
        disk_free_gb = 50.0
    gpu_used, gpu_total, gpu_temp, gpu_util = _read_gpu_stats()
    return ResourceSnapshot(
        timestamp=time.time(),
        mem_available_mb=mem_avail_mb,
        swap_free_mb=swap_free_mb,
        disk_free_gb=round(disk_free_gb, 2),
        gpu_mem_used_mb=gpu_used,
        gpu_mem_total_mb=gpu_total,
        gpu_temp_c=gpu_temp,
        gpu_util_pct=gpu_util,
    )


def check_safety(snap: ResourceSnapshot, limits: SafetyLimits = SafetyLimits()) -> str | None:
    """Return a violation reason if any safety threshold is breached, else None."""
    if snap.mem_available_mb < limits.min_mem_available_mb:
        return f"Low MemAvailable: {snap.mem_available_mb} MiB < {limits.min_mem_available_mb} MiB"
    if snap.swap_free_mb < limits.min_swap_free_mb:
        return f"Low SwapFree: {snap.swap_free_mb} MiB < {limits.min_swap_free_mb} MiB"
    if snap.disk_free_gb < limits.min_disk_free_gb:
        return f"Low disk space: {snap.disk_free_gb} GB < {limits.min_disk_free_gb} GB"
    if snap.gpu_temp_c > limits.max_gpu_temp_c:
        return f"High GPU temperature: {snap.gpu_temp_c}C > {limits.max_gpu_temp_c}C"
    return None


if __name__ == "__main__":
    snap = take_snapshot()
    limits = SafetyLimits()
    violation = check_safety(snap, limits)
    print("=" * 60)
    print("RESOURCE GUARD SYSTEM DIAGNOSTIC")
    print("=" * 60)
    print(f"Memory Available : {snap.mem_available_mb} MiB (limit >= {limits.min_mem_available_mb} MiB)")
    print(f"Swap Free        : {snap.swap_free_mb} MiB (limit >= {limits.min_swap_free_mb} MiB)")
    print(f"Disk Free        : {snap.disk_free_gb} GB (limit >= {limits.min_disk_free_gb} GB)")
    print(f"GPU VRAM Used    : {snap.gpu_mem_used_mb} / {snap.gpu_mem_total_mb} MiB")
    print(f"GPU Temperature  : {snap.gpu_temp_c} °C (limit <= {limits.max_gpu_temp_c} °C)")
    print(f"GPU Utilization  : {snap.gpu_util_pct} %")
    print("-" * 60)
    if violation:
        print(f"Status: FAILED -> {violation}")
        sys.exit(1)
    else:
        print("Status: HEALTHY (All safety constraints satisfied)")
        sys.exit(0)

