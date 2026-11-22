"""Unit tests for system telemetry snapshot and safety watchdog."""

from __future__ import annotations

import time

from resource_guard import ResourceSnapshot, SafetyLimits, check_safety, take_snapshot


def test_take_snapshot_returns_valid_data() -> None:
    snap = take_snapshot()
    assert isinstance(snap, ResourceSnapshot)
    assert snap.mem_available_mb > 0
    assert snap.disk_free_gb >= 0.0
    assert snap.gpu_temp_c >= 0
    assert snap.timestamp <= time.time()


def test_check_safety_healthy() -> None:
    snap = ResourceSnapshot(
        timestamp=time.time(),
        mem_available_mb=4096,
        swap_free_mb=2048,
        disk_free_gb=10.0,
        gpu_mem_used_mb=1000,
        gpu_mem_total_mb=6144,
        gpu_temp_c=65,
        gpu_util_pct=10,
    )
    limits = SafetyLimits(
        min_mem_available_mb=1024,
        min_swap_free_mb=1024,
        min_disk_free_gb=2.0,
        max_gpu_temp_c=88,
    )
    assert check_safety(snap, limits) is None


def test_check_safety_mem_violation() -> None:
    snap = ResourceSnapshot(
        timestamp=time.time(),
        mem_available_mb=512,
        swap_free_mb=2048,
        disk_free_gb=10.0,
        gpu_mem_used_mb=1000,
        gpu_mem_total_mb=6144,
        gpu_temp_c=65,
        gpu_util_pct=10,
    )
    limits = SafetyLimits(min_mem_available_mb=1024)
    reason = check_safety(snap, limits)
    assert reason is not None
    assert "Low MemAvailable" in reason


def test_check_safety_thermal_violation() -> None:
    snap = ResourceSnapshot(
        timestamp=time.time(),
        mem_available_mb=4096,
        swap_free_mb=2048,
        disk_free_gb=10.0,
        gpu_mem_used_mb=1000,
        gpu_mem_total_mb=6144,
        gpu_temp_c=90,
        gpu_util_pct=99,
    )
    limits = SafetyLimits(max_gpu_temp_c=88)
    reason = check_safety(snap, limits)
    assert reason is not None
    assert "High GPU temperature" in reason
