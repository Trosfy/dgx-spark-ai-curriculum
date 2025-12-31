#!/usr/bin/env python3
"""
Memory Management Utilities for DGX Spark

This module provides backward-compatible access to memory utilities.
All functionality is now consolidated in utils.monitoring package.

Usage:
    # Direct import (recommended)
    from utils.monitoring import MemorySnapshot, memory_tracked, MemoryMonitor

    # Legacy import (still supported)
    from utils.memory_utils import get_memory_snapshot, print_memory_status

For new code, prefer importing from utils.monitoring directly.
"""

# Re-export everything from the consolidated monitoring package
from .monitoring.memory import (
    DGX_SPARK_TOTAL_MEMORY_GB,
    MemoryMonitor,
    MemorySnapshot,
    can_fit_model,
    clear_all_memory,
    estimate_model_memory,
    get_memory_snapshot,
    memory_tracked,
    memory_tracker,
    print_memory_status,
)

__all__ = [
    "MemorySnapshot",
    "get_memory_snapshot",
    "print_memory_status",
    "clear_all_memory",
    "memory_tracked",
    "memory_tracker",
    "MemoryMonitor",
    "estimate_model_memory",
    "can_fit_model",
    "DGX_SPARK_TOTAL_MEMORY_GB",
]


if __name__ == "__main__":
    # Demo
    print_memory_status("Startup")

    print("\nModel Memory Estimates:")
    for size in [7, 13, 34, 70, 120]:
        for dtype in ["bf16", "int8", "int4"]:
            mem = estimate_model_memory(size, dtype)
            print(f"  {size}B {dtype}: {mem:.1f} GB")

    print("\nCan fit on DGX Spark?")
    for size in [8, 13, 34, 70, 120]:
        can_fit, explanation = can_fit_model(size, "bf16", training=False)
        status = "✓" if can_fit else "✗"
        print(f"  {status} {explanation}")
