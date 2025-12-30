"""
DGX Spark Monitoring Utilities
==============================

Consolidated monitoring tools for memory, GPU, and system resources.

Usage:
    from utils.monitoring import MemoryMonitor, get_memory_snapshot, print_memory_status

    # Quick status
    print_memory_status("Before loading model")

    # Continuous monitoring
    monitor = MemoryMonitor(interval=1.0)
    monitor.start()
    # ... do work ...
    monitor.stop()
    print(f"Peak memory: {monitor.peak_memory_gb:.2f} GB")

    # Context manager tracking
    with memory_tracked("Model Loading"):
        model = load_model()
"""

from .memory import (
    MemorySnapshot,
    get_memory_snapshot,
    print_memory_status,
    clear_all_memory,
    memory_tracked,
    memory_tracker,
    MemoryMonitor,
    estimate_model_memory,
    can_fit_model,
)

from .realtime import (
    Colors,
    RealtimeMonitor,
    format_bar,
    get_gpu_processes,
)

__all__ = [
    # Core memory utilities
    "MemorySnapshot",
    "get_memory_snapshot",
    "print_memory_status",
    "clear_all_memory",
    "memory_tracked",
    "memory_tracker",
    "MemoryMonitor",
    "estimate_model_memory",
    "can_fit_model",
    # Realtime monitoring
    "Colors",
    "RealtimeMonitor",
    "format_bar",
    "get_gpu_processes",
]
