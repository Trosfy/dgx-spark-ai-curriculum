"""
Core Memory Utilities for DGX Spark

Provides memory monitoring, tracking, and estimation for DGX Spark's
unified memory architecture with 128GB shared between GPU and CPU.
"""

import gc
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Tuple

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class MemorySnapshot:
    """
    Snapshot of memory state across GPU and system.

    Attributes:
        timestamp: Unix timestamp when snapshot was taken
        gpu_allocated_gb: GPU memory allocated by PyTorch
        gpu_reserved_gb: GPU memory reserved by PyTorch
        gpu_free_gb: Free GPU memory
        system_total_gb: Total system RAM
        system_available_gb: Available system RAM
        system_buffers_gb: System buffer/cache memory
    """

    timestamp: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_free_gb: float
    system_total_gb: float
    system_available_gb: float
    system_buffers_gb: float

    def __str__(self) -> str:
        return (
            f"GPU: {self.gpu_allocated_gb:.2f}/{self.gpu_reserved_gb:.2f}GB | "
            f"System: {self.system_available_gb:.2f}/{self.system_total_gb:.2f}GB"
        )


def get_memory_snapshot() -> MemorySnapshot:
    """
    Get current memory state across GPU and system.

    Returns:
        MemorySnapshot with current memory metrics
    """
    # GPU memory
    if HAS_TORCH and torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_free = gpu_total - gpu_reserved
    else:
        gpu_allocated = gpu_reserved = gpu_free = 0

    # System memory via /proc/meminfo or free command
    try:
        mem_result = subprocess.run(
            ["free", "-b"], capture_output=True, text=True, timeout=5
        )
        mem_lines = mem_result.stdout.split("\n")
        mem_parts = mem_lines[1].split()

        system_total = float(mem_parts[1]) / 1e9
        system_available = float(mem_parts[6]) / 1e9 if len(mem_parts) > 6 else 0
        system_buffers = float(mem_parts[5]) / 1e9 if len(mem_parts) > 5 else 0
    except Exception:
        system_total = system_available = system_buffers = 0

    return MemorySnapshot(
        timestamp=time.time(),
        gpu_allocated_gb=gpu_allocated,
        gpu_reserved_gb=gpu_reserved,
        gpu_free_gb=gpu_free,
        system_total_gb=system_total,
        system_available_gb=system_available,
        system_buffers_gb=system_buffers,
    )


def print_memory_status(label: str = "Current") -> None:
    """
    Print formatted memory status.

    Args:
        label: Description label for the status
    """
    snapshot = get_memory_snapshot()

    print(f"\n{'─' * 50}")
    print(f"Memory Status: {label}")
    print(f"{'─' * 50}")
    print(f"GPU Allocated:     {snapshot.gpu_allocated_gb:>8.2f} GB")
    print(f"GPU Reserved:      {snapshot.gpu_reserved_gb:>8.2f} GB")
    print(f"GPU Free:          {snapshot.gpu_free_gb:>8.2f} GB")
    print(f"System Total:      {snapshot.system_total_gb:>8.2f} GB")
    print(f"System Available:  {snapshot.system_available_gb:>8.2f} GB")
    print(f"System Buffers:    {snapshot.system_buffers_gb:>8.2f} GB")
    print(f"{'─' * 50}\n")


def clear_all_memory(clear_buffer_cache: bool = False) -> None:
    """
    Aggressively clear all memory.

    Args:
        clear_buffer_cache: Whether to clear Linux buffer cache (requires sudo)
    """
    # Python garbage collection
    gc.collect()

    # PyTorch GPU cache
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Linux buffer cache
    if clear_buffer_cache:
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("Warning: Could not clear buffer cache (needs sudo)")

    gc.collect()
    print("✓ Memory cleared")


@contextmanager
def memory_tracked(label: str = "Operation"):
    """
    Context manager to track memory usage of a block.

    Usage:
        with memory_tracked("Model Loading"):
            model = load_model()
    """
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    before = get_memory_snapshot()
    start_time = time.time()

    try:
        yield
    finally:
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

        after = get_memory_snapshot()
        elapsed = time.time() - start_time

        gpu_delta = after.gpu_allocated_gb - before.gpu_allocated_gb
        system_delta = before.system_available_gb - after.system_available_gb

        peak_gpu = 0
        if HAS_TORCH and torch.cuda.is_available():
            peak_gpu = torch.cuda.max_memory_allocated() / 1e9

        print(f"\n{'═' * 50}")
        print(f"Memory Report: {label}")
        print(f"{'═' * 50}")
        print(f"Duration:          {elapsed:>8.2f} s")
        print(f"GPU Delta:         {gpu_delta:>+8.2f} GB")
        print(f"GPU Peak:          {peak_gpu:>8.2f} GB")
        print(f"System Delta:      {system_delta:>+8.2f} GB")
        print(f"{'═' * 50}\n")


def memory_tracker(func: Callable) -> Callable:
    """
    Decorator to track memory usage of a function.

    Usage:
        @memory_tracker
        def load_model():
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with memory_tracked(func.__name__):
            return func(*args, **kwargs)

    return wrapper


class MemoryMonitor:
    """
    Continuous memory monitor for long-running operations.

    Usage:
        monitor = MemoryMonitor(interval=1.0)
        monitor.start()
        # ... operations ...
        monitor.stop()
        monitor.print_summary()
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots = []
        self._running = False
        self._thread = None

    def _monitor_loop(self):
        while self._running:
            self.snapshots.append(get_memory_snapshot())
            time.sleep(self.interval)

    def start(self) -> None:
        """Start monitoring in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"Memory monitor started (interval: {self.interval}s)")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"Memory monitor stopped ({len(self.snapshots)} snapshots)")

    def print_summary(self) -> None:
        """Print summary of monitored memory usage."""
        if not self.snapshots:
            print("No snapshots collected")
            return

        gpu_values = [s.gpu_allocated_gb for s in self.snapshots]
        system_values = [s.system_available_gb for s in self.snapshots]

        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp

        print(f"\n{'═' * 50}")
        print("Memory Monitor Summary")
        print(f"{'═' * 50}")
        print(f"Duration:          {duration:.1f} s")
        print(f"Snapshots:         {len(self.snapshots)}")
        print(f"GPU Min:           {min(gpu_values):.2f} GB")
        print(f"GPU Max:           {max(gpu_values):.2f} GB")
        print(f"GPU Avg:           {sum(gpu_values)/len(gpu_values):.2f} GB")
        print(f"System Avail Min:  {min(system_values):.2f} GB")
        print(f"System Avail Max:  {max(system_values):.2f} GB")
        print(f"{'═' * 50}\n")

    @property
    def peak_memory_gb(self) -> float:
        """Get peak GPU memory usage."""
        if not self.snapshots:
            return 0.0
        return max(s.gpu_allocated_gb for s in self.snapshots)


# DGX Spark Memory Constants
DGX_SPARK_TOTAL_MEMORY_GB = 128  # GB unified memory


def estimate_model_memory(
    num_params_billions: float,
    dtype: str = "bf16",
    include_optimizer: bool = False,
    include_gradients: bool = False,
) -> float:
    """
    Estimate memory required for a model on DGX Spark.

    Args:
        num_params_billions: Number of parameters in billions
        dtype: Data type (fp32, fp16, bf16, int8, int4)
        include_optimizer: Include Adam optimizer states
        include_gradients: Include gradient storage

    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
        "fp8": 1,
        "fp4": 0.5,
        "q4_k": 0.5,
        "q8_0": 1,
    }

    param_bytes = bytes_per_param.get(dtype.lower(), 2)
    base_memory = num_params_billions * param_bytes

    total = base_memory

    if include_gradients:
        # Gradients are typically in fp32
        total += num_params_billions * 4

    if include_optimizer:
        # Adam has 2 states per parameter (momentum, variance)
        total += num_params_billions * 4 * 2

    return total


def can_fit_model(
    num_params_billions: float,
    dtype: str = "bf16",
    training: bool = False,
    safety_margin_gb: float = 10.0,
) -> Tuple[bool, str]:
    """
    Check if a model can fit in DGX Spark memory.

    Args:
        num_params_billions: Model size in billions
        dtype: Data type
        training: Whether this is for training (needs gradients + optimizer)
        safety_margin_gb: GB to reserve for other operations

    Returns:
        (can_fit, explanation)
    """
    available_gb = DGX_SPARK_TOTAL_MEMORY_GB - safety_margin_gb

    required = estimate_model_memory(
        num_params_billions,
        dtype,
        include_optimizer=training,
        include_gradients=training,
    )

    can_fit = required <= available_gb

    explanation = (
        f"{num_params_billions}B model in {dtype}: "
        f"{required:.1f}GB required, {available_gb:.1f}GB available"
    )

    if not can_fit:
        # Suggest alternatives
        if dtype.lower() in ["fp32", "bf16", "fp16"]:
            int4_required = estimate_model_memory(
                num_params_billions,
                "int4",
                include_optimizer=training,
                include_gradients=training,
            )
            explanation += f"\n  → Try int4 quantization: {int4_required:.1f}GB"

    return can_fit, explanation
