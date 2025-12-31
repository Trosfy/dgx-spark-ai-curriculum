"""
Memory Utilities for DGX Spark Quantization Experiments

This module provides memory monitoring and management functions optimized
for DGX Spark's 128GB unified memory architecture.

Features:
- GPU memory tracking and reporting
- Automatic cache clearing
- Context manager for memory-aware operations
- Memory profiling for quantization comparisons

Example:
    >>> from memory_utils import get_gpu_memory, clear_memory, MemoryTracker
    >>>
    >>> # Check current memory
    >>> allocated, reserved = get_gpu_memory()
    >>> print(f"GPU: {allocated:.2f} GB allocated")
    >>>
    >>> # Track memory during operations
    >>> with MemoryTracker("Loading model") as tracker:
    ...     model = load_model()
    >>> print(f"Model used {tracker.delta_allocated:.2f} GB")
"""

import gc
import torch
import subprocess
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
import time


def get_gpu_memory() -> Tuple[float, float]:
    """
    Get current GPU memory usage in GB.

    Returns:
        Tuple of (allocated_gb, reserved_gb)
        - allocated: Memory actively used by tensors
        - reserved: Memory reserved by PyTorch (includes cached memory)

    Example:
        >>> allocated, reserved = get_gpu_memory()
        >>> print(f"Using {allocated:.2f} GB, reserved {reserved:.2f} GB")
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0.0, 0.0


def get_system_memory() -> Dict[str, float]:
    """
    Get system (CPU) memory information in GB.

    Returns:
        Dict with 'total', 'available', 'used', 'percent' keys

    Example:
        >>> mem = get_system_memory()
        >>> print(f"System: {mem['available']:.1f} GB available of {mem['total']:.1f} GB")
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1e9,
            'available': mem.available / 1e9,
            'used': mem.used / 1e9,
            'percent': mem.percent
        }
    except ImportError:
        # Fallback to reading /proc/meminfo on Linux
        try:
            with open('/proc/meminfo') as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(':')
                        value = int(parts[1]) / 1e6  # Convert KB to GB
                        meminfo[key] = value

            return {
                'total': meminfo.get('MemTotal', 0),
                'available': meminfo.get('MemAvailable', 0),
                'used': meminfo.get('MemTotal', 0) - meminfo.get('MemAvailable', 0),
                'percent': 100 * (1 - meminfo.get('MemAvailable', 0) / meminfo.get('MemTotal', 1))
            }
        except Exception:
            return {'total': 0, 'available': 0, 'used': 0, 'percent': 0}


def clear_memory(verbose: bool = False) -> Tuple[float, float]:
    """
    Clear GPU memory cache and run garbage collection.

    Args:
        verbose: If True, print memory before and after clearing

    Returns:
        Tuple of (allocated_before, allocated_after) in GB

    Example:
        >>> before, after = clear_memory(verbose=True)
        GPU Memory: 5.23 GB -> 0.12 GB (freed 5.11 GB)
    """
    before = get_gpu_memory()[0]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    after = get_gpu_memory()[0]

    if verbose:
        freed = before - after
        print(f"GPU Memory: {before:.2f} GB -> {after:.2f} GB (freed {freed:.2f} GB)")

    return before, after


def clear_buffer_cache() -> bool:
    """
    Clear Linux buffer cache for maximum available memory.

    Requires sudo privileges. Use before loading very large models.

    Returns:
        True if successful, False otherwise

    Example:
        >>> if clear_buffer_cache():
        ...     print("Buffer cache cleared!")

    Note:
        This is particularly important on DGX Spark before loading
        models >50GB to ensure maximum unified memory availability.
    """
    try:
        subprocess.run(
            ['sudo', 'sh', '-c', 'sync; echo 3 > /proc/sys/vm/drop_caches'],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def estimate_model_memory(
    num_params: int,
    dtype: str = 'fp16',
    include_optimizer: bool = False,
    include_gradients: bool = False
) -> float:
    """
    Estimate memory required for a model.

    Args:
        num_params: Number of parameters (e.g., 7e9 for 7B)
        dtype: Data type ('fp32', 'fp16', 'bf16', 'int8', 'int4', 'fp8', 'fp4')
        include_optimizer: Add memory for Adam optimizer states (2x params)
        include_gradients: Add memory for gradients (1x params)

    Returns:
        Estimated memory in GB

    Example:
        >>> mem = estimate_model_memory(7e9, dtype='int4')
        >>> print(f"7B model in INT4: ~{mem:.1f} GB")
        7B model in INT4: ~3.5 GB
    """
    bytes_per_param = {
        'fp32': 4.0,
        'fp16': 2.0,
        'bf16': 2.0,
        'int8': 1.0,
        'fp8': 1.0,
        'int4': 0.5,
        'fp4': 0.5,
        'nvfp4': 0.5,
    }

    dtype = dtype.lower()
    if dtype not in bytes_per_param:
        raise ValueError(f"Unknown dtype: {dtype}. Use one of {list(bytes_per_param.keys())}")

    base_memory = num_params * bytes_per_param[dtype] / 1e9

    total = base_memory
    if include_gradients:
        # Gradients are typically in FP32 or FP16
        total += num_params * 2 / 1e9
    if include_optimizer:
        # Adam: 2 states (m, v) in FP32
        total += num_params * 8 / 1e9

    return total


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: float
    gpu_allocated: float
    gpu_reserved: float
    system_available: float
    label: str = ""


class MemoryTracker:
    """
    Context manager for tracking memory usage during operations.

    Example:
        >>> with MemoryTracker("Loading Llama 70B") as tracker:
        ...     model = load_model("llama-70b")
        >>> print(f"Model used {tracker.delta_allocated:.2f} GB")
        >>> tracker.report()
    """

    def __init__(self, label: str = "Operation"):
        """
        Initialize memory tracker.

        Args:
            label: Description of the operation being tracked
        """
        self.label = label
        self.start_snapshot: Optional[MemorySnapshot] = None
        self.end_snapshot: Optional[MemorySnapshot] = None

    def __enter__(self) -> 'MemoryTracker':
        """Start tracking memory."""
        self.start_snapshot = self._take_snapshot("start")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Stop tracking memory."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_snapshot = self._take_snapshot("end")
        return False

    def _take_snapshot(self, label: str) -> MemorySnapshot:
        """Take a memory snapshot."""
        gpu_alloc, gpu_res = get_gpu_memory()
        sys_mem = get_system_memory()
        return MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated=gpu_alloc,
            gpu_reserved=gpu_res,
            system_available=sys_mem['available'],
            label=label
        )

    @property
    def delta_allocated(self) -> float:
        """GPU memory change in GB."""
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.gpu_allocated - self.start_snapshot.gpu_allocated
        return 0.0

    @property
    def delta_reserved(self) -> float:
        """GPU reserved memory change in GB."""
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.gpu_reserved - self.start_snapshot.gpu_reserved
        return 0.0

    @property
    def duration(self) -> float:
        """Operation duration in seconds."""
        if self.start_snapshot and self.end_snapshot:
            return self.end_snapshot.timestamp - self.start_snapshot.timestamp
        return 0.0

    def report(self) -> None:
        """Print a summary report of memory usage."""
        if not self.start_snapshot or not self.end_snapshot:
            print("No data to report (tracker not used as context manager)")
            return

        print(f"\n{'='*60}")
        print(f"Memory Report: {self.label}")
        print(f"{'='*60}")
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"\nGPU Memory:")
        print(f"  Before: {self.start_snapshot.gpu_allocated:.2f} GB allocated")
        print(f"  After:  {self.end_snapshot.gpu_allocated:.2f} GB allocated")
        print(f"  Delta:  {self.delta_allocated:+.2f} GB")
        print(f"\nSystem Memory:")
        print(f"  Before: {self.start_snapshot.system_available:.2f} GB available")
        print(f"  After:  {self.end_snapshot.system_available:.2f} GB available")
        print(f"{'='*60}\n")


def get_dgx_spark_info() -> Dict[str, Any]:
    """
    Get DGX Spark system information.

    Returns:
        Dict with GPU, memory, and capability information

    Example:
        >>> info = get_dgx_spark_info()
        >>> print(f"GPU: {info['gpu_name']}")
        >>> print(f"FP4 support: {info['supports_fp4']}")
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': None,
        'compute_capability': None,
        'total_memory_gb': None,
        'supports_bf16': False,
        'supports_fp8': False,
        'supports_fp4': False,
        'is_blackwell': False,
    }

    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        info['compute_capability'] = f"{cc[0]}.{cc[1]}"
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['supports_bf16'] = torch.cuda.is_bf16_supported()

        # FP8 requires Hopper (9.0+) or later
        info['supports_fp8'] = cc[0] >= 9

        # FP4 requires Blackwell (10.0+)
        info['supports_fp4'] = cc[0] >= 10
        info['is_blackwell'] = cc[0] >= 10

    return info


def print_dgx_spark_status() -> None:
    """
    Print formatted DGX Spark system status.

    Example:
        >>> print_dgx_spark_status()
        ============================================================
        DGX Spark System Status
        ============================================================
        GPU: NVIDIA GB10 Superchip
        Compute Capability: 10.0
        Total Memory: 128.0 GB
        ...
    """
    info = get_dgx_spark_info()

    print("=" * 60)
    print("DGX Spark System Status")
    print("=" * 60)

    if not info['cuda_available']:
        print("CUDA is not available!")
        return

    print(f"GPU: {info['gpu_name']}")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"Total Memory: {info['total_memory_gb']:.1f} GB")
    print()

    gpu_alloc, gpu_res = get_gpu_memory()
    print(f"GPU Memory Usage:")
    print(f"  Allocated: {gpu_alloc:.2f} GB")
    print(f"  Reserved:  {gpu_res:.2f} GB")
    print(f"  Free:      {info['total_memory_gb'] - gpu_res:.2f} GB")
    print()

    sys_mem = get_system_memory()
    print(f"System Memory:")
    print(f"  Total:     {sys_mem['total']:.1f} GB")
    print(f"  Available: {sys_mem['available']:.1f} GB")
    print(f"  Used:      {sys_mem['percent']:.1f}%")
    print()

    print("Feature Support:")
    print(f"  BF16: {'Yes' if info['supports_bf16'] else 'No'}")
    print(f"  FP8:  {'Yes' if info['supports_fp8'] else 'No'}")
    print(f"  FP4:  {'Yes (Blackwell!)' if info['supports_fp4'] else 'No'}")

    if info['is_blackwell']:
        print("\n  Blackwell GPU detected! Full NVFP4 support available.")

    print("=" * 60)


if __name__ == "__main__":
    print("Memory Utils Demo")
    print("=" * 50)

    # Print system status
    print_dgx_spark_status()

    # Estimate model sizes
    print("\nModel Memory Estimates:")
    for params, name in [(7e9, "7B"), (13e9, "13B"), (70e9, "70B")]:
        for dtype in ['fp16', 'int8', 'int4']:
            mem = estimate_model_memory(params, dtype)
            print(f"  {name} @ {dtype}: {mem:.1f} GB")

    # Demo memory tracker
    print("\nMemory Tracker Demo:")
    with MemoryTracker("Creating dummy tensor") as tracker:
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, 1000, device='cuda')
    tracker.report()

    # Cleanup
    if torch.cuda.is_available():
        del x
    clear_memory(verbose=True)
