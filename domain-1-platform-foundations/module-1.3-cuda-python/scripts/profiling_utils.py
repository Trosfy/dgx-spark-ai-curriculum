"""
Profiling Utilities for Module 1.3

Utilities for GPU profiling and memory analysis.

Example usage:
    >>> from profiling_utils import memory_snapshot, ProfileContext
    >>>
    >>> # Take memory snapshot
    >>> snapshot = memory_snapshot()
    >>> print(f"GPU memory: {snapshot['used_gb']:.2f} GB")
    >>>
    >>> # Profile a code block
    >>> with ProfileContext("training step"):
    ...     loss = train_step(model, batch)
"""

import time
import gc
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state."""
    timestamp: float
    label: str
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    free_bytes: int = 0
    total_bytes: int = 0

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / 1e6

    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / 1e9

    @property
    def utilization_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes * 100


def memory_snapshot(label: str = "snapshot") -> Dict[str, Any]:
    """
    Take a snapshot of current GPU memory usage.

    Args:
        label: Label for this snapshot

    Returns:
        Dictionary with memory information

    Example:
        >>> before = memory_snapshot("before")
        >>> # ... do work ...
        >>> after = memory_snapshot("after")
        >>> print(f"Memory increased by {after['allocated_gb'] - before['allocated_gb']:.2f} GB")
    """
    result = {
        "timestamp": time.time(),
        "label": label,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "free_bytes": 0,
        "total_bytes": 0,
        "allocated_mb": 0.0,
        "allocated_gb": 0.0,
        "utilization_pct": 0.0,
    }

    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            result["allocated_bytes"] = torch.cuda.memory_allocated()
            result["reserved_bytes"] = torch.cuda.memory_reserved()
            props = torch.cuda.get_device_properties(0)
            result["total_bytes"] = props.total_memory
            result["free_bytes"] = result["total_bytes"] - result["reserved_bytes"]
    except ImportError:
        pass

    # Try Numba
    if result["allocated_bytes"] == 0:
        try:
            from numba import cuda
            if cuda.is_available():
                ctx = cuda.current_context()
                free, total = ctx.get_memory_info()
                result["free_bytes"] = free
                result["total_bytes"] = total
                result["allocated_bytes"] = total - free
        except Exception:
            pass

    # Try CuPy
    if result["allocated_bytes"] == 0:
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            result["allocated_bytes"] = mempool.used_bytes()
            result["reserved_bytes"] = mempool.total_bytes()
            device = cp.cuda.Device(0)
            result["free_bytes"], result["total_bytes"] = device.mem_info
        except ImportError:
            pass

    # Calculate derived values
    if result["allocated_bytes"] > 0:
        result["allocated_mb"] = result["allocated_bytes"] / 1e6
        result["allocated_gb"] = result["allocated_bytes"] / 1e9
        if result["total_bytes"] > 0:
            result["utilization_pct"] = result["allocated_bytes"] / result["total_bytes"] * 100

    return result


class MemoryTracker:
    """
    Track GPU memory usage over time.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.snapshot("start")
        >>> # ... do work ...
        >>> tracker.snapshot("after_forward")
        >>> # ... more work ...
        >>> tracker.snapshot("after_backward")
        >>> tracker.print_summary()
    """

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []

    def snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        """Take a snapshot and store it."""
        snap = memory_snapshot(label)
        self.snapshots.append(snap)
        return snap

    def clear(self):
        """Clear all snapshots."""
        self.snapshots = []

    def print_summary(self):
        """Print a summary of all snapshots."""
        if not self.snapshots:
            print("No snapshots recorded")
            return

        print(f"\n{'Label':<30} {'Allocated (MB)':<15} {'Delta (MB)':<15}")
        print("-" * 60)

        prev_allocated = 0
        for snap in self.snapshots:
            delta = snap["allocated_mb"] - prev_allocated
            delta_str = f"{delta:+.1f}" if prev_allocated > 0 else "-"
            print(f"{snap['label']:<30} {snap['allocated_mb']:<15.1f} {delta_str:<15}")
            prev_allocated = snap["allocated_mb"]


def print_memory_summary():
    """
    Print a summary of current GPU memory usage.

    Example:
        >>> print_memory_summary()
        GPU Memory Summary
        ==================
        Allocated: 1.23 GB
        Reserved:  2.00 GB
        Free:      126.00 GB
        Total:     128.00 GB
        Utilization: 0.96%
    """
    snap = memory_snapshot("current")

    print("\nGPU Memory Summary")
    print("=" * 30)
    print(f"Allocated: {snap['allocated_gb']:.2f} GB")
    if snap.get("reserved_bytes", 0) > 0:
        print(f"Reserved:  {snap['reserved_bytes'] / 1e9:.2f} GB")
    print(f"Free:      {snap['free_bytes'] / 1e9:.2f} GB")
    print(f"Total:     {snap['total_bytes'] / 1e9:.2f} GB")
    print(f"Utilization: {snap['utilization_pct']:.2f}%")


class ProfileContext:
    """
    Context manager for profiling code blocks with timing and memory tracking.

    Example:
        >>> with ProfileContext("model forward", track_memory=True) as ctx:
        ...     output = model(input)
        >>> print(f"Time: {ctx.elapsed_ms:.2f} ms")
        >>> print(f"Memory delta: {ctx.memory_delta_mb:.2f} MB")
    """

    def __init__(
        self,
        name: str = "block",
        sync_cuda: bool = True,
        track_memory: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the profile context.

        Args:
            name: Name for this profiled block
            sync_cuda: Whether to synchronize CUDA before/after
            track_memory: Whether to track memory changes
            verbose: Whether to print results automatically
        """
        self.name = name
        self.sync_cuda = sync_cuda
        self.track_memory = track_memory
        self.verbose = verbose

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.memory_before: Optional[Dict] = None
        self.memory_after: Optional[Dict] = None

    def _cuda_sync(self):
        """Synchronize all CUDA devices."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

        try:
            from numba import cuda
            if cuda.is_available():
                cuda.synchronize()
        except ImportError:
            pass

        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        except ImportError:
            pass

    def __enter__(self):
        if self.sync_cuda:
            self._cuda_sync()

        if self.track_memory:
            self.memory_before = memory_snapshot("before")

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda:
            self._cuda_sync()

        self.end_time = time.perf_counter()

        if self.track_memory:
            self.memory_after = memory_snapshot("after")

        if self.verbose:
            self._print_results()

    def _print_results(self):
        """Print profiling results."""
        print(f"\n[{self.name}]")
        print(f"  Time: {self.elapsed_ms:.3f} ms")

        if self.track_memory and self.memory_before and self.memory_after:
            print(f"  Memory: {self.memory_before['allocated_mb']:.1f} → "
                  f"{self.memory_after['allocated_mb']:.1f} MB "
                  f"(Δ{self.memory_delta_mb:+.1f} MB)")

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000

    @property
    def memory_delta_bytes(self) -> int:
        """Memory change in bytes."""
        if not self.memory_before or not self.memory_after:
            return 0
        return self.memory_after["allocated_bytes"] - self.memory_before["allocated_bytes"]

    @property
    def memory_delta_mb(self) -> float:
        """Memory change in megabytes."""
        return self.memory_delta_bytes / 1e6


def clear_gpu_memory():
    """
    Clear GPU memory from all frameworks.

    Example:
        >>> clear_gpu_memory()
        >>> print("Memory cleared!")
    """
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except ImportError:
        pass

    try:
        from numba import cuda
        if cuda.is_available():
            cuda.current_context().reset()
    except Exception:
        pass


@contextmanager
def profile_section(
    name: str,
    sync_cuda: bool = True,
    track_memory: bool = False,
    verbose: bool = True,
):
    """
    Convenience context manager for profiling.

    Args:
        name: Section name
        sync_cuda: Sync CUDA
        track_memory: Track memory
        verbose: Print results

    Example:
        >>> with profile_section("data loading", track_memory=True):
        ...     data = load_data()
    """
    ctx = ProfileContext(name, sync_cuda, track_memory, verbose)
    with ctx:
        yield ctx


def profile_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 1,
    iterations: int = 5,
    track_memory: bool = False,
) -> Dict[str, Any]:
    """
    Profile a function with detailed metrics.

    Args:
        func: Function to profile
        args: Positional arguments
        kwargs: Keyword arguments
        warmup: Warmup iterations
        iterations: Profiled iterations
        track_memory: Track memory usage

    Returns:
        Dictionary with profiling results

    Example:
        >>> results = profile_function(model.forward, (x,), track_memory=True)
        >>> print(f"Mean time: {results['mean_ms']:.2f} ms")
        >>> print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    """
    kwargs = kwargs or {}
    times = []
    memory_peaks = []

    # Warmup
    for _ in range(warmup):
        with ProfileContext(sync_cuda=True):
            _ = func(*args, **kwargs)

    # Profiled iterations
    for i in range(iterations):
        with ProfileContext(sync_cuda=True, track_memory=track_memory) as ctx:
            _ = func(*args, **kwargs)
        times.append(ctx.elapsed_ms)

        if track_memory and ctx.memory_after:
            memory_peaks.append(ctx.memory_after["allocated_mb"])

    import numpy as np

    results = {
        "name": getattr(func, "__name__", "function"),
        "iterations": iterations,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "times": times,
    }

    if memory_peaks:
        results["peak_memory_mb"] = np.max(memory_peaks)
        results["mean_memory_mb"] = np.mean(memory_peaks)

    return results


if __name__ == "__main__":
    # Demo
    print("Profiling Utils Demo")
    print("=" * 50)

    # Memory snapshot
    print("\nCurrent memory:")
    print_memory_summary()

    # Profile a section
    print("\nProfiling numpy operations...")
    import numpy as np

    with profile_section("numpy matmul", track_memory=False):
        a = np.random.randn(1000, 1000)
        b = np.random.randn(1000, 1000)
        c = a @ b

    # Memory tracker
    print("\nMemory tracking demo:")
    tracker = MemoryTracker()
    tracker.snapshot("initial")
    large_array = np.random.randn(10000, 10000)
    tracker.snapshot("after allocation")
    del large_array
    gc.collect()
    tracker.snapshot("after deletion")
    tracker.print_summary()
