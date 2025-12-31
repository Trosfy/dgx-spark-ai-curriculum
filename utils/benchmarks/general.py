"""
General GPU/CPU Benchmarking Utilities
======================================

Utilities for accurate GPU and CPU benchmarking with proper warmup,
CUDA synchronization, and statistical analysis.

Example:
    >>> from utils.benchmarks import benchmark_function, compare_implementations
    >>>
    >>> result = benchmark_function(my_gpu_function, (large_array,), warmup=3, iterations=10)
    >>> print(f"Time: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")
"""

import time
from typing import Callable, Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class TimingResult:
    """Container for timing benchmark results."""
    name: str
    times: List[float] = field(default_factory=list)
    iterations: int = 0
    warmup_iterations: int = 0

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        if not self.times:
            return 0.0
        if HAS_NUMPY:
            return float(np.mean(self.times)) * 1000
        return (sum(self.times) / len(self.times)) * 1000

    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        if not self.times or len(self.times) < 2:
            return 0.0
        if HAS_NUMPY:
            return float(np.std(self.times)) * 1000
        mean = sum(self.times) / len(self.times)
        variance = sum((t - mean) ** 2 for t in self.times) / len(self.times)
        return (variance ** 0.5) * 1000

    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return min(self.times) * 1000 if self.times else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        return max(self.times) * 1000 if self.times else 0.0

    @property
    def median_ms(self) -> float:
        """Median time in milliseconds."""
        if not self.times:
            return 0.0
        if HAS_NUMPY:
            return float(np.median(self.times)) * 1000
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2 * 1000
        return sorted_times[n//2] * 1000

    def __str__(self) -> str:
        return f"{self.name}: {self.mean_ms:.3f} ± {self.std_ms:.3f} ms"


class BenchmarkTimer:
    """
    Context manager for timing code blocks with GPU synchronization.

    Example:
        >>> timer = BenchmarkTimer("my_operation", sync_cuda=True)
        >>> with timer:
        ...     result = expensive_gpu_operation()
        >>> print(f"Time: {timer.elapsed_ms:.2f} ms")
    """

    def __init__(self, name: str = "operation", sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        if self.sync_cuda:
            self._cuda_sync()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda:
            self._cuda_sync()
        self.end_time = time.perf_counter()

    def _cuda_sync(self):
        """Synchronize CUDA if available."""
        try:
            from numba import cuda
            if cuda.is_available():
                cuda.synchronize()
        except ImportError:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

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


def _sync_cuda():
    """Helper to synchronize CUDA."""
    try:
        from numba import cuda
        if cuda.is_available():
            cuda.synchronize()
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def benchmark_function(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict = None,
    warmup: int = 3,
    iterations: int = 10,
    sync_cuda: bool = True,
    name: Optional[str] = None,
) -> TimingResult:
    """
    Benchmark a function with proper warmup and statistics.

    Args:
        func: Function to benchmark
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        sync_cuda: Whether to synchronize CUDA
        name: Name for the benchmark (defaults to function name)

    Returns:
        TimingResult with timing statistics

    Example:
        >>> import numpy as np
        >>>
        >>> def matmul(a, b):
        ...     return a @ b
        >>>
        >>> a = np.random.randn(1000, 1000).astype(np.float32)
        >>> b = np.random.randn(1000, 1000).astype(np.float32)
        >>>
        >>> result = benchmark_function(matmul, (a, b), iterations=10)
        >>> print(f"Time: {result.mean_ms:.2f} ± {result.std_ms:.2f} ms")
    """
    kwargs = kwargs or {}
    name = name or getattr(func, "__name__", "function")

    result = TimingResult(
        name=name,
        iterations=iterations,
        warmup_iterations=warmup,
    )

    def sync():
        if sync_cuda:
            _sync_cuda()

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)
        sync()

    # Timed iterations
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        _ = func(*args, **kwargs)
        sync()
        elapsed = time.perf_counter() - start
        result.times.append(elapsed)

    return result


def compare_implementations(
    implementations: Dict[str, Callable],
    args: Tuple = (),
    kwargs: Dict = None,
    warmup: int = 3,
    iterations: int = 10,
    baseline: Optional[str] = None,
) -> Dict[str, TimingResult]:
    """
    Compare multiple implementations of the same operation.

    Args:
        implementations: Dictionary mapping names to functions
        args: Positional arguments for all functions
        kwargs: Keyword arguments for all functions
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        baseline: Name of baseline implementation for speedup calculation

    Returns:
        Dictionary of TimingResults

    Example:
        >>> implementations = {
        ...     "numpy": lambda a, b: np.dot(a, b),
        ...     "cupy": lambda a, b: cp.dot(cp.asarray(a), cp.asarray(b)),
        ... }
        >>>
        >>> results = compare_implementations(implementations, (a, b), baseline="numpy")
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.mean_ms:.2f} ms")
    """
    kwargs = kwargs or {}
    results = {}

    for name, func in implementations.items():
        results[name] = benchmark_function(
            func, args, kwargs, warmup, iterations, name=name
        )

    return results


def format_benchmark_table(
    results: Dict[str, TimingResult],
    baseline: Optional[str] = None,
    show_std: bool = True,
) -> str:
    """
    Format benchmark results as a table.

    Args:
        results: Dictionary of TimingResults
        baseline: Name of baseline for speedup calculation
        show_std: Whether to show standard deviation

    Returns:
        Formatted table string
    """
    lines = []

    if show_std:
        header = f"{'Name':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12}"
    else:
        header = f"{'Name':<20} {'Mean (ms)':<12} {'Min (ms)':<12}"

    if baseline and baseline in results:
        header += f" {'Speedup':<12}"

    lines.append(header)
    lines.append("-" * len(header))

    baseline_time = results[baseline].mean_ms if baseline and baseline in results else None

    for name, result in results.items():
        if show_std:
            row = f"{name:<20} {result.mean_ms:<12.3f} {result.std_ms:<12.3f} {result.min_ms:<12.3f}"
        else:
            row = f"{name:<20} {result.mean_ms:<12.3f} {result.min_ms:<12.3f}"

        if baseline_time:
            speedup = baseline_time / result.mean_ms
            row += f" {speedup:<12.2f}x"

        lines.append(row)

    return "\n".join(lines)


def calculate_throughput(
    n_elements: int,
    time_seconds: float,
    element_size_bytes: int = 4,
) -> Dict[str, float]:
    """
    Calculate throughput metrics.

    Args:
        n_elements: Number of elements processed
        time_seconds: Time taken in seconds
        element_size_bytes: Size of each element in bytes

    Returns:
        Dictionary with throughput metrics

    Example:
        >>> throughput = calculate_throughput(1_000_000, 0.001, element_size_bytes=4)
        >>> print(f"Throughput: {throughput['gb_per_sec']:.1f} GB/s")
    """
    total_bytes = n_elements * element_size_bytes

    return {
        "elements_per_sec": n_elements / time_seconds,
        "bytes_per_sec": total_bytes / time_seconds,
        "mb_per_sec": total_bytes / time_seconds / 1e6,
        "gb_per_sec": total_bytes / time_seconds / 1e9,
    }


def calculate_gflops(n_operations: int, time_seconds: float) -> float:
    """
    Calculate GFLOPS (billions of floating-point operations per second).

    Args:
        n_operations: Number of floating-point operations
        time_seconds: Time taken in seconds

    Returns:
        GFLOPS value

    Example:
        >>> # For matrix multiply C = A @ B where A is MxK and B is KxN
        >>> # Operations = 2*M*N*K (multiply and add for each element)
        >>> flops = 2 * 1024 * 1024 * 1024
        >>> gflops = calculate_gflops(flops, 0.1)
        >>> print(f"Performance: {gflops:.1f} GFLOPS")
    """
    return n_operations / time_seconds / 1e9


@contextmanager
def timed_section(name: str = "section", sync_cuda: bool = True, verbose: bool = True):
    """
    Context manager for timing code sections with optional output.

    Args:
        name: Name of the section
        sync_cuda: Whether to synchronize CUDA
        verbose: Whether to print timing

    Example:
        >>> with timed_section("data loading"):
        ...     data = load_large_dataset()
        # Prints: "data loading: 1.234 ms"
    """
    timer = BenchmarkTimer(name, sync_cuda)
    with timer:
        yield timer

    if verbose:
        print(f"{name}: {timer.elapsed_ms:.3f} ms")


if __name__ == "__main__":
    print("General Benchmark Utils Demo")
    print("=" * 50)

    if HAS_NUMPY:
        import numpy as np

        a = np.random.randn(1000, 1000).astype(np.float32)
        b = np.random.randn(1000, 1000).astype(np.float32)

        result = benchmark_function(np.dot, (a, b), iterations=5)
        print(f"\nNumPy matmul: {result}")

        flops = 2 * 1000 * 1000 * 1000
        gflops = calculate_gflops(flops, result.mean_ms / 1000)
        print(f"Performance: {gflops:.1f} GFLOPS")

        print("\nTimed section demo:")
        with timed_section("numpy sum"):
            _ = np.sum(a)
    else:
        print("NumPy not available for demo")
