"""
Profiling Utilities for Python Performance Optimization
========================================================

Tools for profiling and optimizing Python code, with a focus on
numerical computing and machine learning workloads.

This module is part of the DGX Spark AI Curriculum - Module 2.

Features:
- Simple timing decorators and context managers
- Memory usage tracking
- CPU profiling wrappers
- Comparison utilities for benchmarking
- DGX Spark specific optimizations

Example Usage:
    >>> from profiling_utils import Timer, profile_function, compare_implementations
    >>>
    >>> # Time a code block
    >>> with Timer("my operation"):
    ...     result = expensive_computation()
    >>>
    >>> # Profile a function
    >>> @profile_function
    >>> def my_function():
    ...     pass
    >>>
    >>> # Compare implementations
    >>> results = compare_implementations(
    ...     [loop_version, vectorized_version],
    ...     ['Loop', 'Vectorized'],
    ...     args=(large_array,)
    ... )

Author: Professor SPARK
Date: 2024
"""

__all__ = [
    'Timer',
    'TimingResult',
    'MemoryResult',
    'timeit',
    'profile_function',
    'memory_tracker',
    'compare_implementations',
    'get_array_memory',
    'estimate_computation_memory',
    'ProgressTimer',
    'check_numba_available',
    'suggest_optimizations',
]

import time
import functools
import cProfile
import pstats
import io
import gc
import sys
from typing import Callable, List, Tuple, Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np

# Try to import optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False


@dataclass
class TimingResult:
    """Container for timing results."""
    name: str
    total_time: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    n_runs: int

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Total: {self.total_time*1000:.3f} ms\n"
            f"  Mean:  {self.mean_time*1000:.3f} ± {self.std_time*1000:.3f} ms\n"
            f"  Range: [{self.min_time*1000:.3f}, {self.max_time*1000:.3f}] ms\n"
            f"  Runs:  {self.n_runs}"
        )


@dataclass
class MemoryResult:
    """Container for memory profiling results."""
    peak_memory_mb: float
    current_memory_mb: float
    allocated_mb: float

    def __str__(self) -> str:
        return (
            f"Memory Usage:\n"
            f"  Peak:      {self.peak_memory_mb:.2f} MB\n"
            f"  Current:   {self.current_memory_mb:.2f} MB\n"
            f"  Allocated: {self.allocated_mb:.2f} MB"
        )


class Timer:
    """
    A versatile timer for profiling code execution.

    Can be used as a context manager or decorator.

    Attributes:
        name: Descriptive name for the operation being timed
        verbose: If True, print timing information when exiting context

    Example as context manager:
        >>> with Timer("matrix multiply") as t:
        ...     result = np.dot(A, B)
        >>> print(f"Took {t.elapsed:.3f} seconds")

    Example as decorator:
        >>> @Timer("expensive function")
        >>> def compute():
        ...     pass
    """

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize the timer.

        Args:
            name: Name for the operation being timed
            verbose: Whether to print timing when exiting context
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> 'Timer':
        """Start timing when entering context."""
        gc.collect()  # Clean up before timing
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop timing and optionally print results."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        if self.verbose:
            self._print_result()

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(self.name or func.__name__, self.verbose):
                return func(*args, **kwargs)
        return wrapper

    def _print_result(self) -> None:
        """Print formatted timing result."""
        if self.elapsed < 1e-3:
            print(f"⏱️ {self.name}: {self.elapsed*1e6:.2f} μs")
        elif self.elapsed < 1:
            print(f"⏱️ {self.name}: {self.elapsed*1000:.2f} ms")
        else:
            print(f"⏱️ {self.name}: {self.elapsed:.3f} s")


def timeit(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    n_runs: int = 5,
    warmup: int = 1,
    name: Optional[str] = None
) -> TimingResult:
    """
    Time a function over multiple runs with statistics.

    Includes warmup runs and garbage collection between runs
    for accurate benchmarking.

    Args:
        func: Function to time
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        n_runs: Number of timed runs
        warmup: Number of warmup runs (not timed)
        name: Name for the result (defaults to function name)

    Returns:
        TimingResult with statistics

    Example:
        >>> result = timeit(np.dot, args=(A, B), n_runs=10)
        >>> print(result)
    """
    kwargs = kwargs or {}
    name = name or func.__name__

    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    times = np.array(times)

    return TimingResult(
        name=name,
        total_time=times.sum(),
        mean_time=times.mean(),
        std_time=times.std(),
        min_time=times.min(),
        max_time=times.max(),
        n_runs=n_runs
    )


def profile_function(
    func: Optional[Callable] = None,
    sort_by: str = 'cumulative',
    top_n: int = 20,
    print_results: bool = True
) -> Union[Callable, str]:
    """
    Profile a function using cProfile.

    Can be used as a decorator or called directly.

    Args:
        func: Function to profile (if using as decorator)
        sort_by: How to sort results ('cumulative', 'time', 'calls')
        top_n: Number of top functions to show
        print_results: Whether to print results

    Returns:
        Decorated function or profile string

    Example as decorator:
        >>> @profile_function
        >>> def expensive_operation():
        ...     pass

    Example direct call:
        >>> profile_function(my_function, sort_by='time')
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = f(*args, **kwargs)

            profiler.disable()

            # Format results
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats(sort_by)
            stats.print_stats(top_n)

            profile_output = stream.getvalue()

            if print_results:
                print(f"\n{'='*60}")
                print(f"Profile for {f.__name__}")
                print('='*60)
                print(profile_output)

            return result
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


@contextmanager
def memory_tracker(name: str = "Operation"):
    """
    Context manager to track memory allocation.

    Uses tracemalloc if available, falls back to psutil.

    Args:
        name: Name for the operation

    Yields:
        MemoryResult object (populated after context exits)

    Example:
        >>> with memory_tracker("load data") as mem:
        ...     data = load_large_dataset()
        >>> print(f"Allocated {mem.allocated_mb:.2f} MB")
    """
    result = MemoryResult(0, 0, 0)

    if HAS_TRACEMALLOC:
        tracemalloc.start()

        yield result

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.current_memory_mb = current / 1024 / 1024
        result.peak_memory_mb = peak / 1024 / 1024
        result.allocated_mb = peak / 1024 / 1024

    elif HAS_PSUTIL:
        process = psutil.Process()
        mem_before = process.memory_info().rss

        yield result

        mem_after = process.memory_info().rss

        result.current_memory_mb = mem_after / 1024 / 1024
        result.peak_memory_mb = mem_after / 1024 / 1024  # Approximate
        result.allocated_mb = (mem_after - mem_before) / 1024 / 1024

    else:
        yield result
        print("⚠️ Neither tracemalloc nor psutil available for memory tracking")

    print(f"📊 {name}: Peak {result.peak_memory_mb:.2f} MB, "
          f"Allocated {result.allocated_mb:.2f} MB")


def compare_implementations(
    functions: List[Callable],
    names: List[str],
    args: tuple = (),
    kwargs: Optional[dict] = None,
    n_runs: int = 5,
    warmup: int = 1,
    verify_equal: bool = True,
    print_results: bool = True
) -> Dict[str, TimingResult]:
    """
    Compare timing of multiple implementations.

    Useful for benchmarking different approaches to the same problem.

    Args:
        functions: List of functions to compare
        names: Names for each function
        args: Arguments to pass to each function
        kwargs: Keyword arguments for each function
        n_runs: Number of timed runs per function
        warmup: Number of warmup runs
        verify_equal: Check if all functions return equal results
        print_results: Whether to print comparison table

    Returns:
        Dict mapping names to TimingResult objects

    Example:
        >>> results = compare_implementations(
        ...     [loop_sum, np.sum, math.fsum],
        ...     ['Loop', 'NumPy', 'Math'],
        ...     args=(large_array,)
        ... )
    """
    kwargs = kwargs or {}

    if len(functions) != len(names):
        raise ValueError("Number of functions must match number of names")

    results = {}
    outputs = []

    # Time each implementation
    for func, name in zip(functions, names):
        result = timeit(func, args, kwargs, n_runs, warmup, name)
        results[name] = result

        # Store output for verification
        outputs.append(func(*args, **kwargs))

    # Verify outputs are equal
    if verify_equal and len(outputs) > 1:
        try:
            reference = outputs[0]
            all_equal = True
            for i, output in enumerate(outputs[1:], 1):
                if isinstance(reference, np.ndarray):
                    if not np.allclose(reference, output, rtol=1e-5):
                        all_equal = False
                        print(f"⚠️ Output mismatch: {names[0]} vs {names[i]}")
                else:
                    if reference != output:
                        all_equal = False
                        print(f"⚠️ Output mismatch: {names[0]} vs {names[i]}")

            if all_equal and print_results:
                print("✅ All implementations return equivalent results\n")
        except Exception as e:
            print(f"⚠️ Could not verify equality: {e}\n")

    # Print comparison table
    if print_results:
        # Find fastest
        fastest_name = min(results.keys(), key=lambda k: results[k].mean_time)
        fastest_time = results[fastest_name].mean_time

        print("=" * 70)
        print(f"{'Implementation':<20} {'Mean Time':<15} {'Std':<12} {'Speedup':<10}")
        print("=" * 70)

        for name, result in sorted(results.items(), key=lambda x: x[1].mean_time):
            speedup = result.mean_time / fastest_time

            # Format time appropriately
            if result.mean_time < 1e-3:
                time_str = f"{result.mean_time*1e6:.2f} μs"
                std_str = f"± {result.std_time*1e6:.2f} μs"
            elif result.mean_time < 1:
                time_str = f"{result.mean_time*1000:.2f} ms"
                std_str = f"± {result.std_time*1000:.2f} ms"
            else:
                time_str = f"{result.mean_time:.3f} s"
                std_str = f"± {result.std_time:.3f} s"

            speedup_str = "🏆 (fastest)" if speedup == 1.0 else f"{speedup:.1f}x slower"

            print(f"{name:<20} {time_str:<15} {std_str:<12} {speedup_str:<10}")

        print("=" * 70)

    return results


def get_array_memory(arr: np.ndarray) -> str:
    """
    Get human-readable memory size of a NumPy array.

    Args:
        arr: NumPy array

    Returns:
        Formatted string like "128.5 MB"

    Example:
        >>> print(get_array_memory(large_array))
        "256.0 MB"
    """
    bytes_size = arr.nbytes

    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.1f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / 1024**2:.1f} MB"
    else:
        return f"{bytes_size / 1024**3:.2f} GB"


def estimate_computation_memory(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    n_intermediate: int = 1
) -> str:
    """
    Estimate memory needed for a computation.

    Args:
        shape: Shape of the main array
        dtype: Data type
        n_intermediate: Number of intermediate arrays of same size

    Returns:
        Formatted memory estimate string

    Example:
        >>> print(estimate_computation_memory((10000, 10000), np.float32, 3))
        "Estimated memory: 1.5 GB (3 intermediate arrays)"
    """
    itemsize = np.dtype(dtype).itemsize
    base_size = np.prod(shape) * itemsize
    total_size = base_size * (1 + n_intermediate)

    size_str = get_array_memory(np.empty(1, dtype=np.uint8).view(np.uint8))

    if total_size < 1024 ** 3:
        size_str = f"{total_size / 1024**2:.1f} MB"
    else:
        size_str = f"{total_size / 1024**3:.2f} GB"

    return f"Estimated memory: {size_str} ({n_intermediate} intermediate arrays)"


class ProgressTimer:
    """
    Timer with progress tracking for iterative operations.

    Useful for timing loops and providing ETA estimates.

    Example:
        >>> timer = ProgressTimer(total=100, name="Training")
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     timer.update()
        >>> timer.finish()
    """

    def __init__(self, total: int, name: str = "Progress"):
        """
        Initialize progress timer.

        Args:
            total: Total number of iterations
            name: Name for the operation
        """
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = time.perf_counter()
        self.times: List[float] = []
        self.last_print = 0

    def update(self, n: int = 1) -> None:
        """
        Update progress and optionally print status.

        Args:
            n: Number of iterations completed
        """
        now = time.perf_counter()
        self.times.append(now)
        self.current += n

        # Print every 10% or at least every 5 seconds
        progress = self.current / self.total
        if progress - self.last_print >= 0.1 or (now - self.start_time - self.last_print * (now - self.start_time) / max(progress, 0.01)) > 5:
            self._print_progress()
            self.last_print = progress

    def _print_progress(self) -> None:
        """Print current progress with ETA."""
        elapsed = time.perf_counter() - self.start_time
        progress = self.current / self.total

        if progress > 0:
            eta = elapsed / progress * (1 - progress)
            eta_str = f"{eta:.1f}s" if eta < 60 else f"{eta/60:.1f}min"
        else:
            eta_str = "calculating..."

        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"\r{self.name}: [{bar}] {progress*100:.1f}% | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta_str}", end="")

    def finish(self) -> None:
        """Mark operation as complete and print summary."""
        elapsed = time.perf_counter() - self.start_time
        per_iter = elapsed / max(self.current, 1)

        print(f"\n✅ {self.name} complete: {self.current} iterations in {elapsed:.2f}s "
              f"({per_iter*1000:.2f} ms/iter)")


# Numba-related utilities
def check_numba_available() -> bool:
    """
    Check if Numba is available and working.

    Returns:
        True if Numba can be imported and compiled
    """
    try:
        import numba

        @numba.jit(nopython=True)
        def test_func(x):
            return x + 1

        test_func(1)  # Force compilation
        return True
    except Exception:
        return False


def suggest_optimizations(func: Callable) -> List[str]:
    """
    Analyze a function and suggest potential optimizations.

    This is a simple heuristic-based analyzer.

    Args:
        func: Function to analyze

    Returns:
        List of optimization suggestions
    """
    import inspect

    suggestions = []

    try:
        source = inspect.getsource(func)

        # Check for common anti-patterns
        if 'for ' in source and 'range' in source:
            suggestions.append(
                "🔄 Contains explicit loops - consider vectorization with NumPy"
            )

        if '.append(' in source:
            suggestions.append(
                "📝 Uses list.append() in loop - consider pre-allocating array"
            )

        if 'import ' in source:
            suggestions.append(
                "📦 Contains imports inside function - move to module level"
            )

        if source.count('for ') >= 2:
            suggestions.append(
                "🔁 Nested loops detected - consider np.einsum or broadcasting"
            )

        if 'float(' in source or 'int(' in source:
            suggestions.append(
                "🔢 Type conversions in loop - ensure input types are correct"
            )

        if not suggestions:
            suggestions.append("✅ No obvious anti-patterns detected")

    except Exception as e:
        suggestions.append(f"⚠️ Could not analyze function: {e}")

    return suggestions


if __name__ == "__main__":
    print("Profiling Utils Demo")
    print("=" * 50)

    # Demo Timer
    print("\n1. Timer demo:")
    with Timer("Matrix creation"):
        A = np.random.randn(1000, 1000)

    # Demo timeit
    print("\n2. timeit demo:")
    result = timeit(np.sum, args=(A,), n_runs=10)
    print(result)

    # Demo compare_implementations
    print("\n3. Implementation comparison:")

    def loop_sum(arr):
        total = 0
        for x in arr.flat:
            total += x
        return total

    small_array = np.random.randn(10000)

    results = compare_implementations(
        [loop_sum, np.sum, sum],
        ['Python Loop', 'NumPy', 'Built-in sum'],
        args=(small_array,),
        n_runs=3
    )

    # Demo memory tracking
    print("\n4. Memory tracking:")
    with memory_tracker("Create large array"):
        large = np.random.randn(5000, 5000)
    del large

    # Demo suggestions
    print("\n5. Optimization suggestions for loop_sum:")
    for suggestion in suggest_optimizations(loop_sum):
        print(f"   {suggestion}")

    print("\n" + "=" * 50)
    print("Demo complete!")
