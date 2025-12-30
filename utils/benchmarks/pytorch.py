"""
PyTorch Operation Benchmarking Utilities

Specialized benchmarking for PyTorch operations on DGX Spark,
including precision comparison and operation timing.
"""

import time
import gc
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BenchmarkResult, BenchmarkSummary, get_gpu_memory_gb


@dataclass
class PyTorchBenchmarkResult:
    """
    Result container for PyTorch operation benchmarks.

    Attributes:
        operation: Name of the operation
        dtype: Data type used (float32, float16, bfloat16)
        input_shape: Shape of input tensor
        time_ms: Average execution time in milliseconds
        std_ms: Standard deviation of execution time
        memory_mb: Memory used in megabytes
        throughput: Operations per second (if applicable)
        metadata: Additional operation-specific data
    """
    operation: str
    dtype: str
    input_shape: tuple
    time_ms: float
    std_ms: float = 0.0
    memory_mb: float = 0.0
    throughput: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def speedup_vs(self, baseline: "PyTorchBenchmarkResult") -> float:
        """Calculate speedup compared to baseline."""
        if self.time_ms > 0:
            return baseline.time_ms / self.time_ms
        return 0.0


class PyTorchBenchmark:
    """
    Benchmark utility for PyTorch operations on DGX Spark.

    Measures execution time, memory usage, and throughput for various
    operations across different precisions.

    Example:
        >>> bench = PyTorchBenchmark()
        >>> result = bench.benchmark_matmul((4096, 4096), dtype=torch.bfloat16)
        >>> print(f"Time: {result.time_ms:.2f} ms")
    """

    def __init__(self, device: str = "cuda"):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for PyTorchBenchmark")
        self.device = device if torch.cuda.is_available() else "cpu"

    def _warmup(self, fn: Callable, warmup_runs: int = 3):
        """Run warmup iterations."""
        for _ in range(warmup_runs):
            fn()
            if self.device == "cuda":
                torch.cuda.synchronize()

    def _time_operation(
        self,
        fn: Callable,
        runs: int = 10
    ) -> tuple:
        """Time an operation over multiple runs."""
        times = []

        for _ in range(runs):
            if self.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            fn()

            if self.device == "cuda":
                torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)  # ms

        import statistics
        return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0

    def benchmark_matmul(
        self,
        shape: tuple = (4096, 4096),
        dtype: Any = None,
        runs: int = 10,
        warmup: int = 3
    ) -> PyTorchBenchmarkResult:
        """
        Benchmark matrix multiplication.

        Args:
            shape: Matrix dimensions (M, N) or (M, K, N)
            dtype: Torch dtype (default: float32)
            runs: Number of benchmark runs
            warmup: Number of warmup runs
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        dtype = dtype or torch.float32

        if len(shape) == 2:
            m, n = shape
            k = n
        else:
            m, k, n = shape

        a = torch.randn(m, k, dtype=dtype, device=self.device)
        b = torch.randn(k, n, dtype=dtype, device=self.device)

        def fn():
            return torch.matmul(a, b)  # noqa: F821 - closure captures a, b

        self._warmup(fn, warmup)

        if self.device == "cuda":
            start_mem = torch.cuda.memory_allocated()

        avg_time, std_time = self._time_operation(fn, runs)

        if self.device == "cuda":
            end_mem = torch.cuda.memory_allocated()
            memory_mb = (end_mem - start_mem) / 1e6
        else:
            memory_mb = 0

        # Calculate TFLOPS
        flops = 2 * m * k * n  # multiply-add
        tflops = flops / (avg_time / 1000) / 1e12

        # Cleanup
        del a, b
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        dtype_name = str(dtype).split(".")[-1]
        return PyTorchBenchmarkResult(
            operation="matmul",
            dtype=dtype_name,
            input_shape=shape,
            time_ms=avg_time,
            std_ms=std_time,
            memory_mb=memory_mb,
            throughput=tflops,
            metadata={"tflops": tflops}
        )

    def benchmark_attention(
        self,
        batch_size: int = 8,
        seq_len: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        dtype: Any = None,
        runs: int = 10,
        warmup: int = 3
    ) -> PyTorchBenchmarkResult:
        """
        Benchmark scaled dot-product attention.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dtype: Torch dtype
            runs: Number of benchmark runs
            warmup: Number of warmup runs
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        dtype = dtype or torch.float32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=self.device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=self.device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=self.device)

        def fn():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)  # noqa: F821

        self._warmup(fn, warmup)
        avg_time, std_time = self._time_operation(fn, runs)

        # Cleanup
        del q, k, v
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        dtype_name = str(dtype).split(".")[-1]
        return PyTorchBenchmarkResult(
            operation="attention",
            dtype=dtype_name,
            input_shape=(batch_size, num_heads, seq_len, head_dim),
            time_ms=avg_time,
            std_ms=std_time,
            metadata={"batch_size": batch_size, "seq_len": seq_len}
        )

    def benchmark_conv2d(
        self,
        batch_size: int = 32,
        in_channels: int = 64,
        out_channels: int = 128,
        size: int = 224,
        kernel_size: int = 3,
        dtype: Any = None,
        runs: int = 10,
        warmup: int = 3
    ) -> PyTorchBenchmarkResult:
        """Benchmark 2D convolution."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        dtype = dtype or torch.float32

        x = torch.randn(batch_size, in_channels, size, size, dtype=dtype, device=self.device)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1).to(self.device).to(dtype)

        def fn():
            return conv(x)  # noqa: F821 - closure captures conv, x

        self._warmup(fn, warmup)
        avg_time, std_time = self._time_operation(fn, runs)

        # Cleanup
        del x, conv
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        dtype_name = str(dtype).split(".")[-1]
        return PyTorchBenchmarkResult(
            operation="conv2d",
            dtype=dtype_name,
            input_shape=(batch_size, in_channels, size, size),
            time_ms=avg_time,
            std_ms=std_time,
            metadata={"kernel_size": kernel_size, "out_channels": out_channels}
        )


def benchmark_operation(
    operation: str,
    dtype: Any = None,
    **kwargs
) -> PyTorchBenchmarkResult:
    """
    Convenience function to benchmark a specific operation.

    Args:
        operation: One of "matmul", "attention", "conv2d"
        dtype: Torch dtype
        **kwargs: Operation-specific parameters

    Returns:
        PyTorchBenchmarkResult
    """
    bench = PyTorchBenchmark()

    if operation == "matmul":
        return bench.benchmark_matmul(dtype=dtype, **kwargs)
    elif operation == "attention":
        return bench.benchmark_attention(dtype=dtype, **kwargs)
    elif operation == "conv2d":
        return bench.benchmark_conv2d(dtype=dtype, **kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def compare_precisions(
    operation: str = "matmul",
    dtypes: Optional[List] = None,
    **kwargs
) -> Dict[str, PyTorchBenchmarkResult]:
    """
    Compare operation performance across different precisions.

    Args:
        operation: Operation to benchmark
        dtypes: List of dtypes to compare (default: fp32, fp16, bf16)
        **kwargs: Operation-specific parameters

    Returns:
        Dictionary mapping dtype name to benchmark result
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    if dtypes is None:
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)

    results = {}
    for dtype in dtypes:
        dtype_name = str(dtype).split(".")[-1]
        print(f"Benchmarking {operation} with {dtype_name}...")
        results[dtype_name] = benchmark_operation(operation, dtype=dtype, **kwargs)

    # Print comparison table
    print("\nPrecision Comparison Results:")
    print("-" * 60)
    print(f"{'Precision':<12} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    baseline = results.get("float32")
    for name, result in results.items():
        speedup = baseline.time_ms / result.time_ms if baseline else 1.0
        print(f"{name:<12} {result.time_ms:<15.2f} {speedup:<10.2f}x")

    return results
