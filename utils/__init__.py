"""
DGX Spark AI Curriculum Utilities
=================================

A consolidated collection of utility functions for working with NVIDIA DGX Spark,
including system information, memory management, benchmarking, and monitoring tools.

Subpackages:
    utils.benchmarks  - LLM, PyTorch, and quantization benchmarking
    utils.monitoring  - Memory and resource monitoring

Usage:
    # Quick imports
    from utils import get_system_info, print_memory_status, quick_benchmark

    # System info
    info = get_system_info()
    print(f"GPU: {info.gpu_name}")

    # Memory tracking
    from utils import memory_tracked
    with memory_tracked("Model Loading"):
        model = load_model()

    # Benchmarking
    from utils import quick_benchmark
    result = quick_benchmark("llama3.1:8b")

    # For specialized benchmarking, use subpackages directly:
    from utils.benchmarks import PyTorchBenchmark, compare_precisions
    from utils.monitoring import RealtimeMonitor
"""

# Core system utilities
from .dgx_spark_utils import (
    get_system_info,
    print_system_info,
    clear_buffer_cache,
    clear_gpu_memory,
    get_gpu_memory_usage,
    check_ngc_container,
    verify_environment,
    optimal_batch_size,
    recommended_quantization,
    SystemInfo,
)

# Memory utilities (re-exported from monitoring)
from .memory_utils import (
    get_memory_snapshot,
    print_memory_status,
    clear_all_memory,
    memory_tracked,
    memory_tracker,
    MemoryMonitor,
    MemorySnapshot,
    estimate_model_memory,
    can_fit_model,
)

# Benchmark utilities (re-exported from benchmarks)
from .benchmark_utils import (
    BenchmarkResult,
    BenchmarkSummary,
    OllamaBenchmark,
    LlamaCppBenchmark,
    BenchmarkSuite,
    quick_benchmark,
    get_gpu_memory_gb,
    PyTorchBenchmark,
    compare_precisions,
)

__version__ = "0.2.0"

__all__ = [
    # dgx_spark_utils
    "get_system_info",
    "print_system_info",
    "clear_buffer_cache",
    "clear_gpu_memory",
    "get_gpu_memory_usage",
    "check_ngc_container",
    "verify_environment",
    "optimal_batch_size",
    "recommended_quantization",
    "SystemInfo",

    # memory_utils
    "get_memory_snapshot",
    "print_memory_status",
    "clear_all_memory",
    "memory_tracked",
    "memory_tracker",
    "MemoryMonitor",
    "MemorySnapshot",
    "estimate_model_memory",
    "can_fit_model",

    # benchmark_utils
    "BenchmarkResult",
    "BenchmarkSummary",
    "OllamaBenchmark",
    "LlamaCppBenchmark",
    "BenchmarkSuite",
    "quick_benchmark",
    "get_gpu_memory_gb",
    "PyTorchBenchmark",
    "compare_precisions",
]
