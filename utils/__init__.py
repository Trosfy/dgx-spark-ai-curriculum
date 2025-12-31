"""
DGX Spark AI Curriculum Utilities
=================================

A consolidated collection of utility functions for working with NVIDIA DGX Spark,
including system information, memory management, benchmarking, training, and visualization.

Subpackages:
    utils.benchmarks     - LLM, PyTorch, and quantization benchmarking
    utils.monitoring     - Memory and resource monitoring
    utils.system         - System info and environment utilities
    utils.training       - NumPy and HuggingFace training utilities
    utils.visualization  - ML visualization tools

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

    # Benchmarking (Ollama)
    from utils import quick_benchmark
    result = quick_benchmark("llama3.1:8b")

    # For specialized tools, use subpackages directly:
    from utils.benchmarks import benchmark_inference, run_benchmark
    from utils.training import create_training_args, EarlyStopping
    from utils.visualization import MLVisualizer, plot_training_curves
    from utils.monitoring import RealtimeMonitor
    from utils.system import verify_environment
"""

# Benchmark utilities
from .benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSummary,
    LlamaCppBenchmark,
    OllamaBenchmark,
    PyTorchBenchmark,
    compare_precisions,
    get_gpu_memory_gb,
    quick_benchmark,
)

# System utilities
from .system import (
    SystemInfo,
    check_ngc_container,
    clear_buffer_cache,
    clear_gpu_memory,
    get_gpu_memory_usage,
    get_system_info,
    optimal_batch_size,
    print_system_info,
    recommended_quantization,
    verify_environment,
)

# Memory/monitoring utilities
from .monitoring import (
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

__version__ = "0.3.0"

__all__ = [
    # system
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
    # monitoring
    "get_memory_snapshot",
    "print_memory_status",
    "clear_all_memory",
    "memory_tracked",
    "memory_tracker",
    "MemoryMonitor",
    "MemorySnapshot",
    "estimate_model_memory",
    "can_fit_model",
    # benchmarks
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
