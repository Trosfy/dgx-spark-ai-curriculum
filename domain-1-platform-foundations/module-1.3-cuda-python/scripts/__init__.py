"""
Module 1.3: CUDA Python & GPU Programming - Utility Scripts

This package contains reusable utilities for GPU programming:
- cuda_utils: Core CUDA operations and helpers
- benchmark_utils: Benchmarking and timing utilities
- profiling_utils: GPU profiling helpers
"""

from .cuda_utils import (
    get_device_info,
    check_cuda_available,
    clear_gpu_memory,
    get_memory_info,
    optimal_block_size,
    optimal_grid_size,
)

from .benchmark_utils import (
    BenchmarkTimer,
    benchmark_function,
    compare_implementations,
    format_benchmark_table,
)

from .profiling_utils import (
    ProfileContext,
    memory_snapshot,
    print_memory_summary,
)

__all__ = [
    # cuda_utils
    "get_device_info",
    "check_cuda_available",
    "clear_gpu_memory",
    "get_memory_info",
    "optimal_block_size",
    "optimal_grid_size",
    # benchmark_utils
    "BenchmarkTimer",
    "benchmark_function",
    "compare_implementations",
    "format_benchmark_table",
    # profiling_utils
    "ProfileContext",
    "memory_snapshot",
    "print_memory_summary",
]
