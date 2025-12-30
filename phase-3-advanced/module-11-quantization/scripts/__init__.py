"""
Module 11: Model Quantization & Optimization - Utility Scripts

This package provides reusable utilities for quantizing and benchmarking models
on DGX Spark with 128GB unified memory.

Components:
    - quantization_utils: Core quantization helpers
    - benchmark_utils: Performance benchmarking tools
    - memory_utils: Memory monitoring for DGX Spark
    - perplexity: Perplexity calculation utilities
"""

__version__ = "1.0.0"


from .quantization_utils import (
    symmetric_quantize,
    asymmetric_quantize,
    dequantize,
    get_quantization_error,
    compute_optimal_scale,
    simulate_quantization,
)

from .benchmark_utils import (
    BenchmarkResult,
    benchmark_inference_speed,
    benchmark_memory_usage,
    compare_models,
)

from .memory_utils import (
    get_gpu_memory,
    clear_memory,
    monitor_memory,
    MemoryTracker,
)

from .perplexity import (
    calculate_perplexity,
    calculate_perplexity_batch,
)

__all__ = [
    # Quantization
    'symmetric_quantize',
    'asymmetric_quantize',
    'dequantize',
    'get_quantization_error',
    'compute_optimal_scale',
    'simulate_quantization',
    # Benchmarking
    'BenchmarkResult',
    'benchmark_inference_speed',
    'benchmark_memory_usage',
    'compare_models',
    # Memory
    'get_gpu_memory',
    'clear_memory',
    'monitor_memory',
    'MemoryTracker',
    # Perplexity
    'calculate_perplexity',
    'calculate_perplexity_batch',
]
