"""
Module 3.2: Model Quantization & Optimization - Utility Scripts

This package provides reusable utilities for quantizing and benchmarking models
on DGX Spark with 128GB unified memory.

Components:
    - quantization_utils: Core quantization helpers (INT4/8, FP8, FP4)
    - memory_utils: Memory monitoring and management
    - benchmark_utils: Inference benchmarking
    - perplexity: Perplexity calculation utilities

Example:
    >>> from scripts.quantization_utils import symmetric_quantize, quantize_to_fp4
    >>> from scripts.memory_utils import get_gpu_memory, MemoryTracker
    >>> from scripts.benchmark_utils import benchmark_inference
    >>> from scripts.perplexity import calculate_perplexity
"""

__version__ = "2.0.0"

from .quantization_utils import (
    symmetric_quantize,
    asymmetric_quantize,
    dequantize,
    get_quantization_error,
    compute_optimal_scale,
    simulate_quantization,
    # FP8 support (Blackwell native)
    quantize_to_fp8,
    dequantize_from_fp8,
    FP8_E4M3,
    FP8_E5M2,
    # FP4 support (Blackwell exclusive)
    quantize_to_fp4,
    dequantize_from_fp4,
    # Group quantization
    group_quantize,
    group_dequantize,
    # Analysis
    compare_quantization_methods,
)

from .perplexity import (
    calculate_perplexity,
    calculate_perplexity_batch,
    calculate_word_perplexity,
    perplexity_by_domain,
    compare_perplexity,
)

from .memory_utils import (
    get_gpu_memory,
    get_system_memory,
    clear_memory,
    clear_buffer_cache,
    estimate_model_memory,
    MemoryTracker,
    get_dgx_spark_info,
    print_dgx_spark_status,
)

from .benchmark_utils import (
    benchmark_inference,
    benchmark_batch_inference,
    compare_models,
    benchmark_quantization_methods,
    BenchmarkResult,
    ComparisonResult,
)


__all__ = [
    # Quantization - Basic
    'symmetric_quantize',
    'asymmetric_quantize',
    'dequantize',
    'get_quantization_error',
    'compute_optimal_scale',
    'simulate_quantization',
    # Quantization - FP8
    'quantize_to_fp8',
    'dequantize_from_fp8',
    'FP8_E4M3',
    'FP8_E5M2',
    # Quantization - FP4 (Blackwell)
    'quantize_to_fp4',
    'dequantize_from_fp4',
    # Quantization - Group
    'group_quantize',
    'group_dequantize',
    # Analysis
    'compare_quantization_methods',
    # Perplexity
    'calculate_perplexity',
    'calculate_perplexity_batch',
    'calculate_word_perplexity',
    'perplexity_by_domain',
    'compare_perplexity',
    # Memory
    'get_gpu_memory',
    'get_system_memory',
    'clear_memory',
    'clear_buffer_cache',
    'estimate_model_memory',
    'MemoryTracker',
    'get_dgx_spark_info',
    'print_dgx_spark_status',
    # Benchmarking
    'benchmark_inference',
    'benchmark_batch_inference',
    'compare_models',
    'benchmark_quantization_methods',
    'BenchmarkResult',
    'ComparisonResult',
]
