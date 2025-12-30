"""
Module 3.2: Model Quantization & Optimization - Utility Scripts

This package provides reusable utilities for quantizing and benchmarking models
on DGX Spark with 128GB unified memory.

Components:
    - quantization_utils: Core quantization helpers
    - perplexity: Perplexity calculation utilities

For benchmarking and memory utilities, use the consolidated utils package:
    from utils.benchmarks import BenchmarkResult, benchmark_quantized_model
    from utils.monitoring import MemoryMonitor, get_memory_snapshot
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

from .perplexity import (
    calculate_perplexity,
    calculate_perplexity_batch,
)

# Re-export consolidated utilities for convenience
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.benchmarks import (
        BenchmarkResult,
        QuantizationBenchmarkResult,
        benchmark_quantized_model,
        compare_quantizations,
    )
    from utils.monitoring import (
        MemorySnapshot,
        get_memory_snapshot,
        print_memory_status,
        MemoryMonitor,
        clear_all_memory,
    )
except ImportError:
    # utils not available, running standalone
    pass

__all__ = [
    # Quantization
    'symmetric_quantize',
    'asymmetric_quantize',
    'dequantize',
    'get_quantization_error',
    'compute_optimal_scale',
    'simulate_quantization',
    # Perplexity
    'calculate_perplexity',
    'calculate_perplexity_batch',
    # Re-exported from utils
    'BenchmarkResult',
    'QuantizationBenchmarkResult',
    'benchmark_quantized_model',
    'compare_quantizations',
    'MemorySnapshot',
    'get_memory_snapshot',
    'print_memory_status',
    'MemoryMonitor',
    'clear_all_memory',
]
