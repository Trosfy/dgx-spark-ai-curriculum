"""
DGX Spark Benchmark Utilities
=============================

Consolidated benchmarking tools for the DGX Spark AI Curriculum.
Provides specialized benchmarking for different use cases:

- LLM inference (Ollama, llama.cpp)
- PyTorch operations
- Quantization metrics
- Model deployment

Usage:
    from utils.benchmarks import BenchmarkResult, OllamaBenchmark, quick_benchmark

    # Quick LLM benchmark
    result = quick_benchmark("llama3.1:8b")

    # Detailed benchmark
    bench = OllamaBenchmark()
    summary = bench.benchmark("llama3.1:8b", prompt, runs=5)
"""

from .base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkSummary,
    format_results_table,
    get_gpu_memory_gb,
)
from .llm import (
    BenchmarkSuite,
    LlamaCppBenchmark,
    OllamaBenchmark,
    quick_benchmark,
)
from .pytorch import (
    PyTorchBenchmark,
    benchmark_operation,
    compare_precisions,
)
from .quantization import (
    QuantizationBenchmarkResult,
    benchmark_quantized_model,
    compare_quantizations,
)

__all__ = [
    # Base classes
    "BenchmarkResult",
    "BenchmarkSummary",
    "BaseBenchmark",
    "get_gpu_memory_gb",
    "format_results_table",
    # LLM benchmarking
    "OllamaBenchmark",
    "LlamaCppBenchmark",
    "BenchmarkSuite",
    "quick_benchmark",
    # PyTorch benchmarking
    "PyTorchBenchmark",
    "benchmark_operation",
    "compare_precisions",
    # Quantization benchmarking
    "QuantizationBenchmarkResult",
    "benchmark_quantized_model",
    "compare_quantizations",
]
