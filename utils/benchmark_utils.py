#!/usr/bin/env python3
"""
LLM Benchmark Utilities for DGX Spark

This module provides backward-compatible access to benchmark utilities.
All functionality is now consolidated in utils.benchmarks package.

Usage:
    # Direct import (recommended)
    from utils.benchmarks import BenchmarkResult, OllamaBenchmark, quick_benchmark

    # Legacy import (still supported)
    from utils.benchmark_utils import BenchmarkResult, quick_benchmark

For new code, prefer importing from utils.benchmarks directly.
"""

# Re-export everything from the consolidated benchmarks package
# This maintains backward compatibility with existing code

from .benchmarks.base import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkSummary,
    format_results_table,
    get_gpu_memory_gb,
)
from .benchmarks.llm import (
    DEFAULT_PROMPTS,
    BenchmarkSuite,
    LlamaCppBenchmark,
    OllamaBenchmark,
    quick_benchmark,
)
from .benchmarks.pytorch import (
    PyTorchBenchmark,
    PyTorchBenchmarkResult,
    benchmark_operation,
    compare_precisions,
)
from .benchmarks.quantization import (
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
    "DEFAULT_PROMPTS",
    # PyTorch benchmarking
    "PyTorchBenchmark",
    "PyTorchBenchmarkResult",
    "benchmark_operation",
    "compare_precisions",
    # Quantization benchmarking
    "QuantizationBenchmarkResult",
    "benchmark_quantized_model",
    "compare_quantizations",
]


# CLI support for direct execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DGX Spark LLM Benchmark")
    parser.add_argument(
        "--models", nargs="+", default=["llama3.1:8b"], help="Models to benchmark"
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output file path"
    )

    args = parser.parse_args()

    prompt = DEFAULT_PROMPTS["long"]

    suite = BenchmarkSuite()
    results = suite.run_ollama_suite(
        args.models, prompt, max_tokens=128, runs=args.runs
    )
    suite.print_results(results)
    suite.save_results(args.output)
