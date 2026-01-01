"""
DGX Spark Benchmark Utilities
=============================

Consolidated benchmarking tools for the DGX Spark AI Curriculum.
Provides specialized benchmarking for different use cases:

- General GPU/CPU timing (general.py)
- LLM inference with Ollama/llama.cpp (llm.py)
- LLM inference with HF models (inference.py)
- PyTorch operations (pytorch.py)
- Quantization metrics (quantization.py)
- LM Evaluation Harness wrappers (evaluation.py)

Usage:
    # Quick LLM benchmark via Ollama
    from utils.benchmarks import quick_benchmark
    result = quick_benchmark("qwen3:8b")

    # HF model inference benchmark
    from utils.benchmarks import benchmark_inference
    result = benchmark_inference(model, tokenizer, "Hello world")

    # General function timing
    from utils.benchmarks import benchmark_function, timed_section
    result = benchmark_function(my_func, args=(x,))

    # LM Eval Harness
    from utils.benchmarks import run_benchmark, get_benchmark_suite
    suite = get_benchmark_suite("quick")
    result = run_benchmark("microsoft/phi-2", suite["tasks"], "./results")
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
from .general import (
    TimingResult,
    BenchmarkTimer,
    benchmark_function,
    compare_implementations,
    format_benchmark_table,
    calculate_throughput,
    calculate_gflops,
    timed_section,
)
from .inference import (
    InferenceBenchmarkResult,
    benchmark_inference,
    benchmark_batch_inference,
    warmup_model,
    compare_models,
    ModelComparisonResult,
    benchmark_quantization_methods,
    format_inference_table,
)
from .evaluation import (
    EvalBenchmarkConfig,
    EvalBenchmarkResult,
    run_benchmark,
    compare_models_eval,
    format_eval_results,
    BENCHMARK_SUITES,
    get_benchmark_suite,
)

__all__ = [
    # Base classes
    "BenchmarkResult",
    "BenchmarkSummary",
    "BaseBenchmark",
    "get_gpu_memory_gb",
    "format_results_table",
    # LLM benchmarking (Ollama)
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
    # General timing
    "TimingResult",
    "BenchmarkTimer",
    "benchmark_function",
    "compare_implementations",
    "format_benchmark_table",
    "calculate_throughput",
    "calculate_gflops",
    "timed_section",
    # HF inference benchmarking
    "InferenceBenchmarkResult",
    "benchmark_inference",
    "benchmark_batch_inference",
    "warmup_model",
    "compare_models",
    "ModelComparisonResult",
    "benchmark_quantization_methods",
    "format_inference_table",
    # LM Eval Harness
    "EvalBenchmarkConfig",
    "EvalBenchmarkResult",
    "run_benchmark",
    "compare_models_eval",
    "format_eval_results",
    "BENCHMARK_SUITES",
    "get_benchmark_suite",
]
