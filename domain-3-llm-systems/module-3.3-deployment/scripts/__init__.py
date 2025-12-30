"""
Module 3.3: Model Deployment & Inference Engines - Utility Scripts

This package provides utility modules for benchmarking, inference, and monitoring
of LLM deployment on DGX Spark and other NVIDIA platforms.

Available modules:
    benchmark_utils: Tools for benchmarking inference engines (Ollama, vLLM, TensorRT-LLM)
    inference_client: Unified client for multiple inference backends
    monitoring: GPU and server monitoring utilities

Example usage:
    from scripts.benchmark_utils import InferenceBenchmark, BenchmarkResult
    from scripts.inference_client import UnifiedInferenceClient, GenerationConfig
    from scripts.monitoring import GPUMonitor, InferenceMonitor

Note:
    These scripts are designed for educational purposes as part of the
    DGX Spark AI Curriculum. For production use, consider additional
    error handling and configuration options.
"""

from .benchmark_utils import (
    InferenceBenchmark,
    BenchmarkResult,
    BatchBenchmarkResult,
)
from .inference_client import (
    UnifiedInferenceClient,
    GenerationConfig,
    EngineType,
)
from .monitoring import (
    GPUMonitor,
    InferenceMonitor,
)

__all__ = [
    # Benchmark utilities
    "InferenceBenchmark",
    "BenchmarkResult",
    "BatchBenchmarkResult",
    # Inference client
    "UnifiedInferenceClient",
    "GenerationConfig",
    "EngineType",
    # Monitoring
    "GPUMonitor",
    "InferenceMonitor",
]

__version__ = "1.0.0"
