"""
Module 2.4: Efficient Architectures - Utility Scripts

This module provides utilities for working with Mamba (State Space Models)
and Mixture of Experts (MoE) architectures on DGX Spark.

Available modules:
- mamba_utils: Utilities for Mamba model loading, inference, and analysis
- moe_utils: Utilities for MoE model loading and expert analysis
- benchmark_utils: Performance benchmarking tools for architecture comparison

Example Usage:
    from scripts.mamba_utils import load_mamba_model, benchmark_mamba
    from scripts.moe_utils import load_moe_model, analyze_expert_activation
    from scripts.benchmark_utils import ArchitectureBenchmark, compare_architectures
"""

from .mamba_utils import (
    load_mamba_model,
    generate_with_mamba,
    benchmark_mamba_inference,
    visualize_state_evolution,
    get_mamba_memory_usage,
)

from .moe_utils import (
    load_moe_model,
    extract_expert_activations,
    analyze_expert_specialization,
    visualize_expert_distribution,
    get_router_weights,
)

from .benchmark_utils import (
    ArchitectureBenchmark,
    BenchmarkResult,
    compare_architectures,
    plot_benchmark_results,
    measure_memory_usage,
    measure_inference_speed,
)

__all__ = [
    # Mamba utilities
    "load_mamba_model",
    "generate_with_mamba",
    "benchmark_mamba_inference",
    "visualize_state_evolution",
    "get_mamba_memory_usage",
    # MoE utilities
    "load_moe_model",
    "extract_expert_activations",
    "analyze_expert_specialization",
    "visualize_expert_distribution",
    "get_router_weights",
    # Benchmark utilities
    "ArchitectureBenchmark",
    "BenchmarkResult",
    "compare_architectures",
    "plot_benchmark_results",
    "measure_memory_usage",
    "measure_inference_speed",
]

__version__ = "1.0.0"
__author__ = "DGX Spark AI Curriculum"
