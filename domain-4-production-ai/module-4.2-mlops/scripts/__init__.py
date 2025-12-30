"""
Module 15: Benchmarking, Evaluation & MLOps

Utility scripts for LLM benchmarking, custom evaluation,
experiment tracking, model versioning, and reproducibility.

Components:
- benchmark_utils: LM Evaluation Harness helpers for running benchmarks
- evaluation_framework: Custom evaluation framework with LLM-as-judge
- mlflow_utils: MLflow tracking and experiment management utilities (requires mlflow)
- versioning_utils: Model and data versioning tools
- reproducibility: Reproducibility utilities for ML training

Example usage:
    # Benchmarking
    from scripts.benchmark_utils import run_benchmark, compare_models
    result = run_benchmark("microsoft/phi-2", ["hellaswag"], "./results")

    # Custom evaluation
    from scripts.evaluation_framework import CustomEvaluator, EvalSample, MetricType
    evaluator = CustomEvaluator(model_fn=my_model)
    samples = [EvalSample(input="What is 2+2?", expected="4")]
    evaluator.evaluate_dataset(samples, MetricType.CONTAINS)

    # Experiment tracking (requires mlflow)
    from scripts.mlflow_utils import setup_mlflow, log_training_run
    setup_mlflow(experiment_name="my-experiment")

    # Data versioning
    from scripts.versioning_utils import compute_data_hash, create_data_manifest
    manifest = create_data_manifest("data/")

    # Reproducibility
    from scripts.reproducibility import set_seed, capture_environment
    set_seed(42)
"""

__version__ = "0.1.0"

# Always available - no external dependencies
from .benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
    compare_models,
    format_benchmark_results,
    get_benchmark_suite,
    BENCHMARK_SUITES,
)

from .evaluation_framework import (
    MetricType,
    EvalSample,
    EvalResult,
    EvaluationMetrics,
    CustomEvaluator,
    LLMJudge,
    PairwiseJudge,
)

from .reproducibility import (
    set_seed,
    get_seed_state,
    set_seed_state,
    capture_environment,
    EnvironmentSnapshot,
    verify_reproducibility,
    audit_reproducibility,
    AuditResult,
    generate_requirements,
    worker_init_fn,
)

# Versioning utils - mostly available, some features need mlflow
from .versioning_utils import (
    compute_data_hash,
    compute_directory_hash,
    create_data_manifest,
    DataManifest,
    compare_data_versions,
)

# Optional: MLflow-dependent features
_mlflow_exports = []
_versioning_mlflow_exports = []

try:
    from .mlflow_utils import (
        setup_mlflow,
        create_experiment,
        log_training_run,
        log_metrics_over_time,
        get_best_run,
        compare_runs,
        load_model_from_run,
        get_run_artifacts,
        ExperimentTracker,
    )
    _mlflow_exports = [
        "setup_mlflow",
        "create_experiment",
        "log_training_run",
        "log_metrics_over_time",
        "get_best_run",
        "compare_runs",
        "load_model_from_run",
        "get_run_artifacts",
        "ExperimentTracker",
    ]
except ImportError:
    # MLflow not installed - these functions won't be available
    pass

try:
    from .versioning_utils import (
        register_model_version,
        transition_model_stage,
        get_model_versions,
        load_production_model,
    )
    _versioning_mlflow_exports = [
        "register_model_version",
        "transition_model_stage",
        "get_model_versions",
        "load_production_model",
    ]
except ImportError:
    # MLflow not installed - these functions won't be available
    pass

__all__ = [
    # Version
    "__version__",
    # Benchmark utilities (always available)
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
    "compare_models",
    "format_benchmark_results",
    "get_benchmark_suite",
    "BENCHMARK_SUITES",
    # Evaluation framework (always available)
    "MetricType",
    "EvalSample",
    "EvalResult",
    "EvaluationMetrics",
    "CustomEvaluator",
    "LLMJudge",
    "PairwiseJudge",
    # Reproducibility utilities (always available)
    "set_seed",
    "get_seed_state",
    "set_seed_state",
    "capture_environment",
    "EnvironmentSnapshot",
    "verify_reproducibility",
    "audit_reproducibility",
    "AuditResult",
    "generate_requirements",
    "worker_init_fn",
    # Versioning utilities (always available)
    "compute_data_hash",
    "compute_directory_hash",
    "create_data_manifest",
    "DataManifest",
    "compare_data_versions",
] + _mlflow_exports + _versioning_mlflow_exports
