"""
LLM Evaluation Harness Utilities
================================

Utilities for running LLM benchmarks using the LM Evaluation Harness
and analyzing results for standardized model evaluation.

Example:
    >>> from utils.benchmarks import run_benchmark, compare_models_eval
    >>>
    >>> results = run_benchmark(
    ...     model_name="microsoft/phi-2",
    ...     tasks=["hellaswag", "arc_easy"],
    ...     output_dir="./results"
    ... )
    >>> print(f"Average score: {results.average_score:.2%}")
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import glob as glob_module


@dataclass
class EvalBenchmarkConfig:
    """Configuration for an evaluation benchmark run."""

    model_name: str
    tasks: List[str]
    output_dir: str
    batch_size: int = 8
    num_fewshot: Optional[int] = None
    limit: Optional[int] = None
    dtype: str = "bfloat16"
    device: str = "cuda"

    def to_command_args(self) -> List[str]:
        """Convert config to lm_eval command arguments."""
        args = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_name},dtype={self.dtype}",
            "--tasks", ",".join(self.tasks),
            "--batch_size", str(self.batch_size),
            "--output_path", self.output_dir,
        ]

        if self.num_fewshot is not None:
            args.extend(["--num_fewshot", str(self.num_fewshot)])

        if self.limit is not None:
            args.extend(["--limit", str(self.limit)])

        return args


@dataclass
class EvalBenchmarkResult:
    """Result of an evaluation benchmark run."""

    model_name: str
    tasks: Dict[str, Dict[str, float]]
    runtime_seconds: float
    config: EvalBenchmarkConfig
    raw_results: Dict = field(default_factory=dict)

    @property
    def average_score(self) -> float:
        """Calculate average score across all tasks."""
        scores = []
        for task_results in self.tasks.values():
            score = task_results.get("acc_norm", task_results.get("acc", 0))
            if isinstance(score, (int, float)):
                scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "tasks": self.tasks,
            "runtime_seconds": self.runtime_seconds,
            "average_score": self.average_score,
        }


def run_benchmark(
    model_name: str,
    tasks: List[str],
    output_dir: str,
    batch_size: int = 8,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    dtype: str = "bfloat16",
    verbose: bool = True,
) -> Optional[EvalBenchmarkResult]:
    """
    Run LM Evaluation Harness benchmark on a model.

    Args:
        model_name: HuggingFace model path (e.g., "microsoft/phi-2")
        tasks: List of benchmark tasks (e.g., ["hellaswag", "arc_easy"])
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples (None for task default)
        limit: Limit number of samples (None for full evaluation)
        dtype: Data type for model (bfloat16 recommended for DGX Spark)
        verbose: Print progress information

    Returns:
        EvalBenchmarkResult object with results, or None if failed

    Example:
        >>> result = run_benchmark(
        ...     model_name="microsoft/phi-2",
        ...     tasks=["hellaswag", "arc_easy"],
        ...     output_dir="./results/phi2",
        ...     limit=100  # Quick test
        ... )
        >>> print(f"Average score: {result.average_score:.2%}")
    """
    config = EvalBenchmarkConfig(
        model_name=model_name,
        tasks=tasks,
        output_dir=output_dir,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        dtype=dtype,
    )

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running benchmark for {model_name}")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"Limit: {limit if limit else 'Full evaluation'}")
        print(f"{'='*60}")

    cmd = config.to_command_args()
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 4  # 4 hour timeout
        )

        if result.returncode != 0:
            print(f"Error running benchmark: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print("Benchmark timed out")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

    runtime = time.time() - start_time

    # Load results
    result_files = glob_module.glob(f"{output_dir}/*/results.json")
    if not result_files:
        print("No results file found")
        return None

    with open(result_files[0], 'r') as f:
        raw_results = json.load(f)

    # Parse task results
    task_results = {}
    for task_name, metrics in raw_results.get("results", {}).items():
        task_results[task_name] = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

    benchmark_result = EvalBenchmarkResult(
        model_name=model_name,
        tasks=task_results,
        runtime_seconds=runtime,
        config=config,
        raw_results=raw_results,
    )

    if verbose:
        print(f"\nCompleted in {runtime/60:.1f} minutes")
        print(f"Average score: {benchmark_result.average_score:.2%}")

    return benchmark_result


def compare_models_eval(
    models: List[str],
    tasks: List[str],
    output_dir: str,
    **benchmark_kwargs,
) -> Dict[str, EvalBenchmarkResult]:
    """
    Compare multiple models on the same benchmark tasks.

    Args:
        models: List of model names to compare
        tasks: Benchmark tasks to run
        output_dir: Base output directory
        **benchmark_kwargs: Additional arguments for run_benchmark

    Returns:
        Dictionary mapping model names to their results

    Example:
        >>> comparison = compare_models_eval(
        ...     models=["microsoft/phi-2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        ...     tasks=["hellaswag"],
        ...     output_dir="./comparison",
        ...     limit=50
        ... )
        >>> for model, result in comparison.items():
        ...     print(f"{model}: {result.average_score:.2%}")
    """
    results = {}

    for model_name in models:
        safe_name = model_name.replace("/", "_").replace("-", "_")
        model_output_dir = os.path.join(output_dir, safe_name)

        result = run_benchmark(
            model_name=model_name,
            tasks=tasks,
            output_dir=model_output_dir,
            **benchmark_kwargs,
        )

        if result:
            results[model_name] = result

    return results


def format_eval_results(
    results: Dict[str, EvalBenchmarkResult],
    include_per_task: bool = True,
) -> str:
    """
    Format benchmark results as a readable table.

    Args:
        results: Dictionary of model results
        include_per_task: Include per-task breakdown

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK COMPARISON")
    lines.append("=" * 70)

    all_tasks = set()
    for result in results.values():
        all_tasks.update(result.tasks.keys())
    all_tasks = sorted(all_tasks)

    if include_per_task and all_tasks:
        header = f"{'Model':<30} " + " ".join(f"{t[:12]:<12}" for t in all_tasks) + f" {'Average':<10}"
        lines.append(header)
        lines.append("-" * len(header))
    else:
        lines.append(f"{'Model':<40} {'Average Score':<15} {'Runtime':<15}")
        lines.append("-" * 70)

    for model_name, result in results.items():
        short_name = model_name.split("/")[-1][:28]

        if include_per_task and all_tasks:
            task_scores = []
            for task in all_tasks:
                task_result = result.tasks.get(task, {})
                score = task_result.get("acc_norm", task_result.get("acc", 0))
                task_scores.append(f"{score*100:>10.1f}%")

            line = f"{short_name:<30} " + " ".join(task_scores) + f" {result.average_score*100:>8.1f}%"
        else:
            line = f"{short_name:<40} {result.average_score*100:>13.1f}% {result.runtime_seconds/60:>12.1f}m"

        lines.append(line)

    lines.append("=" * 70)

    return "\n".join(lines)


# Standard benchmark suites
BENCHMARK_SUITES = {
    "quick": {
        "description": "Quick sanity check (1-2 minutes)",
        "tasks": ["hellaswag"],
        "limit": 50,
        "num_fewshot": 0,
    },
    "standard": {
        "description": "Standard evaluation (10-20 minutes)",
        "tasks": ["hellaswag", "arc_easy", "winogrande", "truthfulqa_mc2"],
        "limit": 200,
        "num_fewshot": 0,
    },
    "full": {
        "description": "Full evaluation (1-2 hours)",
        "tasks": ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "truthfulqa_mc2"],
        "limit": None,
        "num_fewshot": None,
    },
    "open_llm_leaderboard": {
        "description": "Open LLM Leaderboard suite",
        "tasks": ["arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc2", "winogrande", "gsm8k"],
        "limit": None,
        "num_fewshot": None,
    },
}


def get_benchmark_suite(suite_name: str) -> Dict:
    """
    Get a predefined benchmark suite configuration.

    Args:
        suite_name: Name of the suite ("quick", "standard", "full", etc.)

    Returns:
        Dictionary with suite configuration

    Example:
        >>> suite = get_benchmark_suite("quick")
        >>> result = run_benchmark(
        ...     model_name="microsoft/phi-2",
        ...     tasks=suite["tasks"],
        ...     output_dir="./results",
        ...     limit=suite["limit"]
        ... )
    """
    if suite_name not in BENCHMARK_SUITES:
        available = ", ".join(BENCHMARK_SUITES.keys())
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")

    return BENCHMARK_SUITES[suite_name].copy()


if __name__ == "__main__":
    print("LM Evaluation Harness Utilities")
    print("=" * 50)
    print("\nAvailable benchmark suites:")
    for name, config in BENCHMARK_SUITES.items():
        print(f"  {name}: {config['description']}")

    print("\nExample usage:")
    print("  from utils.benchmarks import run_benchmark, get_benchmark_suite")
    print("  suite = get_benchmark_suite('quick')")
    print("  result = run_benchmark('microsoft/phi-2', suite['tasks'], './results', limit=suite['limit'])")
