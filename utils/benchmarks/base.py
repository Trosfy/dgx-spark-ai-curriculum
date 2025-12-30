"""
Base Benchmark Classes and Utilities

Provides common infrastructure for all benchmark types in the curriculum.
"""

import time
import subprocess
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class BenchmarkResult:
    """
    Universal benchmark result container.

    This base class captures common metrics across all benchmark types.
    Specialized benchmarks can extend this with additional fields.

    Attributes:
        model: Model name or identifier
        backend: Backend used (ollama, pytorch, vllm, etc.)
        quantization: Quantization type (fp32, fp16, bf16, q4, q8, etc.)
        prompt_tokens: Number of input tokens
        generated_tokens: Number of output tokens
        prefill_time_s: Time to process input (seconds)
        decode_time_s: Time to generate output (seconds)
        total_time_s: Total execution time
        prefill_tps: Input processing speed (tokens/second)
        decode_tps: Output generation speed (tokens/second)
        memory_gb: Memory used (GB)
        timestamp: When the benchmark was run
        error: Error message if benchmark failed
        metadata: Additional benchmark-specific data
    """
    model: str
    backend: str = "unknown"
    quantization: str = "unknown"
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0
    total_time_s: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    memory_gb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.error:
            return f"{self.model}: ERROR - {self.error}"
        return (
            f"{self.model} ({self.backend}/{self.quantization}): "
            f"Prefill: {self.prefill_tps:.1f} tok/s, "
            f"Decode: {self.decode_tps:.1f} tok/s, "
            f"Memory: {self.memory_gb:.1f} GB"
        )


@dataclass
class BenchmarkSummary:
    """
    Summary of multiple benchmark runs.

    Aggregates results from multiple runs to provide statistical measures.
    """
    model: str
    backend: str
    quantization: str
    runs: int
    avg_prefill_tps: float
    std_prefill_tps: float
    avg_decode_tps: float
    std_decode_tps: float
    avg_memory_gb: float
    avg_total_time_s: float = 0.0
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: List[BenchmarkResult]) -> "BenchmarkSummary":
        """Create summary from a list of results."""
        if not results:
            raise ValueError("Cannot create summary from empty results")

        prefill_values = [r.prefill_tps for r in results if r.error is None]
        decode_values = [r.decode_tps for r in results if r.error is None]
        memory_values = [r.memory_gb for r in results if r.error is None]
        time_values = [r.total_time_s for r in results if r.error is None]

        return cls(
            model=results[0].model,
            backend=results[0].backend,
            quantization=results[0].quantization,
            runs=len(results),
            avg_prefill_tps=statistics.mean(prefill_values) if prefill_values else 0,
            std_prefill_tps=statistics.stdev(prefill_values) if len(prefill_values) > 1 else 0,
            avg_decode_tps=statistics.mean(decode_values) if decode_values else 0,
            std_decode_tps=statistics.stdev(decode_values) if len(decode_values) > 1 else 0,
            avg_memory_gb=statistics.mean(memory_values) if memory_values else 0,
            avg_total_time_s=statistics.mean(time_values) if time_values else 0,
            results=results
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "backend": self.backend,
            "quantization": self.quantization,
            "runs": self.runs,
            "avg_prefill_tps": self.avg_prefill_tps,
            "std_prefill_tps": self.std_prefill_tps,
            "avg_decode_tps": self.avg_decode_tps,
            "std_decode_tps": self.std_decode_tps,
            "avg_memory_gb": self.avg_memory_gb,
            "avg_total_time_s": self.avg_total_time_s,
            "individual_runs": [r.to_dict() for r in self.results],
        }


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmark implementations.

    Provides common interface and utilities for benchmarking.
    """

    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: List[BenchmarkResult] = []

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the benchmark backend is available."""
        pass

    @abstractmethod
    def benchmark_single(self, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        pass

    def warmup(self, *args, **kwargs) -> None:
        """Optional warmup before benchmarking."""
        pass

    def benchmark(
        self,
        *args,
        runs: int = 5,
        warmup_runs: int = 1,
        **kwargs
    ) -> BenchmarkSummary:
        """
        Run multiple benchmark iterations and return summary.

        Args:
            runs: Number of benchmark iterations
            warmup_runs: Number of warmup iterations
            **kwargs: Arguments passed to benchmark_single

        Returns:
            BenchmarkSummary with aggregated statistics
        """
        # Warmup
        for _ in range(warmup_runs):
            self.warmup(*args, **kwargs)

        # Benchmark runs
        results = []
        for i in range(runs):
            try:
                result = self.benchmark_single(*args, **kwargs)
                results.append(result)
                self.results.append(result)
                print(f"  Run {i+1}/{runs}: {result}")
            except Exception as e:
                error_result = BenchmarkResult(
                    model=kwargs.get("model", "unknown"),
                    error=str(e)
                )
                results.append(error_result)

        return BenchmarkSummary.from_results(results)


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB."""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return float(result.stdout.strip()) / 1024
    except Exception:
        return 0.0


def format_results_table(
    results: List[BenchmarkSummary],
    title: str = "Benchmark Results"
) -> str:
    """
    Format benchmark results as a markdown table.

    Args:
        results: List of benchmark summaries
        title: Table title

    Returns:
        Formatted markdown table string
    """
    if not results:
        return "No results to display"

    lines = [
        f"\n{'=' * 90}",
        title,
        "=" * 90,
        f"{'Model':<30} {'Backend':<12} {'Quant':<8} {'Prefill':<15} {'Decode':<15} {'Memory':<10}",
        f"{'':30} {'':12} {'':8} {'(tok/s)':<15} {'(tok/s)':<15} {'(GB)':<10}",
        "-" * 90,
    ]

    for r in results:
        prefill_str = f"{r.avg_prefill_tps:.1f} ± {r.std_prefill_tps:.1f}"
        decode_str = f"{r.avg_decode_tps:.1f} ± {r.std_decode_tps:.1f}"
        lines.append(
            f"{r.model:<30} {r.backend:<12} {r.quantization:<8} "
            f"{prefill_str:<15} {decode_str:<15} {r.avg_memory_gb:<10.1f}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)
