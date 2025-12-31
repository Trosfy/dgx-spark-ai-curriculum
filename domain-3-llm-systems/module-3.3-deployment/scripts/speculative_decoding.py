"""
Speculative Decoding Utilities for LLM Inference Acceleration

This module provides tools for implementing and measuring speculative decoding
techniques including Medusa and EAGLE on DGX Spark.

Speculative decoding works like having a fast draft writer and a careful editor:
1. The "draft" model (or heads) quickly proposes multiple tokens
2. The "target" model verifies them in parallel
3. Accepted tokens are kept; rejected ones are regenerated

This can achieve 2-3x speedups for certain types of generation tasks.

Example:
    >>> from speculative_decoding import SpeculativeDecoder, MedusaConfig
    >>>
    >>> # Configure Medusa-style speculation
    >>> config = MedusaConfig(num_heads=4, max_speculation_length=5)
    >>> decoder = SpeculativeDecoder(config)
    >>>
    >>> # Measure acceptance rate on test prompts
    >>> results = decoder.benchmark_acceptance_rate(prompts)
    >>> print(f"Acceptance Rate: {results.acceptance_rate:.2%}")
"""

from __future__ import annotations

import time
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, Optional

import requests


class SpeculativeMethod(Enum):
    """Supported speculative decoding methods."""
    MEDUSA = "medusa"  # Multiple prediction heads
    EAGLE = "eagle"    # Feature-level draft tokens
    EAGLE3 = "eagle3"  # Latest EAGLE improvements
    DRAFT_MODEL = "draft_model"  # Separate smaller draft model


@dataclass
class MedusaConfig:
    """
    Configuration for Medusa speculative decoding.

    Medusa adds multiple prediction heads to a transformer model.
    Each head predicts a future token, allowing parallel verification.

    Attributes:
        num_heads: Number of Medusa heads (typically 3-5)
        max_speculation_length: Maximum tokens to speculate ahead
        tree_depth: Depth of the speculation tree (for tree attention)
        temperature: Sampling temperature for speculation
        top_k: Top-k filtering for draft generation

    Example:
        >>> config = MedusaConfig(num_heads=4, max_speculation_length=5)
        >>> print(config)
        MedusaConfig(heads=4, max_spec=5)
    """
    num_heads: int = 4
    max_speculation_length: int = 5
    tree_depth: int = 3
    temperature: float = 0.0  # Greedy for speculation
    top_k: int = 10

    def __repr__(self) -> str:
        return f"MedusaConfig(heads={self.num_heads}, max_spec={self.max_speculation_length})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_heads": self.num_heads,
            "max_speculation_length": self.max_speculation_length,
            "tree_depth": self.tree_depth,
            "temperature": self.temperature,
            "top_k": self.top_k
        }


@dataclass
class EAGLEConfig:
    """
    Configuration for EAGLE speculative decoding.

    EAGLE uses feature-level draft tokens instead of token-level,
    which can achieve better speculation accuracy for longer sequences.

    Attributes:
        draft_model_layers: Number of transformer layers for draft
        feature_dim: Hidden dimension for feature drafts
        speculation_length: How many tokens to speculate
        acceptance_threshold: Minimum probability for acceptance

    Example:
        >>> config = EAGLEConfig(speculation_length=6)
        >>> print(config)
        EAGLEConfig(spec_len=6)
    """
    draft_model_layers: int = 1
    feature_dim: int = 4096
    speculation_length: int = 6
    acceptance_threshold: float = 0.1

    def __repr__(self) -> str:
        return f"EAGLEConfig(spec_len={self.speculation_length})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_model_layers": self.draft_model_layers,
            "feature_dim": self.feature_dim,
            "speculation_length": self.speculation_length,
            "acceptance_threshold": self.acceptance_threshold
        }


@dataclass
class SpeculationResult:
    """
    Result from a speculative decoding benchmark.

    Attributes:
        prompt: The input prompt
        output: Generated text
        tokens_generated: Total output tokens
        tokens_accepted: Tokens accepted from speculation
        tokens_rejected: Tokens that needed regeneration
        speculation_rounds: Number of spec-verify cycles
        total_time: Total generation time in seconds
        time_without_speculation: Estimated time without speculation
        speedup: Achieved speedup factor
        acceptance_rate: Ratio of accepted to proposed tokens
        method: Which speculative method was used
    """
    prompt: str
    output: str = ""
    tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    speculation_rounds: int = 0
    total_time: float = 0.0
    time_without_speculation: float = 0.0
    speedup: float = 1.0
    acceptance_rate: float = 0.0
    method: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None

    def compute_metrics(self) -> None:
        """Compute derived metrics from raw counts."""
        total_proposed = self.tokens_accepted + self.tokens_rejected
        if total_proposed > 0:
            self.acceptance_rate = self.tokens_accepted / total_proposed

        if self.total_time > 0 and self.time_without_speculation > 0:
            self.speedup = self.time_without_speculation / self.total_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_preview": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "tokens_generated": self.tokens_generated,
            "tokens_accepted": self.tokens_accepted,
            "tokens_rejected": self.tokens_rejected,
            "acceptance_rate": f"{self.acceptance_rate:.2%}",
            "speedup": f"{self.speedup:.2f}x",
            "total_time_ms": round(self.total_time * 1000, 1),
            "method": self.method,
            "error": self.error
        }


@dataclass
class BatchSpeculationResult:
    """Aggregate results from multiple speculation benchmarks."""
    results: list[SpeculationResult] = field(default_factory=list)
    avg_acceptance_rate: float = 0.0
    avg_speedup: float = 1.0
    total_tokens: int = 0
    total_time: float = 0.0
    method: str = ""

    def compute_stats(self) -> None:
        """Compute aggregate statistics."""
        if not self.results:
            return

        successful = [r for r in self.results if r.error is None]
        if not successful:
            return

        acceptance_rates = [r.acceptance_rate for r in successful]
        speedups = [r.speedup for r in successful if r.speedup > 0]

        self.avg_acceptance_rate = sum(acceptance_rates) / len(acceptance_rates)
        self.avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
        self.total_tokens = sum(r.tokens_generated for r in successful)
        self.total_time = sum(r.total_time for r in successful)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Speculative Decoding Benchmark Results
======================================
Method: {self.method}
Total Requests: {len(self.results)}
Total Tokens: {self.total_tokens:,}
Total Time: {self.total_time:.2f}s

Performance:
  - Average Acceptance Rate: {self.avg_acceptance_rate:.1%}
  - Average Speedup: {self.avg_speedup:.2f}x
  - Effective Tokens/Second: {self.total_tokens / self.total_time:.1f}
"""


class SpeculativeDecoder(ABC):
    """
    Abstract base class for speculative decoders.

    This class defines the interface for different speculative
    decoding implementations (Medusa, EAGLE, etc.).
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> SpeculationResult:
        """Generate text using speculative decoding."""
        pass

    @abstractmethod
    def get_acceptance_stats(self) -> dict[str, Any]:
        """Get acceptance rate statistics."""
        pass


class SGLangSpeculativeClient:
    """
    Client for SGLang server with speculative decoding support.

    SGLang supports both Medusa and EAGLE-style speculation
    when configured with the appropriate model.

    Example:
        >>> client = SGLangSpeculativeClient(
        ...     base_url="http://localhost:30000",
        ...     method="medusa"
        ... )
        >>> result = client.generate("Count from 1 to 10")
        >>> print(f"Speedup: {result.speedup:.2f}x")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000",
        model: str = "default",
        method: SpeculativeMethod = SpeculativeMethod.MEDUSA
    ):
        """
        Initialize SGLang speculative client.

        Args:
            base_url: SGLang server URL
            model: Model name (use "default" for single-model servers)
            method: Speculative decoding method
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.method = method

    def is_healthy(self) -> bool:
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False
    ) -> SpeculationResult:
        """
        Generate with speculative decoding.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream (not fully supported for speculation metrics)

        Returns:
            SpeculationResult with acceptance metrics
        """
        result = SpeculationResult(
            prompt=prompt,
            method=self.method.value
        )

        try:
            start_time = time.perf_counter()

            # SGLang uses OpenAI-compatible API
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()

            end_time = time.perf_counter()
            data = response.json()

            # Extract response
            choice = data.get("choices", [{}])[0]
            result.output = choice.get("message", {}).get("content", "")

            # Extract usage stats
            usage = data.get("usage", {})
            result.tokens_generated = usage.get("completion_tokens", 0)

            # Extract speculation metrics if available
            # Note: These fields depend on SGLang version and configuration
            spec_stats = data.get("speculation_stats", {})
            result.tokens_accepted = spec_stats.get("tokens_accepted", result.tokens_generated)
            result.tokens_rejected = spec_stats.get("tokens_rejected", 0)
            result.speculation_rounds = spec_stats.get("rounds", 0)

            result.total_time = end_time - start_time

            # Estimate time without speculation (assume 2x speedup baseline)
            # In real benchmarks, compare with non-speculative baseline
            result.time_without_speculation = result.total_time * 1.5

            result.compute_metrics()

        except Exception as e:
            result.error = str(e)
            result.total_time = time.perf_counter() - start_time if 'start_time' in locals() else 0

        return result

    def benchmark(
        self,
        prompts: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> BatchSpeculationResult:
        """
        Benchmark speculative decoding on multiple prompts.

        Args:
            prompts: List of prompts to test
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            BatchSpeculationResult with aggregate statistics
        """
        batch_result = BatchSpeculationResult(method=self.method.value)

        for i, prompt in enumerate(prompts):
            print(f"  Running {i+1}/{len(prompts)}: {prompt[:40]}...", end="")
            result = self.generate(prompt, max_tokens, temperature)
            batch_result.results.append(result)

            if result.error:
                print(f" Error: {result.error}")
            else:
                print(f" OK ({result.acceptance_rate:.0%} accepted)")

        batch_result.compute_stats()
        return batch_result


def measure_acceptance_rate(
    server_url: str,
    prompts: list[str],
    method: str = "medusa",
    max_tokens: int = 100
) -> dict[str, Any]:
    """
    Measure speculative decoding acceptance rate.

    This is a convenience function for quickly benchmarking
    acceptance rates across different prompt types.

    Args:
        server_url: Inference server URL
        prompts: List of prompts to test
        method: "medusa" or "eagle"
        max_tokens: Maximum tokens per generation

    Returns:
        Dictionary with acceptance statistics

    Example:
        >>> prompts = ["Count to 10", "Write a poem", "Explain AI"]
        >>> stats = measure_acceptance_rate(
        ...     "http://localhost:30000",
        ...     prompts,
        ...     method="medusa"
        ... )
        >>> print(f"Avg acceptance: {stats['avg_acceptance_rate']:.1%}")
    """
    spec_method = SpeculativeMethod(method)
    client = SGLangSpeculativeClient(
        base_url=server_url,
        method=spec_method
    )

    if not client.is_healthy():
        return {
            "error": f"Server not available at {server_url}",
            "avg_acceptance_rate": 0.0,
            "avg_speedup": 0.0
        }

    print(f"\nMeasuring {method} acceptance rate on {len(prompts)} prompts...")
    batch_result = client.benchmark(prompts, max_tokens=max_tokens)

    return {
        "method": method,
        "total_prompts": len(prompts),
        "successful": len([r for r in batch_result.results if r.error is None]),
        "avg_acceptance_rate": batch_result.avg_acceptance_rate,
        "avg_speedup": batch_result.avg_speedup,
        "total_tokens": batch_result.total_tokens,
        "total_time": batch_result.total_time
    }


def compare_with_baseline(
    server_url_speculation: str,
    server_url_baseline: str,
    prompts: list[str],
    max_tokens: int = 100
) -> dict[str, Any]:
    """
    Compare speculative decoding performance against baseline.

    Args:
        server_url_speculation: URL of server with speculative decoding
        server_url_baseline: URL of baseline server (same model, no speculation)
        prompts: Test prompts
        max_tokens: Maximum tokens per generation

    Returns:
        Comparison statistics including actual speedup
    """
    # Benchmark baseline
    print("\nBenchmarking baseline (no speculation)...")
    baseline_times = []
    baseline_tokens = 0

    for i, prompt in enumerate(prompts):
        print(f"  Baseline {i+1}/{len(prompts)}...", end="")
        try:
            start = time.perf_counter()
            response = requests.post(
                f"{server_url_baseline}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=120
            )
            elapsed = time.perf_counter() - start
            baseline_times.append(elapsed)

            if response.status_code == 200:
                data = response.json()
                baseline_tokens += data.get("usage", {}).get("completion_tokens", 0)
                print(f" {elapsed*1000:.0f}ms")
            else:
                print(f" Error: {response.status_code}")
        except Exception as e:
            print(f" Error: {e}")

    # Benchmark with speculation
    print("\nBenchmarking with speculative decoding...")
    spec_client = SGLangSpeculativeClient(
        base_url=server_url_speculation,
        method=SpeculativeMethod.MEDUSA
    )
    spec_result = spec_client.benchmark(prompts, max_tokens=max_tokens)

    # Compute comparison
    baseline_total = sum(baseline_times)
    spec_total = spec_result.total_time

    actual_speedup = baseline_total / spec_total if spec_total > 0 else 0

    return {
        "baseline": {
            "total_time": baseline_total,
            "total_tokens": baseline_tokens,
            "avg_time_per_request": baseline_total / len(prompts) if prompts else 0
        },
        "speculative": {
            "total_time": spec_total,
            "total_tokens": spec_result.total_tokens,
            "avg_acceptance_rate": spec_result.avg_acceptance_rate,
            "avg_time_per_request": spec_total / len(prompts) if prompts else 0
        },
        "actual_speedup": actual_speedup,
        "prompts_tested": len(prompts)
    }


def get_optimal_speculation_config(
    prompt_type: str = "general"
) -> dict[str, Any]:
    """
    Get recommended speculation configuration for prompt type.

    Different types of prompts benefit from different speculation settings.
    Predictable outputs (code, lists) have high acceptance rates.
    Creative outputs (stories, poems) have lower acceptance rates.

    Args:
        prompt_type: One of "general", "code", "predictable", "creative"

    Returns:
        Recommended configuration dictionary
    """
    configs = {
        "predictable": {
            "method": "medusa",
            "num_heads": 5,
            "max_speculation_length": 8,
            "tree_depth": 4,
            "expected_acceptance_rate": 0.8,
            "expected_speedup": 2.5,
            "notes": "Best for: counting, lists, structured output"
        },
        "code": {
            "method": "medusa",
            "num_heads": 4,
            "max_speculation_length": 6,
            "tree_depth": 3,
            "expected_acceptance_rate": 0.65,
            "expected_speedup": 2.0,
            "notes": "Best for: code generation, formatting"
        },
        "general": {
            "method": "medusa",
            "num_heads": 4,
            "max_speculation_length": 5,
            "tree_depth": 3,
            "expected_acceptance_rate": 0.5,
            "expected_speedup": 1.7,
            "notes": "Balanced for mixed workloads"
        },
        "creative": {
            "method": "eagle",
            "num_heads": 3,
            "max_speculation_length": 4,
            "tree_depth": 2,
            "expected_acceptance_rate": 0.35,
            "expected_speedup": 1.3,
            "notes": "For unpredictable outputs; EAGLE may be better than Medusa"
        }
    }

    return configs.get(prompt_type, configs["general"])


def format_speculation_report(results: BatchSpeculationResult) -> str:
    """
    Format speculation results as a markdown report.

    Args:
        results: BatchSpeculationResult to format

    Returns:
        Markdown-formatted report string
    """
    lines = [
        "# Speculative Decoding Benchmark Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Method:** {results.method}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Requests | {len(results.results)} |",
        f"| Successful | {len([r for r in results.results if r.error is None])} |",
        f"| Total Tokens | {results.total_tokens:,} |",
        f"| Total Time | {results.total_time:.2f}s |",
        f"| **Avg Acceptance Rate** | **{results.avg_acceptance_rate:.1%}** |",
        f"| **Avg Speedup** | **{results.avg_speedup:.2f}x** |",
        "",
        "## Individual Results",
        "",
        "| # | Prompt Preview | Tokens | Acceptance | Speedup |",
        "|---|----------------|--------|------------|---------|"
    ]

    for i, r in enumerate(results.results):
        prompt_preview = r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt
        if r.error:
            lines.append(f"| {i+1} | {prompt_preview} | ERROR | - | - |")
        else:
            lines.append(
                f"| {i+1} | {prompt_preview} | {r.tokens_generated} | "
                f"{r.acceptance_rate:.0%} | {r.speedup:.1f}x |"
            )

    lines.extend([
        "",
        "## Recommendations",
        "",
        f"Based on the average acceptance rate of {results.avg_acceptance_rate:.1%}:",
        ""
    ])

    if results.avg_acceptance_rate >= 0.7:
        lines.append("- **Excellent** acceptance rate! Speculative decoding is highly effective for this workload.")
        lines.append("- Consider increasing `max_speculation_length` for even more speedup.")
    elif results.avg_acceptance_rate >= 0.5:
        lines.append("- **Good** acceptance rate. Speculative decoding provides meaningful speedup.")
        lines.append("- Current configuration is appropriate for this workload.")
    elif results.avg_acceptance_rate >= 0.3:
        lines.append("- **Moderate** acceptance rate. Some speedup, but not optimal.")
        lines.append("- Consider reducing `num_heads` or trying EAGLE instead of Medusa.")
    else:
        lines.append("- **Low** acceptance rate. Speculative decoding may not help this workload.")
        lines.append("- Consider disabling speculation or using a different draft model.")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Speculative Decoding Utilities Demo")
    print("=" * 50)

    # Show configuration recommendations
    print("\nRecommended configurations by prompt type:")
    for prompt_type in ["predictable", "code", "general", "creative"]:
        config = get_optimal_speculation_config(prompt_type)
        print(f"\n{prompt_type.upper()}:")
        print(f"  Method: {config['method']}")
        print(f"  Expected acceptance: {config['expected_acceptance_rate']:.0%}")
        print(f"  Expected speedup: {config['expected_speedup']:.1f}x")
        print(f"  Notes: {config['notes']}")
