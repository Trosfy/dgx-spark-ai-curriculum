"""
Benchmarking Utilities for Architecture Comparison

This module provides tools for comparing different neural network architectures
(Transformers, Mamba, MoE) on metrics like speed, memory, and quality.

Key Features:
- Standardized benchmarking protocol
- Memory tracking with DGX Spark awareness
- Perplexity evaluation for quality comparison
- Visualization utilities for results

Example Usage:
    from scripts.benchmark_utils import ArchitectureBenchmark, compare_architectures

    benchmark = ArchitectureBenchmark()
    results = benchmark.run_full_benchmark(
        models={"mamba": mamba_model, "transformer": transformer_model},
        tokenizers={"mamba": mamba_tok, "transformer": transformer_tok},
    )
    benchmark.plot_results(results)

DGX Spark Notes:
- 128GB allows benchmarking large models without offloading
- Unified memory means accurate memory measurements
- Always clear cache between benchmarks for fair comparison
"""

import gc
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run result.

    Attributes:
        model_name: Identifier for the model
        architecture_type: "transformer", "mamba", or "moe"
        context_length: Input sequence length tested
        generation_length: Tokens generated
        total_time_seconds: Total inference time
        time_to_first_token: Latency to first output token
        tokens_per_second: Generation throughput
        memory_peak_gb: Peak GPU memory usage
        memory_model_gb: Memory used by model weights
        perplexity: Optional perplexity score
        metadata: Additional benchmark metadata
    """
    model_name: str
    architecture_type: str
    context_length: int
    generation_length: int
    total_time_seconds: float
    time_to_first_token: float
    tokens_per_second: float
    memory_peak_gb: float
    memory_model_gb: float
    perplexity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchitectureProfile:
    """Performance profile for an architecture type.

    Used to understand the theoretical properties of each architecture.
    """
    name: str
    attention_complexity: str  # "O(n^2)", "O(n)", "O(n * experts)"
    memory_scaling: str  # "linear", "quadratic", "constant"
    kv_cache_required: bool
    sparse_activation: bool
    typical_context_limit: int
    dgx_spark_context_limit: int  # With 128GB


# Architecture profiles
ARCHITECTURE_PROFILES = {
    "transformer": ArchitectureProfile(
        name="Transformer (Dense Attention)",
        attention_complexity="O(n^2)",
        memory_scaling="quadratic (KV cache)",
        kv_cache_required=True,
        sparse_activation=False,
        typical_context_limit=8192,
        dgx_spark_context_limit=65536,  # Larger with 128GB
    ),
    "mamba": ArchitectureProfile(
        name="Mamba (State Space)",
        attention_complexity="O(n)",
        memory_scaling="constant",
        kv_cache_required=False,
        sparse_activation=False,
        typical_context_limit=65536,
        dgx_spark_context_limit=262144,  # Much larger!
    ),
    "moe": ArchitectureProfile(
        name="Mixture of Experts",
        attention_complexity="O(n^2) but sparse",
        memory_scaling="linear (all experts loaded)",
        kv_cache_required=True,
        sparse_activation=True,
        typical_context_limit=8192,
        dgx_spark_context_limit=32768,
    ),
    "jamba": ArchitectureProfile(
        name="Jamba (Hybrid Mamba-Attention)",
        attention_complexity="O(n) + O(n^2) selective",
        memory_scaling="mostly constant",
        kv_cache_required=True,  # For attention layers
        sparse_activation=True,  # Has MoE components
        typical_context_limit=65536,
        dgx_spark_context_limit=262144,
    ),
}


class ArchitectureBenchmark:
    """
    Comprehensive benchmarking suite for architecture comparison.

    This class provides methods to fairly compare different architectures
    on the same hardware (DGX Spark) with consistent methodology.

    Example:
        >>> benchmark = ArchitectureBenchmark()
        >>> benchmark.add_model("mamba", model, tokenizer, "mamba")
        >>> benchmark.add_model("llama", model2, tokenizer2, "transformer")
        >>> results = benchmark.run_speed_benchmark([1024, 4096, 16384])
        >>> benchmark.save_results(results, "benchmark_results.json")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the benchmark suite.

        Args:
            device: Device to run benchmarks on
        """
        self.device = device
        self.models: Dict[str, Tuple[Any, Any, str]] = {}  # name -> (model, tokenizer, arch_type)
        self.results: List[BenchmarkResult] = []

    def add_model(
        self,
        name: str,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        architecture_type: str,
    ) -> None:
        """
        Register a model for benchmarking.

        Args:
            name: Unique identifier for this model
            model: Loaded model
            tokenizer: Corresponding tokenizer
            architecture_type: One of "transformer", "mamba", "moe", "jamba"
        """
        if architecture_type not in ARCHITECTURE_PROFILES:
            print(f"Warning: Unknown architecture '{architecture_type}'. "
                  f"Known types: {list(ARCHITECTURE_PROFILES.keys())}")

        self.models[name] = (model, tokenizer, architecture_type)
        print(f"Added model '{name}' ({architecture_type})")

    def run_speed_benchmark(
        self,
        context_lengths: List[int] = [1024, 4096, 8192, 16384],
        generation_length: int = 100,
        warmup_runs: int = 2,
        benchmark_runs: int = 3,
    ) -> List[BenchmarkResult]:
        """
        Benchmark generation speed across context lengths.

        Args:
            context_lengths: List of input lengths to test
            generation_length: Tokens to generate each run
            warmup_runs: Iterations to warm up before timing
            benchmark_runs: Iterations to time and average

        Returns:
            List of BenchmarkResult for each model/context combination
        """
        results = []

        for model_name, (model, tokenizer, arch_type) in self.models.items():
            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name} ({arch_type})")
            print(f"{'='*60}")

            for ctx_len in context_lengths:
                print(f"\n  Context: {ctx_len:,} tokens")

                try:
                    result = self._benchmark_single(
                        model=model,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        arch_type=arch_type,
                        context_length=ctx_len,
                        generation_length=generation_length,
                        warmup_runs=warmup_runs,
                        benchmark_runs=benchmark_runs,
                    )
                    results.append(result)
                    print(f"    Speed: {result.tokens_per_second:.1f} tok/s")
                    print(f"    Memory: {result.memory_peak_gb:.2f} GB peak")

                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    # Record failed benchmark
                    results.append(BenchmarkResult(
                        model_name=model_name,
                        architecture_type=arch_type,
                        context_length=ctx_len,
                        generation_length=generation_length,
                        total_time_seconds=float('inf'),
                        time_to_first_token=float('inf'),
                        tokens_per_second=0,
                        memory_peak_gb=0,
                        memory_model_gb=0,
                        metadata={"error": str(e)},
                    ))

        self.results.extend(results)
        return results

    def _benchmark_single(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        model_name: str,
        arch_type: str,
        context_length: int,
        generation_length: int,
        warmup_runs: int,
        benchmark_runs: int,
    ) -> BenchmarkResult:
        """Execute a single benchmark run."""
        # Create input
        input_ids = torch.randint(
            100, tokenizer.vocab_size - 100,
            (1, context_length),
            device=self.device
        )

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        # Clear and prepare for benchmark
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Model memory (before generation)
        model_memory = torch.cuda.memory_allocated() / 1e9

        # Timed runs
        times = []
        for _ in range(benchmark_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=generation_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        return BenchmarkResult(
            model_name=model_name,
            architecture_type=arch_type,
            context_length=context_length,
            generation_length=generation_length,
            total_time_seconds=avg_time,
            time_to_first_token=avg_time / generation_length * 2,  # Approximate
            tokens_per_second=generation_length / avg_time,
            memory_peak_gb=peak_memory,
            memory_model_gb=model_memory,
            metadata={"benchmark_runs": benchmark_runs, "times": times},
        )

    def run_perplexity_benchmark(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        max_samples: int = 100,
        stride: int = 512,
    ) -> List[BenchmarkResult]:
        """
        Benchmark perplexity on a standard dataset.

        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            max_samples: Maximum samples to evaluate
            stride: Sliding window stride for long sequences

        Returns:
            List of BenchmarkResult with perplexity scores
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library required: pip install datasets")

        print(f"\nLoading {dataset_name}/{dataset_config} ({split})...")
        dataset = load_dataset(dataset_name, dataset_config, split=split)

        # Concatenate text
        text = "\n\n".join(dataset["text"][:max_samples])

        results = []

        for model_name, (model, tokenizer, arch_type) in self.models.items():
            print(f"\nCalculating perplexity for {model_name}...")

            try:
                perplexity = self._calculate_perplexity(
                    model, tokenizer, text, stride
                )
                print(f"  Perplexity: {perplexity:.2f}")

                result = BenchmarkResult(
                    model_name=model_name,
                    architecture_type=arch_type,
                    context_length=0,
                    generation_length=0,
                    total_time_seconds=0,
                    time_to_first_token=0,
                    tokens_per_second=0,
                    memory_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                    memory_model_gb=torch.cuda.memory_allocated() / 1e9,
                    perplexity=perplexity,
                    metadata={"dataset": f"{dataset_name}/{dataset_config}"},
                )
                results.append(result)

            except Exception as e:
                print(f"  ERROR: {str(e)}")

        return results

    def _calculate_perplexity(
        self,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
        text: str,
        stride: int,
    ) -> float:
        """Calculate perplexity using sliding window."""
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)

        max_length = getattr(model.config, "max_position_embeddings", 2048)
        seq_len = input_ids.size(1)

        nlls = []
        prev_end = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end
            input_chunk = input_ids[:, begin:end]

            with torch.no_grad():
                outputs = model(input_chunk, labels=input_chunk)
                nll = outputs.loss * target_len

            nlls.append(nll)
            prev_end = end

            if end >= seq_len:
                break

        perplexity = torch.exp(torch.stack(nlls).sum() / prev_end)
        return perplexity.item()

    def save_results(
        self,
        results: List[BenchmarkResult],
        filepath: Union[str, Path],
    ) -> None:
        """
        Save benchmark results to JSON.

        Args:
            results: List of BenchmarkResult to save
            filepath: Output file path
        """
        filepath = Path(filepath)
        data = [asdict(r) for r in results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Results saved to {filepath}")

    @staticmethod
    def load_results(filepath: Union[str, Path]) -> List[BenchmarkResult]:
        """
        Load benchmark results from JSON.

        Args:
            filepath: Path to results file

        Returns:
            List of BenchmarkResult
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        return [BenchmarkResult(**r) for r in data]


def measure_memory_usage(
    model: "AutoModelForCausalLM",
    include_overhead: bool = True,
) -> Dict[str, float]:
    """
    Measure memory usage of a loaded model.

    Args:
        model: Loaded model
        include_overhead: Include PyTorch overhead estimates

    Returns:
        Dictionary with memory statistics in GB
    """
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9

    # Buffers (if any)
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9

    # CUDA statistics
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9

    return {
        "model_params_gb": param_memory,
        "model_buffers_gb": buffer_memory,
        "cuda_allocated_gb": allocated,
        "cuda_reserved_gb": reserved,
        "cuda_max_allocated_gb": max_allocated,
        "dgx_spark_available_gb": 128.0 - reserved,
        "utilization_percent": (allocated / 128.0) * 100,
    }


def measure_inference_speed(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str = "The quick brown fox",
    num_tokens: int = 100,
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    Measure inference speed for a model.

    Args:
        model: Loaded model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt
        num_tokens: Tokens to generate
        num_runs: Number of timing runs

    Returns:
        Dictionary with speed metrics
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean_time_seconds": np.mean(times),
        "std_time_seconds": np.std(times),
        "tokens_per_second": num_tokens / np.mean(times),
        "time_per_token_ms": (np.mean(times) / num_tokens) * 1000,
    }


def compare_architectures(
    models: Dict[str, "AutoModelForCausalLM"],
    tokenizers: Dict[str, "AutoTokenizer"],
    architecture_types: Dict[str, str],
    context_lengths: List[int] = [1024, 4096, 16384],
) -> Dict[str, List[BenchmarkResult]]:
    """
    High-level function to compare multiple architectures.

    Args:
        models: Dict mapping name to model
        tokenizers: Dict mapping name to tokenizer
        architecture_types: Dict mapping name to architecture type
        context_lengths: Context lengths to test

    Returns:
        Dictionary mapping model name to list of results
    """
    benchmark = ArchitectureBenchmark()

    for name in models:
        benchmark.add_model(
            name=name,
            model=models[name],
            tokenizer=tokenizers[name],
            architecture_type=architecture_types[name],
        )

    results = benchmark.run_speed_benchmark(context_lengths)

    # Group by model
    grouped = {}
    for result in results:
        if result.model_name not in grouped:
            grouped[result.model_name] = []
        grouped[result.model_name].append(result)

    return grouped


def plot_benchmark_results(
    results: List[BenchmarkResult],
    metric: str = "tokens_per_second",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot benchmark results for visual comparison.

    Args:
        results: List of BenchmarkResult
        metric: Which metric to plot
        save_path: Optional path to save the figure

    Note:
        Requires matplotlib. Install with: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting: pip install matplotlib")
        return

    # Organize data
    models = list(set(r.model_name for r in results))
    contexts = sorted(set(r.context_length for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name in models:
        model_results = [r for r in results if r.model_name == model_name]
        model_results.sort(key=lambda x: x.context_length)

        x = [r.context_length for r in model_results]
        y = [getattr(r, metric) for r in model_results]

        ax.plot(x, y, marker='o', label=model_name, linewidth=2)

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Architecture Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale often useful for context length
    ax.set_xscale('log', base=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def print_comparison_table(results: List[BenchmarkResult]) -> None:
    """
    Print a formatted comparison table of results.

    Args:
        results: List of BenchmarkResult
    """
    # Group by context length
    contexts = sorted(set(r.context_length for r in results))

    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("=" * 80)

    for ctx_len in contexts:
        print(f"\n--- Context Length: {ctx_len:,} tokens ---")
        print(f"{'Model':<20} {'Type':<12} {'Speed (tok/s)':<15} {'Memory (GB)':<12} {'PPL':<10}")
        print("-" * 70)

        ctx_results = [r for r in results if r.context_length == ctx_len]
        ctx_results.sort(key=lambda x: -x.tokens_per_second)  # Sort by speed

        for r in ctx_results:
            ppl = f"{r.perplexity:.2f}" if r.perplexity else "N/A"
            print(f"{r.model_name:<20} {r.architecture_type:<12} "
                  f"{r.tokens_per_second:<15.1f} {r.memory_peak_gb:<12.2f} {ppl:<10}")


if __name__ == "__main__":
    print("Benchmark Utilities Module")
    print("=" * 50)

    # Print architecture profiles
    print("\nArchitecture Profiles:")
    for arch_name, profile in ARCHITECTURE_PROFILES.items():
        print(f"\n{profile.name}:")
        print(f"  Attention: {profile.attention_complexity}")
        print(f"  Memory: {profile.memory_scaling}")
        print(f"  KV Cache: {'Required' if profile.kv_cache_required else 'Not needed'}")
        print(f"  Sparse: {'Yes' if profile.sparse_activation else 'No'}")
        print(f"  Typical context: {profile.typical_context_limit:,}")
        print(f"  DGX Spark context: {profile.dgx_spark_context_limit:,}")

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem:.1f} GB")
