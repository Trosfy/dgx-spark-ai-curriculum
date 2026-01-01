"""
Benchmark Utilities for DGX Spark Quantization Experiments

This module provides inference benchmarking functions optimized for comparing
quantization methods on DGX Spark's 128GB unified memory architecture.

Features:
- Single and batch inference benchmarking
- Model comparison utilities
- Quantization method comparison
- Configurable warmup and measurement runs
- Results collection and reporting

Example:
    >>> from benchmark_utils import benchmark_inference, compare_models
    >>>
    >>> result = benchmark_inference(model, tokenizer, "Hello world")
    >>> print(f"Throughput: {result.tokens_per_second:.1f} tok/s")
    >>>
    >>> comparison = compare_models([model_fp16, model_int4], tokenizer, ["FP16", "INT4"])
    >>> comparison.report()

Note:
    For performance testing, use your Ollama Web UI to verify benchmark
    results with consistent test conditions.
"""

import torch
import time
import gc
from typing import List, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model_name: str = ""
    precision: str = ""
    tokens_per_second: float = 0.0
    prefill_tokens_per_second: float = 0.0
    decode_tokens_per_second: float = 0.0
    latency_ms: float = 0.0
    memory_gb: float = 0.0
    batch_size: int = 1
    num_tokens_generated: int = 0
    num_runs: int = 1
    times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'model_name': self.model_name,
            'precision': self.precision,
            'tokens_per_second': self.tokens_per_second,
            'prefill_tok_s': self.prefill_tokens_per_second,
            'decode_tok_s': self.decode_tokens_per_second,
            'latency_ms': self.latency_ms,
            'memory_gb': self.memory_gb,
            'batch_size': self.batch_size,
            'num_tokens_generated': self.num_tokens_generated,
        }

    def __str__(self) -> str:
        return (
            f"BenchmarkResult(throughput={self.tokens_per_second:.1f} tok/s, "
            f"latency={self.latency_ms:.1f}ms, memory={self.memory_gb:.2f}GB)"
        )


@dataclass
class ComparisonResult:
    """Result from comparing multiple models."""
    results: List[BenchmarkResult] = field(default_factory=list)
    prompt: str = ""
    max_new_tokens: int = 0

    def report(self) -> None:
        """Print a formatted comparison report."""
        print("\n" + "=" * 70)
        print("Model Comparison Report")
        print("=" * 70)
        print(f"Prompt: {self.prompt[:50]}..." if len(self.prompt) > 50 else f"Prompt: {self.prompt}")
        print(f"Max new tokens: {self.max_new_tokens}")
        print("-" * 70)
        print(f"{'Model':<20} {'Precision':<10} {'tok/s':>12} {'Latency':>12} {'Memory':>12}")
        print("-" * 70)

        for r in self.results:
            print(
                f"{r.model_name:<20} {r.precision:<10} "
                f"{r.tokens_per_second:>12.1f} {r.latency_ms:>10.1f}ms {r.memory_gb:>10.2f}GB"
            )

        print("=" * 70)

        # Find best
        if self.results:
            best_speed = max(self.results, key=lambda x: x.tokens_per_second)
            best_memory = min(self.results, key=lambda x: x.memory_gb)
            print(f"Fastest: {best_speed.model_name} ({best_speed.precision}) at {best_speed.tokens_per_second:.1f} tok/s")
            print(f"Smallest: {best_memory.model_name} ({best_memory.precision}) at {best_memory.memory_gb:.2f} GB")

    def to_dataframe(self):
        """Convert results to pandas DataFrame if available."""
        try:
            import pandas as pd
            return pd.DataFrame([r.to_dict() for r in self.results])
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")


def _get_gpu_memory() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def _clear_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_inference(
    model,
    tokenizer,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 50,
    num_runs: int = 5,
    warmup_runs: int = 2,
    batch_size: int = 1,
    model_name: str = "",
    precision: str = "",
    device: Optional[str] = None,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Benchmark model inference performance.

    Args:
        model: The language model to benchmark
        tokenizer: The tokenizer
        prompt: Input prompt for generation
        max_new_tokens: Number of tokens to generate
        num_runs: Number of benchmark runs (for averaging)
        warmup_runs: Number of warmup runs before measuring
        batch_size: Batch size (prompts will be duplicated)
        model_name: Name to record in results
        precision: Precision label (e.g., "FP16", "INT4", "NVFP4")
        device: Device to use (defaults to model's device)
        verbose: Print progress messages

    Returns:
        BenchmarkResult with timing and memory information

    Example:
        >>> result = benchmark_inference(model, tokenizer, "Hello", max_new_tokens=50)
        >>> print(f"Throughput: {result.tokens_per_second:.1f} tok/s")

    Note:
        For consistent benchmarks, clear buffer cache before running:
        sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

        Verify results with your Ollama Web UI for cross-validation.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Prepare inputs
    if batch_size > 1:
        prompts = [prompt] * batch_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Warmup
    if verbose:
        print(f"  Warming up ({warmup_runs} runs)...")

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    if verbose:
        print(f"  Running benchmark ({num_runs} runs)...")

    times = []

    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    # Calculate metrics
    avg_time = sum(times) / len(times)
    total_tokens = max_new_tokens * batch_size
    tokens_per_second = total_tokens / avg_time
    memory_gb = _get_gpu_memory()

    result = BenchmarkResult(
        model_name=model_name or "model",
        precision=precision or "unknown",
        tokens_per_second=tokens_per_second,
        decode_tokens_per_second=tokens_per_second,  # Simplified: assume mostly decode
        latency_ms=avg_time * 1000,
        memory_gb=memory_gb,
        batch_size=batch_size,
        num_tokens_generated=total_tokens,
        num_runs=num_runs,
        times=times,
    )

    if verbose:
        print(f"  Result: {tokens_per_second:.1f} tok/s, {avg_time*1000:.1f}ms, {memory_gb:.2f}GB")

    return result


def benchmark_batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    num_runs: int = 3,
    warmup_runs: int = 1,
    model_name: str = "",
    precision: str = "",
    verbose: bool = True
) -> BenchmarkResult:
    """
    Benchmark model with multiple different prompts (batch processing).

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Tokens to generate per prompt
        num_runs: Number of benchmark runs
        warmup_runs: Warmup runs before measuring
        model_name: Name for results
        precision: Precision label
        verbose: Print progress

    Returns:
        BenchmarkResult with averaged metrics

    Note:
        Verify batch performance with your Ollama Web UI for accurate
        throughput measurements.
    """
    device = next(model.parameters()).device
    model.eval()

    # Tokenize all prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Warmup
    if verbose:
        print(f"  Warming up batch inference...")

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    if verbose:
        print(f"  Running batch benchmark ({num_runs} runs, batch_size={len(prompts)})...")

    times = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    total_tokens = max_new_tokens * len(prompts)
    tokens_per_second = total_tokens / avg_time

    return BenchmarkResult(
        model_name=model_name or "model",
        precision=precision or "unknown",
        tokens_per_second=tokens_per_second,
        decode_tokens_per_second=tokens_per_second,
        latency_ms=avg_time * 1000,
        memory_gb=_get_gpu_memory(),
        batch_size=len(prompts),
        num_tokens_generated=total_tokens,
        num_runs=num_runs,
        times=times,
    )


def compare_models(
    models: List,
    tokenizer,
    model_names: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 50,
    num_runs: int = 5,
    verbose: bool = True
) -> ComparisonResult:
    """
    Compare inference performance across multiple models.

    Args:
        models: List of models to compare
        tokenizer: Shared tokenizer
        model_names: Names for each model
        precisions: Precision labels for each model
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        num_runs: Benchmark runs per model
        verbose: Print progress

    Returns:
        ComparisonResult with all benchmark results

    Example:
        >>> result = compare_models(
        ...     [model_fp16, model_int4, model_nvfp4],
        ...     tokenizer,
        ...     model_names=["Llama 8B", "Llama 8B", "Llama 8B"],
        ...     precisions=["FP16", "INT4", "NVFP4"]
        ... )
        >>> result.report()

    Note:
        For accurate comparisons, verify results with your Ollama Web UI
        to ensure consistent test conditions.
    """
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]
    if precisions is None:
        precisions = ["unknown"] * len(models)

    results = []

    for model, name, precision in zip(models, model_names, precisions):
        if verbose:
            print(f"\nBenchmarking {name} ({precision})...")

        result = benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
            model_name=name,
            precision=precision,
            verbose=verbose
        )
        results.append(result)

        # Clear memory between models
        _clear_memory()

    return ComparisonResult(
        results=results,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )


def benchmark_quantization_methods(
    quantize_fn: Callable,
    base_model,
    tokenizer,
    methods: List[Dict[str, Any]],
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 50,
    num_runs: int = 5,
    verbose: bool = True
) -> ComparisonResult:
    """
    Benchmark different quantization methods on the same base model.

    Args:
        quantize_fn: Function that takes (model, **method_kwargs) and returns quantized model
        base_model: The base model to quantize
        tokenizer: The tokenizer
        methods: List of dicts with 'name', 'precision', and quantization kwargs
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        num_runs: Benchmark runs per method
        verbose: Print progress

    Returns:
        ComparisonResult with benchmarks for each method

    Example:
        >>> methods = [
        ...     {'name': 'INT8', 'precision': 'INT8', 'bits': 8},
        ...     {'name': 'INT4', 'precision': 'INT4', 'bits': 4},
        ...     {'name': 'NVFP4', 'precision': 'NVFP4', 'use_fp4': True},
        ... ]
        >>> results = benchmark_quantization_methods(
        ...     quantize_fn=my_quantize,
        ...     base_model=model,
        ...     tokenizer=tokenizer,
        ...     methods=methods
        ... )
        >>> results.report()

    Note:
        Expected performance benchmarks (verify with Ollama Web UI):
        - 8B NVFP4: ~10,000 prefill tok/s, ~38 decode tok/s
        - 8B Q4: ~3,000 prefill tok/s, ~45 decode tok/s
        - 70B Q4: ~500 prefill tok/s, ~15 decode tok/s
    """
    results = []

    for method in methods:
        name = method.pop('name', 'unknown')
        precision = method.pop('precision', 'unknown')

        if verbose:
            print(f"\nQuantizing with {name} ({precision})...")

        try:
            # Quantize model
            quantized_model = quantize_fn(base_model, **method)

            # Benchmark
            result = benchmark_inference(
                model=quantized_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
                model_name=name,
                precision=precision,
                verbose=verbose
            )
            results.append(result)

            # Cleanup
            del quantized_model
            _clear_memory()

        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
            continue

        # Restore method dict for potential reuse
        method['name'] = name
        method['precision'] = precision

    return ComparisonResult(
        results=results,
        prompt=prompt,
        max_new_tokens=max_new_tokens
    )


if __name__ == "__main__":
    print("Benchmark Utils Demo")
    print("=" * 60)

    print("\nThis module provides:")
    print("  - benchmark_inference(): Single model benchmarking")
    print("  - benchmark_batch_inference(): Batch processing benchmarks")
    print("  - compare_models(): Multi-model comparison")
    print("  - benchmark_quantization_methods(): Compare quantization methods")
    print("")
    print("Example usage:")
    print("  >>> from benchmark_utils import benchmark_inference")
    print("  >>> result = benchmark_inference(model, tokenizer, 'Hello')")
    print("  >>> print(f'Throughput: {result.tokens_per_second:.1f} tok/s')")
    print("")
    print("Expected DGX Spark performance (verify with Ollama Web UI):")
    print("  | Model         | Precision | Prefill tok/s | Decode tok/s |")
    print("  |---------------|-----------|---------------|--------------|")
    print("  | Llama 3.1 8B  | NVFP4     | ~10,000       | ~38          |")
    print("  | Llama 3.1 8B  | Q4        | ~3,000        | ~45          |")
    print("  | Llama 3.1 70B | NVFP4     | ~2,500        | ~15          |")
    print("  | Llama 3.1 70B | Q4        | ~500          | ~15          |")
