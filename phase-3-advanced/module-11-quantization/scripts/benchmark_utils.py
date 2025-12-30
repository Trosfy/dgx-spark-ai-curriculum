"""
Benchmark Utilities for DGX Spark Quantization

This module provides tools for benchmarking quantized model performance
including inference speed, memory usage, and quality metrics.

Example:
    >>> from benchmark_utils import benchmark_inference_speed, BenchmarkResult
    >>>
    >>> result = benchmark_inference_speed(model, tokenizer, prompt="Hello")
    >>> print(f"Speed: {result.tokens_per_second:.1f} tok/s")
"""

import torch
import time
import gc
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class BenchmarkResult:
    """
    Container for benchmark results.

    Attributes:
        model_name: Name/path of the model
        quantization_type: Type of quantization applied
        model_size_mb: Model size in megabytes
        perplexity: Perplexity score (if calculated)
        tokens_per_second: Generation speed
        prefill_tokens_per_second: Prefill/prompt processing speed
        memory_used_gb: GPU memory used
        latency_ms: Average latency per request
        task_scores: Dictionary of task-specific scores
        metadata: Additional metadata
    """
    model_name: str
    quantization_type: str
    model_size_mb: float
    perplexity: Optional[float] = None
    tokens_per_second: Optional[float] = None
    prefill_tokens_per_second: Optional[float] = None
    memory_used_gb: Optional[float] = None
    latency_ms: Optional[float] = None
    task_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compression_ratio(self, baseline_size_mb: float) -> float:
        """Calculate compression ratio vs baseline."""
        return baseline_size_mb / self.model_size_mb

    def quality_retained(self, baseline_ppl: float) -> float:
        """Calculate quality retention percentage."""
        if self.perplexity is None or baseline_ppl is None:
            return None
        # Lower perplexity is better, so we invert
        return baseline_ppl / self.perplexity * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        result = {
            'Model': self.model_name,
            'Quantization': self.quantization_type,
            'Size (MB)': self.model_size_mb,
            'Perplexity': self.perplexity,
            'Tokens/s': self.tokens_per_second,
            'Prefill tok/s': self.prefill_tokens_per_second,
            'Memory (GB)': self.memory_used_gb,
            'Latency (ms)': self.latency_ms,
        }
        result.update(self.task_scores)
        return result


def benchmark_inference_speed(
    model,
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    num_tokens: int = 50,
    num_warmup: int = 2,
    num_runs: int = 5,
    batch_size: int = 1
) -> BenchmarkResult:
    """
    Benchmark model inference speed.

    Args:
        model: The language model to benchmark
        tokenizer: The tokenizer
        prompt: Input prompt for generation
        num_tokens: Number of tokens to generate
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        batch_size: Batch size for inference

    Returns:
        BenchmarkResult with timing information

    Example:
        >>> result = benchmark_inference_speed(model, tokenizer)
        >>> print(f"Speed: {result.tokens_per_second:.1f} tok/s")
    """
    model.eval()

    # Safely get model device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # Model has no parameters (edge case)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare input
    if batch_size > 1:
        prompts = [prompt] * batch_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark generation
    generation_times = []
    prefill_times = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure prefill (first token)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start
        prefill_times.append(prefill_time)

        # Measure full generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_times.append(time.perf_counter() - start)

    avg_gen_time = sum(generation_times) / len(generation_times)
    avg_prefill_time = sum(prefill_times) / len(prefill_times)

    tokens_per_second = (num_tokens * batch_size) / avg_gen_time
    prefill_tps = inputs['input_ids'].numel() / avg_prefill_time

    # Get memory usage
    memory_gb = 0
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1e9

    # Get model size
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    model_size_mb = param_bytes / 1e6

    return BenchmarkResult(
        model_name=getattr(model.config, '_name_or_path', 'unknown'),
        quantization_type='unknown',
        model_size_mb=model_size_mb,
        tokens_per_second=tokens_per_second,
        prefill_tokens_per_second=prefill_tps,
        memory_used_gb=memory_gb,
        latency_ms=avg_gen_time * 1000
    )


def benchmark_memory_usage(
    model_loader: Callable,
    *args,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark memory usage during model loading.

    Args:
        model_loader: Function that loads the model
        *args, **kwargs: Arguments to pass to model_loader

    Returns:
        Dictionary with memory statistics

    Example:
        >>> def load_model():
        ...     return AutoModelForCausalLM.from_pretrained("gpt2")
        >>> mem_stats = benchmark_memory_usage(load_model)
        >>> print(f"Peak memory: {mem_stats['peak_mb']:.1f} MB")
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
    else:
        initial_memory = 0

    # Load model
    model = model_loader(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        final_memory = 0
        peak_memory = 0

    return {
        'initial_mb': initial_memory / 1e6,
        'final_mb': final_memory / 1e6,
        'peak_mb': peak_memory / 1e6,
        'model_mb': (final_memory - initial_memory) / 1e6,
        'overhead_mb': (peak_memory - final_memory) / 1e6,
    }


def compare_models(
    results: List[BenchmarkResult],
    baseline_idx: int = 0
) -> pd.DataFrame:
    """
    Compare multiple benchmark results.

    Args:
        results: List of BenchmarkResult objects
        baseline_idx: Index of baseline result for comparison

    Returns:
        DataFrame with comparison metrics
    """
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in results])

    # Add comparison columns
    baseline = results[baseline_idx]

    if baseline.model_size_mb > 0:
        df['Compression'] = baseline.model_size_mb / df['Size (MB)']
        df['Compression'] = df['Compression'].apply(lambda x: f"{x:.1f}x")

    if baseline.perplexity is not None:
        df['PPL Delta'] = df['Perplexity'] - baseline.perplexity
        df['PPL Delta'] = df['PPL Delta'].apply(
            lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}" if x != 0 else "-"
        )

    if baseline.tokens_per_second is not None and baseline.tokens_per_second > 0:
        df['Speedup'] = df['Tokens/s'] / baseline.tokens_per_second
        df['Speedup'] = df['Speedup'].apply(lambda x: f"{x:.2f}x")

    return df


def run_benchmark_suite(
    model_configs: List[Dict[str, Any]],
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    num_tokens: int = 50,
    eval_texts: Optional[List[str]] = None
) -> List[BenchmarkResult]:
    """
    Run a complete benchmark suite on multiple model configurations.

    Args:
        model_configs: List of dicts with 'name', 'loader', and optional 'type'
        tokenizer: Shared tokenizer
        prompt: Generation prompt
        num_tokens: Tokens to generate
        eval_texts: Texts for perplexity evaluation

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for config in model_configs:
        name = config.get('name', 'unknown')
        loader = config.get('loader')
        quant_type = config.get('type', 'unknown')

        print(f"\nBenchmarking {name} ({quant_type})...")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Load model
            model = loader()

            # Run benchmark
            result = benchmark_inference_speed(
                model, tokenizer, prompt, num_tokens
            )
            result.model_name = name
            result.quantization_type = quant_type

            # Calculate perplexity if texts provided
            if eval_texts:
                try:
                    from .perplexity import calculate_perplexity
                except ImportError:
                    # Fallback for when running outside package context
                    from perplexity import calculate_perplexity
                result.perplexity = calculate_perplexity(
                    model, tokenizer, eval_texts
                )

            results.append(result)

            print(f"  Size: {result.model_size_mb:.1f} MB")
            print(f"  Speed: {result.tokens_per_second:.1f} tok/s")
            if result.perplexity:
                print(f"  Perplexity: {result.perplexity:.2f}")

            # Clean up
            del model

        except Exception as e:
            print(f"  Error: {e}")
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    print("Benchmark Utils Demo")
    print("=" * 50)

    # Example: Create mock results
    results = [
        BenchmarkResult(
            model_name="test-model",
            quantization_type="FP16",
            model_size_mb=700,
            perplexity=15.5,
            tokens_per_second=25.0,
            memory_used_gb=0.7
        ),
        BenchmarkResult(
            model_name="test-model",
            quantization_type="INT8",
            model_size_mb=350,
            perplexity=15.8,
            tokens_per_second=35.0,
            memory_used_gb=0.4
        ),
        BenchmarkResult(
            model_name="test-model",
            quantization_type="INT4",
            model_size_mb=175,
            perplexity=16.5,
            tokens_per_second=45.0,
            memory_used_gb=0.2
        ),
    ]

    df = compare_models(results)
    print("\nComparison Table:")
    print(df.to_string(index=False))
