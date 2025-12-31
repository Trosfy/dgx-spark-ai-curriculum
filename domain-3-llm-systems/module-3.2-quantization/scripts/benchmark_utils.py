"""
Benchmark Utilities for LLM Quantization Experiments

This module provides comprehensive benchmarking functions for measuring
inference performance of quantized models on DGX Spark.

Features:
- Token-level throughput measurement
- Latency profiling (prefill + decode)
- Memory efficiency tracking
- Multi-run statistical analysis
- Comparison across quantization methods

Example:
    >>> from benchmark_utils import benchmark_inference, compare_models
    >>>
    >>> results = benchmark_inference(model, tokenizer, "Hello world")
    >>> print(f"Throughput: {results['tokens_per_second']:.1f} tok/s")
    >>>
    >>> comparison = compare_models(
    ...     models={'fp16': model_fp16, 'int4': model_int4},
    ...     tokenizer=tokenizer
    ... )
    >>> comparison.print_summary()
"""

import torch
import time
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from statistics import mean, stdev
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    tokens_per_second: float
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    memory_gb: float
    num_tokens: int
    num_runs: int
    prompt_tokens: int
    raw_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'tokens_per_second': self.tokens_per_second,
            'prefill_latency_ms': self.prefill_latency_ms,
            'decode_latency_ms': self.decode_latency_ms,
            'total_latency_ms': self.total_latency_ms,
            'memory_gb': self.memory_gb,
            'num_tokens': self.num_tokens,
            'num_runs': self.num_runs,
            'prompt_tokens': self.prompt_tokens,
            'std_dev_ms': stdev(self.raw_times) * 1000 if len(self.raw_times) > 1 else 0
        }


def get_gpu_memory() -> float:
    """Get current GPU memory allocated in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def warmup_model(
    model,
    tokenizer,
    num_warmup: int = 3
) -> None:
    """
    Warm up the model to ensure accurate benchmarks.

    Args:
        model: The language model
        tokenizer: The tokenizer
        num_warmup: Number of warmup iterations
    """
    model.eval()
    inputs = tokenizer("Warmup prompt for benchmarking", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_inference(
    model,
    tokenizer,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 50,
    num_runs: int = 5,
    warmup_runs: int = 2,
    name: str = "model",
    measure_prefill: bool = True
) -> BenchmarkResult:
    """
    Benchmark inference performance of a language model.

    Args:
        model: The language model to benchmark
        tokenizer: The tokenizer
        prompt: Input prompt for generation
        max_new_tokens: Number of tokens to generate
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs before timing
        name: Name for this benchmark (for reporting)
        measure_prefill: If True, separately measure prefill latency

    Returns:
        BenchmarkResult with timing and memory information

    Example:
        >>> result = benchmark_inference(model, tokenizer, "Hello", max_new_tokens=100)
        >>> print(f"Speed: {result.tokens_per_second:.1f} tokens/sec")
    """
    model.eval()

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_tokens = inputs['input_ids'].shape[1]

    # Set pad token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Warmup
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

    # Measure prefill time (first token latency)
    prefill_time = 0.0
    if measure_prefill:
        prefill_times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_times.append(time.perf_counter() - start)
        prefill_time = mean(prefill_times)

    # Benchmark full generation
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

    # Calculate statistics
    avg_time = mean(times)
    tokens_per_second = max_new_tokens / avg_time

    # Decode time is total minus prefill
    decode_time = avg_time - prefill_time if measure_prefill else avg_time

    # Get memory usage
    memory_gb = get_gpu_memory()

    return BenchmarkResult(
        name=name,
        tokens_per_second=tokens_per_second,
        prefill_latency_ms=prefill_time * 1000,
        decode_latency_ms=decode_time * 1000,
        total_latency_ms=avg_time * 1000,
        memory_gb=memory_gb,
        num_tokens=max_new_tokens,
        num_runs=num_runs,
        prompt_tokens=prompt_tokens,
        raw_times=times
    )


def benchmark_batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    batch_size: int = 1,
    name: str = "model"
) -> Dict[str, float]:
    """
    Benchmark batch inference with multiple prompts.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Tokens to generate per prompt
        batch_size: Batch size for processing
        name: Name for this benchmark

    Returns:
        Dict with throughput and latency metrics

    Example:
        >>> prompts = ["Hello", "The weather is", "AI will"]
        >>> results = benchmark_batch_inference(model, tokenizer, prompts, batch_size=2)
    """
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_tokens = 0
    total_time = 0.0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time += time.perf_counter() - start
        total_tokens += max_new_tokens * len(batch)

    return {
        'name': name,
        'total_prompts': len(prompts),
        'batch_size': batch_size,
        'total_tokens': total_tokens,
        'total_time_s': total_time,
        'tokens_per_second': total_tokens / total_time,
        'prompts_per_second': len(prompts) / total_time,
        'memory_gb': get_gpu_memory()
    }


@dataclass
class ComparisonResult:
    """Results from comparing multiple models."""
    results: Dict[str, BenchmarkResult]
    baseline_name: str

    def print_summary(self) -> None:
        """Print a formatted comparison table."""
        print("\n" + "=" * 80)
        print("Model Comparison Results")
        print("=" * 80)
        print(f"{'Model':<15} {'Tok/s':>10} {'Prefill':>12} {'Memory':>10} {'Speedup':>10}")
        print("-" * 80)

        baseline_speed = self.results[self.baseline_name].tokens_per_second

        for name, result in self.results.items():
            speedup = result.tokens_per_second / baseline_speed
            speedup_str = f"{speedup:.2f}x" if name != self.baseline_name else "baseline"
            print(f"{name:<15} {result.tokens_per_second:>10.1f} "
                  f"{result.prefill_latency_ms:>10.1f}ms "
                  f"{result.memory_gb:>9.2f}GB "
                  f"{speedup_str:>10}")

        print("=" * 80)

    def get_best(self, metric: str = 'tokens_per_second') -> str:
        """Get the name of the best model by metric."""
        if metric in ['tokens_per_second']:
            return max(self.results.items(), key=lambda x: getattr(x[1], metric))[0]
        else:  # Lower is better for latency/memory
            return min(self.results.items(), key=lambda x: getattr(x[1], metric))[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'baseline': self.baseline_name,
            'results': {name: result.to_dict() for name, result in self.results.items()}
        }


def compare_models(
    models: Dict[str, Any],
    tokenizer,
    prompt: str = "The future of artificial intelligence is",
    max_new_tokens: int = 50,
    num_runs: int = 5,
    baseline_name: Optional[str] = None
) -> ComparisonResult:
    """
    Compare inference performance across multiple models.

    Args:
        models: Dict mapping names to model objects
        tokenizer: Shared tokenizer
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        num_runs: Benchmark runs per model
        baseline_name: Name of baseline model (default: first model)

    Returns:
        ComparisonResult with all benchmark data

    Example:
        >>> models = {
        ...     'fp16': model_fp16,
        ...     'int8': model_int8,
        ...     'int4': model_int4
        ... }
        >>> comparison = compare_models(models, tokenizer)
        >>> comparison.print_summary()
    """
    if baseline_name is None:
        baseline_name = list(models.keys())[0]

    results = {}
    for name, model in models.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
            name=name
        )
        print(f"  {results[name].tokens_per_second:.1f} tok/s, "
              f"{results[name].memory_gb:.2f} GB")

    return ComparisonResult(results=results, baseline_name=baseline_name)


def benchmark_quantization_methods(
    model_name: str,
    methods: List[str] = ['fp16', 'int8', 'int4'],
    prompt: str = "Explain machine learning in simple terms:",
    max_new_tokens: int = 100,
    num_runs: int = 5,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive benchmark across quantization methods.

    Loads each model variant, benchmarks it, and cleans up memory.

    Args:
        model_name: HuggingFace model name
        methods: List of quantization methods to test
        prompt: Test prompt
        max_new_tokens: Tokens to generate
        num_runs: Benchmark runs per method
        output_path: Optional path to save results as JSON

    Returns:
        Dict with comprehensive benchmark results

    Example:
        >>> results = benchmark_quantization_methods(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     methods=['fp16', 'int8', 'int4']
        ... )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing: {method}")
        print(f"{'='*60}")

        # Clear memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Load model with appropriate quantization
            if method == 'fp32':
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cuda"
                )
            elif method == 'fp16':
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cuda"
                )
            elif method == 'bf16':
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda"
                )
            elif method == 'int8':
                config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=config,
                    device_map="cuda"
                )
            elif method == 'int4':
                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=config,
                    device_map="cuda"
                )
            else:
                print(f"Unknown method: {method}, skipping")
                continue

            # Run benchmark
            result = benchmark_inference(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_runs=num_runs,
                name=method
            )
            results[method] = result.to_dict()

            print(f"Results for {method}:")
            print(f"  Tokens/sec: {result.tokens_per_second:.1f}")
            print(f"  Memory: {result.memory_gb:.2f} GB")
            print(f"  Latency: {result.total_latency_ms:.1f} ms")

            # Clean up
            del model

        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = {'error': str(e)}

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results if path provided
    if output_path:
        output = {
            'model_name': model_name,
            'prompt': prompt,
            'max_new_tokens': max_new_tokens,
            'num_runs': num_runs,
            'results': results
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def format_benchmark_table(
    results: Dict[str, BenchmarkResult],
    baseline: str = None
) -> str:
    """
    Format benchmark results as a markdown table.

    Args:
        results: Dict of model name to BenchmarkResult
        baseline: Name of baseline model for speedup calculation

    Returns:
        Formatted markdown table string
    """
    if baseline is None:
        baseline = list(results.keys())[0]

    baseline_speed = results[baseline].tokens_per_second

    lines = [
        "| Model | Tokens/sec | Memory (GB) | Latency (ms) | Speedup |",
        "|-------|------------|-------------|--------------|---------|"
    ]

    for name, result in results.items():
        speedup = result.tokens_per_second / baseline_speed
        speedup_str = f"{speedup:.2f}x" if name != baseline else "1.00x"
        lines.append(
            f"| {name} | {result.tokens_per_second:.1f} | "
            f"{result.memory_gb:.2f} | {result.total_latency_ms:.1f} | {speedup_str} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print("Benchmark Utils Demo")
    print("=" * 50)

    # Demo with a small model
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\nLoading GPT-2 for demo...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("\nRunning benchmark...")
        result = benchmark_inference(
            model=model,
            tokenizer=tokenizer,
            prompt="The quick brown fox",
            max_new_tokens=50,
            num_runs=3,
            name="gpt2-fp16"
        )

        print(f"\nResults:")
        print(f"  Name: {result.name}")
        print(f"  Tokens/sec: {result.tokens_per_second:.1f}")
        print(f"  Prefill latency: {result.prefill_latency_ms:.1f} ms")
        print(f"  Total latency: {result.total_latency_ms:.1f} ms")
        print(f"  Memory: {result.memory_gb:.2f} GB")

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Demo skipped: {e}")
