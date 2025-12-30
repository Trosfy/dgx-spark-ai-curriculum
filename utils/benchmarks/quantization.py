"""
Quantization Benchmarking Utilities

Specialized benchmarking for quantized models, measuring compression ratio,
quality retention (perplexity), and inference speed on DGX Spark.
"""

import time
import gc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .base import BenchmarkResult, get_gpu_memory_gb


@dataclass
class QuantizationBenchmarkResult:
    """
    Result container for quantization benchmarks.

    Captures metrics specific to quantized model evaluation including
    compression ratio, quality retention, and speed comparisons.

    Attributes:
        model_name: Name/path of the model
        quantization_type: Type of quantization (Q4_K_M, Q8_0, etc.)
        model_size_mb: Model size in megabytes
        perplexity: Perplexity score (lower is better)
        tokens_per_second: Generation speed
        prefill_tokens_per_second: Prompt processing speed
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
        if self.model_size_mb > 0:
            return baseline_size_mb / self.model_size_mb
        return 0.0

    def quality_retained(self, baseline_ppl: float) -> Optional[float]:
        """
        Calculate quality retention percentage.

        Lower perplexity is better, so we use baseline/current * 100.
        Returns percentage of quality retained.
        """
        if self.perplexity is None or baseline_ppl is None or self.perplexity == 0:
            return None
        return baseline_ppl / self.perplexity * 100

    def speedup_vs(self, baseline: "QuantizationBenchmarkResult") -> Optional[float]:
        """Calculate speedup compared to baseline."""
        if self.tokens_per_second and baseline.tokens_per_second:
            return self.tokens_per_second / baseline.tokens_per_second
        return None

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


def benchmark_quantized_model(
    model,
    tokenizer,
    model_name: str = "model",
    quantization_type: str = "unknown",
    prompt: str = "The future of artificial intelligence is",
    num_tokens: int = 50,
    num_warmup: int = 2,
    num_runs: int = 5,
    batch_size: int = 1
) -> QuantizationBenchmarkResult:
    """
    Benchmark a quantized model's inference speed.

    Args:
        model: The language model to benchmark
        tokenizer: The tokenizer
        model_name: Name of the model
        quantization_type: Type of quantization applied
        prompt: Input prompt for generation
        num_tokens: Number of tokens to generate
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        batch_size: Batch size for inference

    Returns:
        QuantizationBenchmarkResult with timing information
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    device = next(model.parameters()).device

    # Prepare input
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True
    ).to(device)

    prompt_len = inputs.input_ids.shape[1]

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    prefill_times = []
    decode_times = []
    total_times = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.perf_counter() - start
        total_times.append(total_time)

        # Estimate prefill vs decode (approximate)
        generated_len = outputs.sequences.shape[1] - prompt_len
        prefill_time = total_time * (prompt_len / (prompt_len + generated_len))
        decode_time = total_time - prefill_time

        prefill_times.append(prefill_time)
        decode_times.append(decode_time)

    # Calculate metrics
    import statistics
    avg_total = statistics.mean(total_times)
    avg_prefill = statistics.mean(prefill_times)
    avg_decode = statistics.mean(decode_times)

    tokens_per_second = num_tokens / avg_decode if avg_decode > 0 else 0
    prefill_tps = prompt_len / avg_prefill if avg_prefill > 0 else 0

    # Memory usage
    memory_gb = 0
    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / 1e9

    # Model size (approximate)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return QuantizationBenchmarkResult(
        model_name=model_name,
        quantization_type=quantization_type,
        model_size_mb=model_size_mb,
        tokens_per_second=tokens_per_second,
        prefill_tokens_per_second=prefill_tps,
        memory_used_gb=memory_gb,
        latency_ms=avg_total * 1000,
    )


def compare_quantizations(
    results: List[QuantizationBenchmarkResult],
    baseline_name: str = "FP16"
) -> None:
    """
    Print comparison table for quantization benchmarks.

    Args:
        results: List of benchmark results
        baseline_name: Name of baseline quantization for comparison
    """
    # Find baseline
    baseline = next(
        (r for r in results if r.quantization_type.upper() == baseline_name.upper()),
        results[0] if results else None
    )

    print("\n" + "=" * 90)
    print("Quantization Comparison")
    print("=" * 90)
    print(f"{'Quantization':<15} {'Size (MB)':<12} {'Compression':<12} {'Tokens/s':<12} {'Speedup':<10} {'Memory (GB)':<12}")
    print("-" * 90)

    for r in results:
        compression = r.compression_ratio(baseline.model_size_mb) if baseline else 0
        speedup = r.speedup_vs(baseline) if baseline else 0

        print(
            f"{r.quantization_type:<15} "
            f"{r.model_size_mb:<12.1f} "
            f"{compression:<12.2f}x "
            f"{r.tokens_per_second or 0:<12.1f} "
            f"{speedup or 0:<10.2f}x "
            f"{r.memory_used_gb or 0:<12.2f}"
        )

    print("=" * 90)

    # Print quality comparison if perplexity available
    if any(r.perplexity for r in results):
        print("\nQuality Comparison (Perplexity - lower is better):")
        print("-" * 50)
        baseline_ppl = baseline.perplexity if baseline and baseline.perplexity else None

        for r in results:
            if r.perplexity:
                retained = r.quality_retained(baseline_ppl) if baseline_ppl else None
                retained_str = f"({retained:.1f}% retained)" if retained else ""
                print(f"  {r.quantization_type}: {r.perplexity:.2f} {retained_str}")
