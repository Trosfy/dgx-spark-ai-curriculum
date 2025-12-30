#!/usr/bin/env python3
"""
Capstone Project Utilities

Common utilities for all capstone project options including:
- Memory monitoring
- Performance profiling
- Model loading helpers
- Evaluation utilities
- Report generation

Usage:
    from capstone_utils import MemoryMonitor, profile_function, load_model_4bit
"""

import torch
import gc
import time
import json
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from functools import wraps
from contextlib import contextmanager

# ==============================================================================
# Memory Monitoring
# ==============================================================================

class MemoryMonitor:
    """
    Monitor GPU and system memory usage.

    Example:
        monitor = MemoryMonitor()
        monitor.snapshot("before_model_load")
        model = load_model()
        monitor.snapshot("after_model_load")
        monitor.report()
    """

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "total": 0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    def get_system_memory(self) -> Dict[str, float]:
        """Get current system memory usage in GB."""
        mem = psutil.virtual_memory()
        return {
            "used": mem.used / 1e9,
            "available": mem.available / 1e9,
            "total": mem.total / 1e9,
            "percent": mem.percent,
        }

    def snapshot(self, name: str = ""):
        """Take a memory snapshot."""
        self.snapshots.append({
            "name": name or f"snapshot_{len(self.snapshots)}",
            "timestamp": datetime.now().isoformat(),
            "gpu": self.get_gpu_memory(),
            "system": self.get_system_memory(),
        })

    def report(self) -> str:
        """Generate a memory usage report."""
        lines = [
            "\n" + "=" * 60,
            "MEMORY USAGE REPORT",
            "=" * 60,
        ]

        for snap in self.snapshots:
            lines.append(f"\n📍 {snap['name']}")
            lines.append(f"   GPU: {snap['gpu']['allocated']:.2f} GB allocated, "
                        f"{snap['gpu']['reserved']:.2f} GB reserved")
            lines.append(f"   System: {snap['system']['used']:.1f} GB used "
                        f"({snap['system']['percent']:.1f}%)")

        # Show changes between snapshots
        if len(self.snapshots) > 1:
            lines.append("\n📊 Changes:")
            for i in range(1, len(self.snapshots)):
                prev = self.snapshots[i-1]
                curr = self.snapshots[i]
                gpu_diff = curr['gpu']['allocated'] - prev['gpu']['allocated']
                sys_diff = curr['system']['used'] - prev['system']['used']
                lines.append(f"   {prev['name']} → {curr['name']}: "
                           f"GPU {gpu_diff:+.2f} GB, System {sys_diff:+.1f} GB")

        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)
        return report

    def to_json(self) -> str:
        """Export snapshots as JSON."""
        return json.dumps(self.snapshots, indent=2)


@contextmanager
def memory_tracked(name: str = "operation"):
    """
    Context manager for tracking memory during an operation.

    Example:
        with memory_tracked("model_loading"):
            model = load_model()
    """
    monitor = MemoryMonitor()
    monitor.snapshot(f"{name}_start")

    try:
        yield monitor
    finally:
        monitor.snapshot(f"{name}_end")
        monitor.report()


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("✅ GPU memory cleared")


# ==============================================================================
# Performance Profiling
# ==============================================================================

def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time and memory.

    Example:
        @profile_function
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        gpu_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Memory after
        gpu_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        print(f"\n⏱️ {func.__name__}:")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   GPU Memory: {gpu_before:.2f} → {gpu_after:.2f} GB "
              f"(Δ{gpu_after - gpu_before:+.2f} GB)")

        return result
    return wrapper


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float  # iterations per second


def benchmark(
    func: Callable,
    iterations: int = 10,
    warmup: int = 2,
    name: str = None
) -> BenchmarkResult:
    """
    Benchmark a function.

    Args:
        func: Function to benchmark (must be callable with no args)
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)
        name: Name for the benchmark

    Returns:
        BenchmarkResult with timing statistics
    """
    name = name or func.__name__

    # Warmup
    print(f"🔄 Benchmarking {name}: {warmup} warmup + {iterations} iterations...")
    for _ in range(warmup):
        func()

    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    # Calculate statistics
    import statistics
    result = BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=sum(times),
        mean_time=statistics.mean(times),
        std_time=statistics.stdev(times) if len(times) > 1 else 0,
        min_time=min(times),
        max_time=max(times),
        throughput=iterations / sum(times),
    )

    print(f"   Mean: {result.mean_time*1000:.2f}ms ± {result.std_time*1000:.2f}ms")
    print(f"   Range: {result.min_time*1000:.2f}ms - {result.max_time*1000:.2f}ms")
    print(f"   Throughput: {result.throughput:.2f} iter/s")

    return result


# ==============================================================================
# Model Loading Helpers
# ==============================================================================

def load_model_4bit(
    model_name: str,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a model in 4-bit quantization.

    Args:
        model_name: Hugging Face model name or path
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"📥 Loading {model_name} in 4-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Report memory
    mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"✅ Model loaded. GPU memory: {mem_gb:.2f} GB")

    return model, tokenizer


def estimate_model_memory(
    model_name: str,
    precision: str = "int4"
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.

    Args:
        model_name: Model identifier
        precision: One of "fp32", "fp16", "bf16", "int8", "int4"

    Returns:
        Dict with estimated memory in GB
    """
    # Common model sizes (approximate parameters in billions)
    MODEL_SIZES = {
        "llama-7b": 7,
        "llama-8b": 8,
        "llama-13b": 13,
        "llama-70b": 70,
        "mistral-7b": 7,
        "qwen-7b": 7,
        "qwen-72b": 72,
    }

    # Try to find model size
    params_b = None
    model_lower = model_name.lower()

    for key, size in MODEL_SIZES.items():
        if key in model_lower:
            params_b = size
            break

    if params_b is None:
        # Try to extract number from name
        import re
        match = re.search(r'(\d+)[bB]', model_name)
        if match:
            params_b = int(match.group(1))
        else:
            return {"error": "Could not determine model size"}

    # Calculate memory based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
    }

    bpp = bytes_per_param.get(precision, 2)
    model_memory = (params_b * 1e9 * bpp) / 1e9  # GB

    # Add overhead for activations, KV cache, etc.
    overhead_factor = 1.2 if precision in ["int4", "int8"] else 1.5

    return {
        "model_memory_gb": model_memory,
        "estimated_total_gb": model_memory * overhead_factor,
        "params_billions": params_b,
        "precision": precision,
        "fits_dgx_spark": model_memory * overhead_factor < 120,
    }


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_experiment_report(
    experiment_name: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    output_path: str = None,
) -> str:
    """
    Generate a markdown experiment report.

    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        results: Experiment results
        output_path: Optional path to save the report

    Returns:
        Markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Experiment Report: {experiment_name}

Generated: {timestamp}

## Configuration

```json
{json.dumps(config, indent=2, default=str)}
```

## Results

"""

    # Add results as table
    if isinstance(results, dict):
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        for key, value in results.items():
            if isinstance(value, float):
                report += f"| {key} | {value:.4f} |\n"
            else:
                report += f"| {key} | {value} |\n"

    report += """
## Environment

| Property | Value |
|----------|-------|
"""
    report += f"| GPU | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'} |\n"
    report += f"| PyTorch | {torch.__version__} |\n"
    report += f"| CUDA | {torch.version.cuda if torch.cuda.is_available() else 'N/A'} |\n"

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"✅ Report saved to {output_path}")

    return report


# ==============================================================================
# Utility Functions
# ==============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"✅ Random seed set to {seed}")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": 100 * trainable / total if total > 0 else 0,
    }


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("Capstone Utilities Module")
    print("=" * 60)
    print("\nAvailable utilities:")
    print("  • MemoryMonitor - Track GPU/system memory")
    print("  • profile_function - Decorator for profiling")
    print("  • benchmark - Benchmark a function")
    print("  • load_model_4bit - Load models in 4-bit")
    print("  • estimate_model_memory - Estimate memory needs")
    print("  • generate_experiment_report - Create markdown reports")
    print("  • set_seed - Set random seeds")
    print("  • count_parameters - Count model parameters")
