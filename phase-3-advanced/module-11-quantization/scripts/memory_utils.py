"""
Memory Utilities for DGX Spark

This module provides memory monitoring and management utilities
optimized for DGX Spark's 128GB unified memory architecture.

Example:
    >>> from memory_utils import get_gpu_memory, clear_memory, MemoryTracker
    >>>
    >>> # Check current memory
    >>> allocated, reserved = get_gpu_memory()
    >>> print(f"Memory: {allocated:.2f} GB allocated")
    >>>
    >>> # Track memory during operations
    >>> with MemoryTracker() as tracker:
    ...     model = load_model()
    >>> print(f"Peak memory: {tracker.peak_mb:.1f} MB")
"""

import torch
import gc
import subprocess
import time
from typing import Tuple, Optional, Dict, List
from contextlib import contextmanager


def get_gpu_memory() -> Tuple[float, float]:
    """
    Get current GPU memory usage.

    Returns:
        Tuple of (allocated_gb, reserved_gb)

    Example:
        >>> allocated, reserved = get_gpu_memory()
        >>> print(f"Using {allocated:.2f} GB of {reserved:.2f} GB reserved")
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    return allocated, reserved


def get_gpu_memory_detailed() -> Dict[str, float]:
    """
    Get detailed GPU memory statistics.

    Returns:
        Dictionary with memory statistics in GB

    Example:
        >>> stats = get_gpu_memory_detailed()
        >>> print(f"Peak memory: {stats['peak_allocated_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'peak_allocated_gb': 0.0,
            'peak_reserved_gb': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0,
        }

    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        'peak_reserved_gb': torch.cuda.max_memory_reserved() / 1e9,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated()) / 1e9,
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def clear_memory(verbose: bool = False):
    """
    Clear GPU memory cache.

    Args:
        verbose: If True, print memory before and after

    Example:
        >>> clear_memory(verbose=True)
        Before: 5.23 GB -> After: 0.12 GB (freed 5.11 GB)
    """
    if verbose:
        before = get_gpu_memory()[0]

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if verbose:
        after = get_gpu_memory()[0]
        print(f"Memory cleared: {before:.2f} GB -> {after:.2f} GB "
              f"(freed {before - after:.2f} GB)")


def clear_system_cache():
    """
    Clear system page cache (requires sudo).

    Useful before loading large models to ensure clean memory state.
    This is important on DGX Spark to maximize available unified memory.

    Example:
        >>> clear_system_cache()  # Requires sudo
    """
    try:
        subprocess.run(
            "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
            shell=True,
            check=True,
            capture_output=True
        )
        print("System cache cleared")
    except subprocess.CalledProcessError as e:
        print(f"Could not clear system cache (may need sudo): {e}")


def estimate_model_memory(
    num_params: int,
    dtype: torch.dtype = torch.float16,
    include_activations: bool = True,
    batch_size: int = 1,
    sequence_length: int = 2048
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.

    Args:
        num_params: Number of model parameters
        dtype: Data type for weights
        include_activations: Include activation memory estimate
        batch_size: Batch size for inference
        sequence_length: Sequence length for inference

    Returns:
        Dictionary with memory estimates in GB

    Example:
        >>> # Llama-7B
        >>> mem = estimate_model_memory(7_000_000_000, torch.float16)
        >>> print(f"Weights: {mem['weights_gb']:.1f} GB")
    """
    # Bytes per element
    bytes_per_elem = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 2)

    weights_gb = num_params * bytes_per_elem / 1e9

    # Rough activation estimate (varies by architecture)
    # Assuming transformer with attention
    if include_activations:
        # Activations scale with batch_size * seq_len * hidden_dim
        # Rough estimate: 2x weights for activations during inference
        activations_gb = weights_gb * 0.5 * batch_size
    else:
        activations_gb = 0

    # KV cache estimate (for transformers)
    # These defaults are for Llama-7B architecture. For accurate estimates with
    # other models, you should pass the actual model config values:
    #   num_layers = model.config.num_hidden_layers
    #   num_heads = model.config.num_attention_heads
    #   head_dim = model.config.hidden_size // model.config.num_attention_heads
    #
    # Common architectures:
    #   Llama-7B:  num_layers=32, num_heads=32, head_dim=128 (hidden=4096)
    #   Llama-13B: num_layers=40, num_heads=40, head_dim=128 (hidden=5120)
    #   Llama-70B: num_layers=80, num_heads=64, head_dim=128 (hidden=8192)
    #   GPT-2:     num_layers=12, num_heads=12, head_dim=64  (hidden=768)
    num_layers = 32  # Default: Llama-7B
    num_heads = 32   # Default: Llama-7B
    head_dim = 128   # Default: Llama-7B (4096 / 32)
    kv_cache_gb = (
        batch_size * sequence_length * num_layers * num_heads * head_dim * 2 *  # key + value
        bytes_per_elem / 1e9
    )

    return {
        'weights_gb': weights_gb,
        'activations_gb': activations_gb,
        'kv_cache_gb': kv_cache_gb,
        'total_gb': weights_gb + activations_gb + kv_cache_gb,
    }


class MemoryTracker:
    """
    Context manager for tracking memory usage during operations.

    Example:
        >>> with MemoryTracker() as tracker:
        ...     model = AutoModelForCausalLM.from_pretrained("gpt2")
        ...     output = model.generate(...)
        >>> print(f"Peak: {tracker.peak_mb:.1f} MB")
        >>> print(f"Delta: {tracker.delta_mb:.1f} MB")
    """

    def __init__(self, device: int = 0):
        """
        Initialize memory tracker.

        Args:
            device: CUDA device index to track
        """
        self.device = device
        self.start_mb = 0
        self.end_mb = 0
        self.peak_mb = 0
        self.samples: List[Tuple[float, float]] = []
        self._monitoring = False

    def __enter__(self):
        """Start tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_mb = torch.cuda.memory_allocated(self.device) / 1e6
        return self

    def __exit__(self, *args):
        """Stop tracking and record final stats."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_mb = torch.cuda.memory_allocated(self.device) / 1e6
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6

    @property
    def delta_mb(self) -> float:
        """Memory change from start to end."""
        return self.end_mb - self.start_mb

    def snapshot(self) -> float:
        """Take a memory snapshot (returns MB)."""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated(self.device) / 1e6
            self.samples.append((time.time(), current))
            return current
        return 0.0

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            'start_mb': self.start_mb,
            'end_mb': self.end_mb,
            'peak_mb': self.peak_mb,
            'delta_mb': self.delta_mb,
            'num_samples': len(self.samples),
        }


@contextmanager
def monitor_memory(name: str = "operation"):
    """
    Simple context manager for monitoring memory with printout.

    Args:
        name: Name of operation being monitored

    Example:
        >>> with monitor_memory("Model Loading"):
        ...     model = load_model()
        Model Loading: 1234.5 MB allocated, 2345.6 MB peak
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated() / 1e6

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end = torch.cuda.memory_allocated() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"{name}: {end - start:.1f} MB allocated, {peak:.1f} MB peak")


def print_memory_summary():
    """Print a formatted memory summary."""
    stats = get_gpu_memory_detailed()

    print("\n" + "=" * 50)
    print("GPU Memory Summary")
    print("=" * 50)
    print(f"Allocated:      {stats['allocated_gb']:>8.2f} GB")
    print(f"Reserved:       {stats['reserved_gb']:>8.2f} GB")
    print(f"Peak Allocated: {stats['peak_allocated_gb']:>8.2f} GB")
    print(f"Free:           {stats['free_gb']:>8.2f} GB")
    print(f"Total:          {stats['total_gb']:>8.2f} GB")
    print("=" * 50)

    # DGX Spark specific note
    if stats['total_gb'] > 100:
        print("\n💡 DGX Spark detected! You have 128GB unified memory.")
        print(f"   Available for models: ~{stats['free_gb']:.0f} GB")


if __name__ == "__main__":
    print("Memory Utils Demo")
    print("=" * 50)

    # Current memory
    allocated, reserved = get_gpu_memory()
    print(f"\nCurrent memory: {allocated:.2f} GB allocated")

    # Estimate model memory
    print("\nMemory estimates for common models:")
    for name, params in [
        ("GPT-2", 124_000_000),
        ("Llama-7B", 7_000_000_000),
        ("Llama-13B", 13_000_000_000),
        ("Llama-70B", 70_000_000_000),
    ]:
        mem = estimate_model_memory(params, torch.float16)
        print(f"  {name}: {mem['total_gb']:.1f} GB (weights: {mem['weights_gb']:.1f} GB)")

    # Memory tracking demo
    print("\nMemory tracking demo:")
    with MemoryTracker() as tracker:
        # Allocate some tensors
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = x @ y
            del x, y, z

    print(f"  Peak: {tracker.peak_mb:.1f} MB")
    print(f"  Delta: {tracker.delta_mb:.1f} MB")
