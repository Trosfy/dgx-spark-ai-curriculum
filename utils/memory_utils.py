#!/usr/bin/env python3
"""
Memory Management Utilities for DGX Spark
Tools for monitoring and optimizing memory usage on unified memory architecture
"""

import torch
import gc
import subprocess
import time
from typing import Optional, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps


@dataclass
class MemorySnapshot:
    """Snapshot of memory state"""
    timestamp: float
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_free_gb: float
    system_total_gb: float
    system_available_gb: float
    system_buffers_gb: float


def get_memory_snapshot() -> MemorySnapshot:
    """Get current memory state across GPU and system"""
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_free = gpu_total - gpu_reserved
    else:
        gpu_allocated = gpu_reserved = gpu_free = 0
    
    # System memory
    mem_result = subprocess.run(["free", "-b"], capture_output=True, text=True)
    mem_lines = mem_result.stdout.split("\n")
    mem_parts = mem_lines[1].split()
    
    system_total = float(mem_parts[1]) / 1e9
    system_available = float(mem_parts[6]) / 1e9
    system_buffers = float(mem_parts[5]) / 1e9
    
    return MemorySnapshot(
        timestamp=time.time(),
        gpu_allocated_gb=gpu_allocated,
        gpu_reserved_gb=gpu_reserved,
        gpu_free_gb=gpu_free,
        system_total_gb=system_total,
        system_available_gb=system_available,
        system_buffers_gb=system_buffers
    )


def print_memory_status(label: str = "Current"):
    """Print formatted memory status"""
    snapshot = get_memory_snapshot()
    
    print(f"\n{'─' * 50}")
    print(f"Memory Status: {label}")
    print(f"{'─' * 50}")
    print(f"GPU Allocated:     {snapshot.gpu_allocated_gb:>8.2f} GB")
    print(f"GPU Reserved:      {snapshot.gpu_reserved_gb:>8.2f} GB")
    print(f"GPU Free:          {snapshot.gpu_free_gb:>8.2f} GB")
    print(f"System Total:      {snapshot.system_total_gb:>8.2f} GB")
    print(f"System Available:  {snapshot.system_available_gb:>8.2f} GB")
    print(f"System Buffers:    {snapshot.system_buffers_gb:>8.2f} GB")
    print(f"{'─' * 50}\n")


def clear_all_memory(clear_buffer_cache: bool = True):
    """
    Aggressively clear all memory.
    
    Args:
        clear_buffer_cache: Whether to clear Linux buffer cache (requires sudo)
    """
    # Python garbage collection
    gc.collect()
    
    # PyTorch GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Linux buffer cache
    if clear_buffer_cache:
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError:
            print("Warning: Could not clear buffer cache (needs sudo)")
    
    gc.collect()
    print("✓ Memory cleared")


@contextmanager
def memory_tracked(label: str = "Operation"):
    """
    Context manager to track memory usage of a block.
    
    Usage:
        with memory_tracked("Model Loading"):
            model = load_model()
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    before = get_memory_snapshot()
    start_time = time.time()
    
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        after = get_memory_snapshot()
        elapsed = time.time() - start_time
        
        gpu_delta = after.gpu_allocated_gb - before.gpu_allocated_gb
        system_delta = before.system_available_gb - after.system_available_gb
        
        peak_gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        print(f"\n{'═' * 50}")
        print(f"Memory Report: {label}")
        print(f"{'═' * 50}")
        print(f"Duration:          {elapsed:>8.2f} s")
        print(f"GPU Delta:         {gpu_delta:>+8.2f} GB")
        print(f"GPU Peak:          {peak_gpu:>8.2f} GB")
        print(f"System Delta:      {system_delta:>+8.2f} GB")
        print(f"{'═' * 50}\n")


def memory_tracker(func: Callable) -> Callable:
    """
    Decorator to track memory usage of a function.
    
    Usage:
        @memory_tracker
        def load_model():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with memory_tracked(func.__name__):
            return func(*args, **kwargs)
    return wrapper


class MemoryMonitor:
    """
    Continuous memory monitor for long-running operations.
    
    Usage:
        monitor = MemoryMonitor(interval=1.0)
        monitor.start()
        # ... operations ...
        monitor.stop()
        monitor.print_summary()
    """
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots = []
        self._running = False
        self._thread = None
    
    def _monitor_loop(self):
        while self._running:
            self.snapshots.append(get_memory_snapshot())
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring in background thread"""
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"Memory monitor started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"Memory monitor stopped ({len(self.snapshots)} snapshots)")
    
    def print_summary(self):
        """Print summary of monitored memory usage"""
        if not self.snapshots:
            print("No snapshots collected")
            return
        
        gpu_values = [s.gpu_allocated_gb for s in self.snapshots]
        system_values = [s.system_available_gb for s in self.snapshots]
        
        print(f"\n{'═' * 50}")
        print("Memory Monitor Summary")
        print(f"{'═' * 50}")
        print(f"Duration:          {self.snapshots[-1].timestamp - self.snapshots[0].timestamp:.1f} s")
        print(f"Snapshots:         {len(self.snapshots)}")
        print(f"GPU Min:           {min(gpu_values):.2f} GB")
        print(f"GPU Max:           {max(gpu_values):.2f} GB")
        print(f"GPU Avg:           {sum(gpu_values)/len(gpu_values):.2f} GB")
        print(f"System Avail Min:  {min(system_values):.2f} GB")
        print(f"System Avail Max:  {max(system_values):.2f} GB")
        print(f"{'═' * 50}\n")
    
    def get_peak_gpu(self) -> float:
        """Get peak GPU memory usage"""
        if not self.snapshots:
            return 0.0
        return max(s.gpu_allocated_gb for s in self.snapshots)


def estimate_model_memory(
    num_params_billions: float,
    dtype: str = "bf16",
    include_optimizer: bool = False,
    include_gradients: bool = False
) -> float:
    """
    Estimate memory required for a model.
    
    Args:
        num_params_billions: Number of parameters in billions
        dtype: Data type (fp32, fp16, bf16, int8, int4)
        include_optimizer: Include Adam optimizer states
        include_gradients: Include gradient storage
        
    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5,
        "fp8": 1,
        "fp4": 0.5
    }
    
    param_bytes = bytes_per_param.get(dtype, 2)
    base_memory = num_params_billions * param_bytes
    
    total = base_memory
    
    if include_gradients:
        # Gradients are typically in fp32 or same as params
        total += num_params_billions * 4  # fp32 gradients
    
    if include_optimizer:
        # Adam has 2 states per parameter (momentum, variance)
        total += num_params_billions * 4 * 2  # 2 fp32 states
    
    return total


def can_fit_model(
    num_params_billions: float,
    dtype: str = "bf16",
    training: bool = False,
    safety_margin_gb: float = 10.0
) -> tuple[bool, str]:
    """
    Check if a model can fit in DGX Spark memory.
    
    Args:
        num_params_billions: Model size in billions
        dtype: Data type
        training: Whether this is for training (needs gradients + optimizer)
        safety_margin_gb: GB to reserve for other operations
        
    Returns:
        (can_fit, explanation)
    """
    available_gb = 128 - safety_margin_gb  # DGX Spark has 128GB
    
    required = estimate_model_memory(
        num_params_billions,
        dtype,
        include_optimizer=training,
        include_gradients=training
    )
    
    can_fit = required <= available_gb
    
    explanation = f"{num_params_billions}B model in {dtype}: {required:.1f}GB required, {available_gb:.1f}GB available"
    
    if not can_fit:
        # Suggest alternatives
        if dtype in ["fp32", "bf16", "fp16"]:
            int4_required = estimate_model_memory(num_params_billions, "int4", training, training)
            explanation += f"\n  → Try int4 quantization: {int4_required:.1f}GB"
    
    return can_fit, explanation


if __name__ == "__main__":
    # Demo
    print_memory_status("Startup")
    
    print("\nModel Memory Estimates:")
    for size in [7, 13, 34, 70, 120]:
        for dtype in ["bf16", "int8", "int4"]:
            mem = estimate_model_memory(size, dtype)
            print(f"  {size}B {dtype}: {mem:.1f} GB")
    
    print("\nCan fit on DGX Spark?")
    for size in [8, 13, 34, 70, 120]:
        can_fit, explanation = can_fit_model(size, "bf16", training=False)
        status = "✓" if can_fit else "✗"
        print(f"  {status} {explanation}")
