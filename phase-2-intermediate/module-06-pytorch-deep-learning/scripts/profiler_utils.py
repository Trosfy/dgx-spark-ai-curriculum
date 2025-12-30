"""
Profiling Utilities - Performance Analysis Tools

This module provides utilities for profiling PyTorch training loops
and identifying performance bottlenecks.

Components:
    - profile_training_step: Profile a single training step
    - benchmark_dataloader: Benchmark data loading performance
    - MemoryTracker: Track GPU memory usage
    - generate_profile_report: Create a comprehensive profiling report

Example:
    >>> from profiler_utils import profile_training_step, MemoryTracker
    >>> timings = profile_training_step(model, batch, criterion, optimizer)
    >>> print(f"Forward: {timings['forward']:.2f}ms")

Author: DGX Spark AI Curriculum
"""

__all__ = [
    'ProfilingResult',
    'Timer',
    'MemoryTracker',
    'profile_training_step',
    'profile_training_loop',
    'benchmark_dataloader',
    'profile_with_pytorch_profiler',
    'find_bottlenecks',
    'generate_profile_report',
]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, schedule
from pathlib import Path
import time
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class ProfilingResult:
    """Container for profiling results."""
    total_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    optimizer_time_ms: float = 0.0
    data_loading_time_ms: float = 0.0
    memory_allocated_gb: float = 0.0
    memory_peak_gb: float = 0.0


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = (time.perf_counter() - self.start_time) * 1000  # ms


class MemoryTracker:
    """
    Track GPU memory usage during training.

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.reset()
        >>> # ... training code ...
        >>> print(tracker.get_stats())
    """

    def __init__(self):
        self.snapshots = []

    def reset(self):
        """Reset memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.snapshots = []

    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if torch.cuda.is_available():
            self.snapshots.append({
                'label': label,
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'peak': torch.cuda.max_memory_allocated() / 1e9,
            })

    def get_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'peak': 0}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'peak_gb': torch.cuda.max_memory_allocated() / 1e9,
        }

    def print_stats(self):
        """Print current memory statistics."""
        stats = self.get_stats()
        print(f"GPU Memory: Allocated={stats['allocated_gb']:.2f}GB, "
              f"Reserved={stats['reserved_gb']:.2f}GB, "
              f"Peak={stats['peak_gb']:.2f}GB")


def profile_training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """
    Profile a single training step with detailed timing.

    Args:
        model: Neural network model
        inputs: Input batch
        labels: Target labels
        criterion: Loss function
        optimizer: Optimizer
        num_iterations: Number of iterations to average

    Returns:
        Dict with timing in milliseconds for each phase

    Example:
        >>> timings = profile_training_step(model, x, y, criterion, optimizer)
        >>> print(f"Forward: {timings['forward']:.2f}ms")
        >>> print(f"Backward: {timings['backward']:.2f}ms")
    """
    model.train()
    timings = {'forward': [], 'backward': [], 'optimizer': [], 'total': []}

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Profile
    for _ in range(num_iterations):
        total_timer = Timer()
        forward_timer = Timer()
        backward_timer = Timer()
        optimizer_timer = Timer()

        with total_timer:
            # Forward
            with forward_timer:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward
            with backward_timer:
                optimizer.zero_grad()
                loss.backward()

            # Optimizer
            with optimizer_timer:
                optimizer.step()

        timings['forward'].append(forward_timer.elapsed)
        timings['backward'].append(backward_timer.elapsed)
        timings['optimizer'].append(optimizer_timer.elapsed)
        timings['total'].append(total_timer.elapsed)

    # Average
    return {k: sum(v) / len(v) for k, v in timings.items()}


def benchmark_dataloader(
    dataloader: DataLoader,
    num_batches: int = 50,
    warmup_batches: int = 5,
    to_device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Benchmark DataLoader performance.

    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to test
        warmup_batches: Number of batches for warmup (default: 5)
        to_device: Optional device to transfer data to

    Returns:
        Dict with timing statistics

    Example:
        >>> results = benchmark_dataloader(train_loader, num_batches=100)
        >>> print(f"Throughput: {results['throughput']:.0f} samples/sec")
    """
    timings = []
    total_samples = 0

    # Warmup
    data_iter = iter(dataloader)
    for _ in range(min(warmup_batches, len(dataloader))):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

    # Benchmark
    data_iter = iter(dataloader)
    for i in range(num_batches):
        with Timer() as t:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if to_device is not None:
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(to_device) if hasattr(b, 'to') else b for b in batch]
                else:
                    batch = batch.to(to_device)

        timings.append(t.elapsed)
        if isinstance(batch, (list, tuple)):
            total_samples += batch[0].size(0)
        else:
            total_samples += batch.size(0)

    avg_time = sum(timings) / len(timings)
    total_time = sum(timings) / 1000  # seconds

    return {
        'avg_batch_time_ms': avg_time,
        'total_time_s': total_time,
        'throughput': total_samples / total_time,
        'batches': num_batches,
        'samples': total_samples,
    }


def profile_with_pytorch_profiler(
    training_fn: Callable,
    num_steps: int = 20,
    warmup_steps: int = 5,
    output_dir: Optional[str] = None,
) -> Tuple[Any, str]:
    """
    Profile training using PyTorch Profiler.

    Args:
        training_fn: Function that performs one training step
        num_steps: Total number of steps to profile (after warmup)
        warmup_steps: Steps for warmup before profiling starts
        output_dir: Optional directory to save trace

    Returns:
        Tuple of (profiler, summary_table)

    Example:
        >>> def train_step():
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(x), y)
        ...     loss.backward()
        ...     optimizer.step()
        >>> prof, summary = profile_with_pytorch_profiler(train_step)
        >>> print(summary)
    """
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Use schedule to properly implement warmup
    # wait=0: Don't skip any steps at the start
    # warmup=warmup_steps: Run this many steps before recording
    # active=num_steps: Record this many steps
    # repeat=1: Only do this cycle once
    prof_schedule = schedule(
        wait=0,
        warmup=warmup_steps,
        active=num_steps,
        repeat=1
    )

    total_steps = warmup_steps + num_steps

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(total_steps):
            training_fn()
            prof.step()  # Signal profiler that one step is complete

    summary = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=15
    )

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        trace_path = output_path / 'trace.json'
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace saved to: {trace_path}")

    return prof, summary


def find_bottlenecks(
    timings: Dict[str, float],
    threshold: float = 0.3,
) -> Dict[str, str]:
    """
    Analyze timings to identify bottlenecks.

    Args:
        timings: Dict of timing measurements in ms
        threshold: Fraction of total time to consider a bottleneck

    Returns:
        Dict with bottleneck analysis and recommendations

    Example:
        >>> analysis = find_bottlenecks(timings)
        >>> print(analysis['bottleneck'])
        >>> print(analysis['recommendation'])
    """
    total = timings.get('total', sum(timings.values()))

    bottlenecks = []
    recommendations = []

    for phase, time_ms in timings.items():
        if phase == 'total':
            continue

        fraction = time_ms / total
        if fraction > threshold:
            bottlenecks.append(f"{phase} ({fraction:.1%} of total)")

            if phase == 'data_loading':
                recommendations.append(
                    "Increase num_workers, use pin_memory=True, or preload data"
                )
            elif phase == 'forward':
                recommendations.append(
                    "Consider model pruning, quantization, or using mixed precision"
                )
            elif phase == 'backward':
                recommendations.append(
                    "Enable gradient checkpointing or use mixed precision"
                )
            elif phase == 'optimizer':
                recommendations.append(
                    "Consider using fused optimizers or reducing model size"
                )

    return {
        'bottlenecks': bottlenecks,
        'recommendations': recommendations,
        'timing_breakdown': {
            k: f"{v:.2f}ms ({v/total:.1%})"
            for k, v in timings.items()
        }
    }


def generate_profile_report(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_batches: int = 20,
) -> str:
    """
    Generate a comprehensive profiling report.

    Args:
        model: Model to profile
        dataloader: DataLoader to use
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        num_batches: Number of batches to profile

    Returns:
        Formatted report string

    Example:
        >>> report = generate_profile_report(model, loader, criterion, optimizer, device)
        >>> print(report)
    """
    report_lines = ["=" * 60, "PROFILING REPORT", "=" * 60, ""]

    # Device info
    report_lines.append("DEVICE INFORMATION")
    report_lines.append("-" * 40)
    report_lines.append(f"Device: {device}")
    if torch.cuda.is_available():
        report_lines.append(f"GPU: {torch.cuda.get_device_name()}")
        report_lines.append(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    report_lines.append("")

    # DataLoader benchmark
    report_lines.append("DATA LOADING PERFORMANCE")
    report_lines.append("-" * 40)
    dl_results = benchmark_dataloader(dataloader, min(num_batches, 50), to_device=device)
    report_lines.append(f"Avg batch time: {dl_results['avg_batch_time_ms']:.2f} ms")
    report_lines.append(f"Throughput: {dl_results['throughput']:.0f} samples/sec")
    report_lines.append("")

    # Training step profiling
    report_lines.append("TRAINING STEP BREAKDOWN")
    report_lines.append("-" * 40)

    # Get a batch
    batch = next(iter(dataloader))
    inputs, labels = batch[0].to(device), batch[1].to(device)

    timings = profile_training_step(model, inputs, labels, criterion, optimizer)
    for phase, time_ms in timings.items():
        pct = time_ms / timings['total'] * 100
        report_lines.append(f"{phase:15s}: {time_ms:7.2f} ms ({pct:5.1f}%)")
    report_lines.append("")

    # Memory usage
    if torch.cuda.is_available():
        report_lines.append("MEMORY USAGE")
        report_lines.append("-" * 40)
        tracker = MemoryTracker()
        stats = tracker.get_stats()
        report_lines.append(f"Allocated: {stats['allocated_gb']:.2f} GB")
        report_lines.append(f"Reserved: {stats['reserved_gb']:.2f} GB")
        report_lines.append(f"Peak: {stats['peak_gb']:.2f} GB")
        report_lines.append("")

    # Bottleneck analysis
    report_lines.append("BOTTLENECK ANALYSIS")
    report_lines.append("-" * 40)
    analysis = find_bottlenecks(timings)

    if analysis['bottlenecks']:
        report_lines.append("Identified bottlenecks:")
        for b in analysis['bottlenecks']:
            report_lines.append(f"  - {b}")
        report_lines.append("")
        report_lines.append("Recommendations:")
        for r in analysis['recommendations']:
            report_lines.append(f"  - {r}")
    else:
        report_lines.append("No significant bottlenecks identified.")

    report_lines.append("")
    report_lines.append("=" * 60)

    return "\n".join(report_lines)


if __name__ == '__main__':
    print("Testing profiler utilities...")

    # Simple test
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = torch.randn(32, 100, device=device)
    y = torch.randint(0, 10, (32,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    print("\nProfiling training step...")
    timings = profile_training_step(model, x, y, criterion, optimizer)
    for phase, time_ms in timings.items():
        print(f"  {phase}: {time_ms:.2f}ms")

    print("\nBottleneck analysis...")
    analysis = find_bottlenecks(timings)
    print(f"  Bottlenecks: {analysis['bottlenecks']}")

    print("\nAll tests passed!")
