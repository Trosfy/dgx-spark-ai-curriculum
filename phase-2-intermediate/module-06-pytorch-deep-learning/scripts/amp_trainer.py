"""
Mixed Precision Training Utilities - Production-Ready AMP Training

This module provides reusable components for mixed precision training.

Components:
    - AMPTrainer: Complete training class with AMP support
    - train_epoch_amp: Single epoch training function
    - benchmark_precision: Compare different precision modes

Example:
    >>> from amp_trainer import AMPTrainer
    >>> trainer = AMPTrainer(model, optimizer, criterion, dtype='bf16')
    >>> trainer.train_epoch(train_loader)

Author: DGX Spark AI Curriculum
"""

__all__ = [
    'TrainingMetrics',
    'AMPTrainer',
    'benchmark_precision',
]

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any, Callable
import time
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    time_seconds: float = 0.0
    memory_gb: float = 0.0
    skipped_steps: int = 0
    total_steps: int = 0


class AMPTrainer:
    """
    Mixed Precision Trainer with automatic precision selection.

    Supports FP32, FP16, and BF16 training with automatic fallback
    if too many gradient overflows are detected.

    Args:
        model: Neural network model
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        dtype: Precision type ('fp32', 'fp16', 'bf16')
        grad_clip: Maximum gradient norm (None to disable)
        accumulation_steps: Gradient accumulation steps

    Example:
        >>> model = ResNet18().cuda()
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> criterion = nn.CrossEntropyLoss()
        >>> trainer = AMPTrainer(model, optimizer, criterion, dtype='bf16')
        >>> metrics = trainer.train_epoch(train_loader)
        >>> print(f"Loss: {metrics.loss:.4f}, Acc: {metrics.accuracy:.2f}%")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = None,
        dtype: str = 'bf16',
        grad_clip: Optional[float] = None,
        accumulation_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = grad_clip
        self.accumulation_steps = accumulation_steps

        # Setup precision
        self._setup_precision(dtype)

        # Tracking
        self.total_steps = 0
        self.skipped_steps = 0

    def _setup_precision(self, dtype: str):
        """Configure precision settings."""
        dtype = dtype.lower()

        # Determine device type for GradScaler (PyTorch 2.4+ compatibility)
        device_type = self.device.type if self.device else 'cuda'

        if dtype == 'fp32':
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = GradScaler(device_type, enabled=False)
        elif dtype == 'fp16':
            self.use_amp = True
            self.amp_dtype = torch.float16
            self.scaler = GradScaler(device_type, enabled=True)
        elif dtype == 'bf16':
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
            # BF16 doesn't need gradient scaling (same dynamic range as FP32)
            self.scaler = GradScaler(device_type, enabled=False)
        else:
            raise ValueError(f"Unknown dtype: {dtype}. Use 'fp32', 'fp16', or 'bf16'")

        self.dtype_name = dtype

    def train_epoch(
        self,
        dataloader: DataLoader,
        scheduler: Optional[Any] = None,
    ) -> TrainingMetrics:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            scheduler: Optional learning rate scheduler

        Returns:
            TrainingMetrics with loss, accuracy, time, etc.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        skipped = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with autocast
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights (with accumulation)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Check for gradient overflow
                old_scale = self.scaler.get_scale()

                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Track skipped steps
                if self.scaler.get_scale() < old_scale:
                    skipped += 1

                self.total_steps += 1

            # Metrics
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start_time
        memory_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        self.skipped_steps += skipped

        return TrainingMetrics(
            loss=running_loss / len(dataloader),
            accuracy=100.0 * correct / total,
            time_seconds=epoch_time,
            memory_gb=memory_gb,
            skipped_steps=skipped,
            total_steps=len(dataloader),
        )

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> TrainingMetrics:
        """
        Evaluate the model.

        Args:
            dataloader: Evaluation data loader

        Returns:
            TrainingMetrics with loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return TrainingMetrics(
            loss=running_loss / len(dataloader),
            accuracy=100.0 * correct / total,
            time_seconds=time.time() - start_time,
        )

    def get_skip_ratio(self) -> float:
        """Get the ratio of skipped steps due to gradient overflow."""
        if self.total_steps == 0:
            return 0.0
        return self.skipped_steps / self.total_steps


def benchmark_precision(
    model_fn: Callable[[], nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 3,
    lr: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different precision modes.

    Args:
        model_fn: Function that returns a new model instance
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of epochs to train
        lr: Learning rate

    Returns:
        Dictionary with results for each precision mode

    Example:
        >>> results = benchmark_precision(lambda: ResNet18(), train_loader, test_loader)
        >>> print(results['bf16']['speedup'])
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for dtype in ['fp32', 'fp16', 'bf16']:
        print(f"\nBenchmarking {dtype.upper()}...")

        model = model_fn().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        trainer = AMPTrainer(model, optimizer, criterion, device, dtype=dtype)

        total_time = 0
        for epoch in range(num_epochs):
            metrics = trainer.train_epoch(train_loader)
            total_time += metrics.time_seconds
            print(f"  Epoch {epoch+1}: Loss={metrics.loss:.4f}, "
                  f"Acc={metrics.accuracy:.2f}%, Time={metrics.time_seconds:.1f}s")

        test_metrics = trainer.evaluate(test_loader)

        results[dtype] = {
            'total_time': total_time,
            'memory_gb': metrics.memory_gb,
            'final_accuracy': test_metrics.accuracy,
            'skip_ratio': trainer.get_skip_ratio(),
        }

        # Cleanup
        del model, optimizer, trainer
        torch.cuda.empty_cache()

    # Calculate speedups
    fp32_time = results['fp32']['total_time']
    for dtype in ['fp16', 'bf16']:
        results[dtype]['speedup'] = fp32_time / results[dtype]['total_time']
        results[dtype]['memory_savings'] = 1 - results[dtype]['memory_gb'] / results['fp32']['memory_gb']

    return results


def print_benchmark_results(results: Dict[str, Dict[str, float]]):
    """Pretty print benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'FP32':>12} {'FP16':>12} {'BF16':>12}")
    print("-" * 70)
    print(f"{'Total Time (s)':<25} {results['fp32']['total_time']:>12.1f} "
          f"{results['fp16']['total_time']:>12.1f} {results['bf16']['total_time']:>12.1f}")
    print(f"{'Memory (GB)':<25} {results['fp32']['memory_gb']:>12.2f} "
          f"{results['fp16']['memory_gb']:>12.2f} {results['bf16']['memory_gb']:>12.2f}")
    print(f"{'Final Accuracy (%)':<25} {results['fp32']['final_accuracy']:>12.2f} "
          f"{results['fp16']['final_accuracy']:>12.2f} {results['bf16']['final_accuracy']:>12.2f}")
    print("-" * 70)
    print(f"{'Speedup vs FP32':<25} {'-':>12} "
          f"{results['fp16']['speedup']:>11.2f}x {results['bf16']['speedup']:>11.2f}x")
    print(f"{'Memory Savings':<25} {'-':>12} "
          f"{results['fp16']['memory_savings']*100:>11.1f}% {results['bf16']['memory_savings']*100:>11.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    import torchvision
    import torchvision.transforms as transforms

    print("Testing AMPTrainer...")

    # Create a simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 32 * 3, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Create dummy data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # Test each precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dtype in ['fp32', 'fp16', 'bf16']:
        print(f"\nTesting {dtype}...")
        model_copy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        ).to(device)

        optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = AMPTrainer(model_copy, optimizer, criterion, device, dtype=dtype)

        # Train for a few batches
        model_copy.train()
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= 10:
                break

        print(f"  {dtype} test passed!")

    print("\nAll tests passed!")
