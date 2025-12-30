"""
Training Utilities for Computer Vision Module

This module provides training helpers optimized for DGX Spark.

Features:
- Training loops with progress bars
- Evaluation functions
- Learning rate schedulers
- Checkpoint management
- Mixed precision training support

Example usage:
    from training_utils import Trainer, get_optimizer, get_scheduler

    trainer = Trainer(model, train_loader, val_loader, device='cuda')
    history = trainer.fit(epochs=10, lr=0.001)
"""

__all__ = [
    'Trainer',
    'get_optimizer',
    'get_scheduler',
    'EarlyStopping',
    'benchmark_inference',
]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Optional, Callable
from tqdm.auto import tqdm
import time
from pathlib import Path


class Trainer:
    """
    A flexible trainer for PyTorch models.

    Supports:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Checkpoint saving/loading
    - Progress tracking

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (default: CrossEntropyLoss)
        device: Device to train on (default: 'cuda' if available)
        use_amp: Use automatic mixed precision (default: True for GPU)

    Example:
        >>> model = ResNet18(num_classes=10)
        >>> trainer = Trainer(model, train_loader, val_loader)
        >>> history = trainer.fit(epochs=10, lr=0.001)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        use_amp: bool = True
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.use_amp = use_amp and self.device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })

        return total_loss / total, 100. * correct / total

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        for inputs, targets in tqdm(self.val_loader, desc='Evaluating', leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / total, 100. * correct / total

    def fit(
        self,
        epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        warmup_epochs: int = 0,
        max_grad_norm: float = 1.0,
        save_best: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            epochs: Number of epochs to train
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_epochs: Number of warmup epochs
            max_grad_norm: Maximum gradient norm for clipping
            save_best: Save best model checkpoint
            save_path: Path to save checkpoints

        Returns:
            Training history dictionary
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Create scheduler with optional warmup
        steps_per_epoch = len(self.train_loader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        if warmup_steps > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_steps
            )
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps]
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        best_acc = 0
        save_path = Path(save_path) if save_path else Path('checkpoints')
        save_path.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training on {self.device} | AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            train_loss, train_acc = self.train_epoch(optimizer, scheduler, max_grad_norm)
            val_loss, val_acc = self.evaluate()

            current_lr = optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.1f}%")
            print(f"  LR:    {current_lr:.6f}")

            if save_best and val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(save_path / 'best_model.pth')
                print(f"  ✓ Saved best model (acc={val_acc:.1f}%)")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    lr: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs
) -> optim.Optimizer:
    """
    Get an optimizer by name.

    Args:
        model: PyTorch model
        optimizer_name: One of 'adam', 'adamw', 'sgd'
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer
    """
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }

    name = optimizer_name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    if name == 'sgd':
        kwargs.setdefault('momentum', 0.9)

    return optimizers[name](model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 10,
    steps_per_epoch: int = 100,
    **kwargs
) -> optim.lr_scheduler._LRScheduler:
    """
    Get a learning rate scheduler by name.

    Args:
        optimizer: Configured optimizer
        scheduler_name: One of 'cosine', 'step', 'plateau', 'onecycle'
        epochs: Number of training epochs
        steps_per_epoch: Number of steps per epoch
        **kwargs: Additional scheduler arguments

    Returns:
        Configured scheduler
    """
    total_steps = epochs * steps_per_epoch

    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    elif scheduler_name == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            total_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for accuracy

    Example:
        >>> early_stop = EarlyStopping(patience=5)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stop(val_loss):
        ...         print("Early stopping!")
        ...         break
    """

    def __init__(self, patience: int = 5, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark model inference speed.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations
        device: Device to run on

    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    torch.cuda.synchronize() if device == 'cuda' else None

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize() if device == 'cuda' else None
            times.append(time.perf_counter() - start)

    times = [t * 1000 for t in times]  # Convert to ms

    return {
        'mean_ms': sum(times) / len(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'min_ms': min(times),
        'max_ms': max(times),
        'fps': 1000 / (sum(times) / len(times))
    }


if __name__ == "__main__":
    print("Training utilities loaded successfully!")
