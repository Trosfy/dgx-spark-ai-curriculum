"""
Checkpoint Manager - Production-Ready Training State Management

This module provides robust checkpointing with best model tracking,
early stopping, and training resumption.

Components:
    - CheckpointManager: Basic checkpoint management
    - ProductionCheckpointManager: Full-featured checkpoint system
    - save_checkpoint / load_checkpoint: Utility functions

Example:
    >>> from checkpoint_manager import ProductionCheckpointManager
    >>> ckpt = ProductionCheckpointManager('./checkpoints', model, optimizer)
    >>> result = ckpt.step(epoch, train_loss, val_loss, val_acc)
    >>> if result['should_stop']:
    ...     print("Early stopping triggered!")

Author: DGX Spark AI Curriculum
"""

__all__ = [
    'save_checkpoint',
    'load_checkpoint',
    'CheckpointManager',
    'ProductionCheckpointManager',
]

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import json
import shutil


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    path: Union[str, Path],
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a complete training checkpoint.

    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional dict of metrics (loss, accuracy, etc.)
        extra: Optional extra data to save

    Returns:
        Path where checkpoint was saved

    Example:
        >>> save_checkpoint(model, optimizer, 10, 'checkpoint.pth',
        ...                 metrics={'loss': 0.5, 'accuracy': 95.0})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics:
        checkpoint['metrics'] = metrics

    if extra:
        checkpoint['extra'] = extra

    # Atomic save
    temp_path = path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(path)

    return path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to map checkpoint to

    Returns:
        Checkpoint dictionary with metadata

    Example:
        >>> checkpoint = load_checkpoint('checkpoint.pth', model, optimizer)
        >>> print(f"Resuming from epoch {checkpoint['epoch']}")
    """
    # weights_only=False is required because checkpoints contain optimizer
    # state dicts and custom metadata beyond just model weights.
    # Note: Only load checkpoints from trusted sources.
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


class CheckpointManager:
    """
    Basic checkpoint manager with best model tracking.

    Features:
        - Save/load checkpoints
        - Track best model
        - Early stopping support
        - Checkpoint rotation

    Args:
        checkpoint_dir: Directory to save checkpoints
        mode: 'min' or 'max' - whether lower or higher metric is better
        patience: Epochs to wait before early stopping
        max_checkpoints: Maximum number of checkpoints to keep

    Example:
        >>> manager = CheckpointManager('./checkpoints', mode='min', patience=5)
        >>> is_best = manager.save_if_best(model, optimizer, epoch, val_loss)
        >>> if manager.should_stop():
        ...     break
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        mode: str = 'min',
        patience: int = 10,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.patience = patience
        self.max_checkpoints = max_checkpoints

        # State
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.saved_checkpoints = []

    def _is_better(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        if self.mode == 'min':
            return metric < self.best_metric
        return metric > self.best_metric

    def save_if_best(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metric: float,
        scheduler: Optional[Any] = None,
    ) -> bool:
        """
        Save checkpoint if metric improved.

        Returns:
            True if this was the best model
        """
        is_best = self._is_better(metric)

        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        save_checkpoint(
            model, optimizer, epoch, latest_path,
            scheduler=scheduler,
            metrics={'metric': metric, 'best': self.best_metric}
        )

        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'epoch_{epoch:04d}.pth'
        save_checkpoint(model, optimizer, epoch, epoch_path, scheduler=scheduler)
        self.saved_checkpoints.append(epoch_path)

        # Cleanup old checkpoints
        while len(self.saved_checkpoints) > self.max_checkpoints:
            old = self.saved_checkpoints.pop(0)
            if old.exists():
                old.unlink()

        # Handle best model
        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
            self.patience_counter = 0

            best_path = self.checkpoint_dir / 'best.pth'
            shutil.copy(latest_path, best_path)
        else:
            self.patience_counter += 1

        return is_best

    def should_stop(self) -> bool:
        """Check if early stopping should trigger."""
        return self.patience_counter >= self.patience

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / 'best.pth'
        return load_checkpoint(best_path, model, optimizer, scheduler)

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load the latest checkpoint for resuming."""
        latest_path = self.checkpoint_dir / 'latest.pth'
        return load_checkpoint(latest_path, model, optimizer, scheduler)


class ProductionCheckpointManager:
    """
    Production-ready checkpoint manager with full features.

    Features:
        - Atomic saves (corruption-proof)
        - Training history persistence
        - Detailed logging
        - Automatic cleanup
        - Resume support

    Args:
        checkpoint_dir: Directory for checkpoints
        model: Model to checkpoint
        optimizer: Optimizer to checkpoint
        scheduler: Optional scheduler
        mode: 'min' or 'max'
        patience: Early stopping patience
        max_checkpoints: Max checkpoints to keep
        save_every: Save every N epochs regardless of improvement
        device: Device to map loaded tensors to (default: None)

    Example:
        >>> manager = ProductionCheckpointManager(
        ...     './checkpoints', model, optimizer, mode='max', patience=10
        ... )
        >>> for epoch in range(100):
        ...     train_loss = train_epoch(...)
        ...     val_loss, val_acc = validate(...)
        ...     result = manager.step(epoch, train_loss, val_loss, val_acc)
        ...     if result['should_stop']:
        ...         break
        >>> manager.load_best()
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        mode: str = 'min',
        patience: int = 10,
        max_checkpoints: int = 5,
        save_every: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Default to CUDA if available, otherwise CPU
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        self.mode = mode
        self.patience = patience
        self.max_checkpoints = max_checkpoints
        self.save_every = save_every

        # State
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.current_epoch = 0
        self.history = []
        self.checkpoint_files = []

        # Try to load existing history
        self._load_history()

    def _load_history(self):
        """Load training history if exists."""
        history_path = self.checkpoint_dir / 'history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.best_metric = data.get('best_metric', self.best_metric)
                self.best_epoch = data.get('best_epoch', 0)

    def _save_history(self):
        """Save training history."""
        history_path = self.checkpoint_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_metric': self.best_metric,
                'best_epoch': self.best_epoch,
            }, f, indent=2)

    def _is_better(self, metric: float) -> bool:
        if self.mode == 'min':
            return metric < self.best_metric
        return metric > self.best_metric

    def _atomic_save(self, path: Path, checkpoint: dict):
        """Save checkpoint atomically."""
        temp_path = path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(path)

    def step(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metric: float,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Record training step and manage checkpoints.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_metric: Metric to track for best model

        Returns:
            Dict with 'is_best', 'should_stop', and status info
        """
        self.current_epoch = epoch
        is_best = self._is_better(val_metric)

        # Build checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().isoformat(),
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest
        self._atomic_save(self.checkpoint_dir / 'latest.pth', checkpoint)

        # Save periodic
        if epoch % self.save_every == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:04d}.pth'
            self._atomic_save(epoch_path, checkpoint)
            self.checkpoint_files.append(epoch_path)

            # Cleanup
            while len(self.checkpoint_files) > self.max_checkpoints:
                old = self.checkpoint_files.pop(0)
                if old.exists():
                    old.unlink()

        # Handle best
        if is_best:
            self.best_metric = val_metric
            self.best_epoch = epoch
            self.patience_counter = 0
            checkpoint['best_metric'] = val_metric
            self._atomic_save(self.checkpoint_dir / 'best.pth', checkpoint)
        else:
            self.patience_counter += 1

        # Update history
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'is_best': is_best,
        }
        if extra_metrics:
            entry.update(extra_metrics)
        self.history.append(entry)
        self._save_history()

        return {
            'is_best': is_best,
            'should_stop': self.patience_counter >= self.patience,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
        }

    def load_best(self) -> Dict[str, Any]:
        """Load best checkpoint."""
        path = self.checkpoint_dir / 'best.pth'
        return self._load(path)

    def load_latest(self) -> Dict[str, Any]:
        """Load latest checkpoint for resuming."""
        path = self.checkpoint_dir / 'latest.pth'
        return self._load(path)

    def _load(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint and restore state."""
        # weights_only=False required for optimizer/scheduler state
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.best_epoch = checkpoint.get('best_epoch', self.best_epoch)
        self.current_epoch = checkpoint['epoch']

        return checkpoint

    def get_history(self) -> list:
        """Get training history."""
        return self.history.copy()


if __name__ == '__main__':
    import tempfile

    print("Testing checkpoint manager...")

    # Create test model
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ProductionCheckpointManager(
            tmpdir, model, optimizer, mode='min', patience=3
        )

        # Simulate training
        for epoch in range(10):
            # Simulated metrics
            val_metric = 1.0 - epoch * 0.1 + (epoch % 3) * 0.05

            result = manager.step(
                epoch=epoch,
                train_loss=0.5,
                val_loss=val_metric,
                val_metric=val_metric,
            )

            status = " (BEST)" if result['is_best'] else ""
            print(f"Epoch {epoch}: metric={val_metric:.3f}{status}")

            if result['should_stop']:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"\nBest metric: {manager.best_metric:.3f} at epoch {manager.best_epoch}")
        print("Test passed!")
