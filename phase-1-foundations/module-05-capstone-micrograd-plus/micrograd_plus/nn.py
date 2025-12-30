"""
Neural Network Utilities for MicroGrad+.

This module provides high-level utilities for building and training neural
networks, including the Sequential container and training helpers.

Example:
    >>> from micrograd_plus import nn, Linear, ReLU, CrossEntropyLoss, Adam
    >>>
    >>> # Build a simple MLP
    >>> model = Sequential(
    ...     Linear(784, 256),
    ...     ReLU(),
    ...     Linear(256, 128),
    ...     ReLU(),
    ...     Linear(128, 10)
    ... )
    >>>
    >>> # Setup training
    >>> loss_fn = CrossEntropyLoss()
    >>> optimizer = Adam(model.parameters(), lr=0.001)
    >>>
    >>> # Train
    >>> for epoch in range(10):
    ...     loss = train_epoch(model, train_loader, loss_fn, optimizer)
    ...     print(f"Epoch {epoch}: Loss = {loss:.4f}")
"""

from __future__ import annotations
from typing import List, Iterator, Tuple, Callable, Optional, TYPE_CHECKING
import numpy as np
from .tensor import Tensor
from .layers import Module

if TYPE_CHECKING:
    from .utils import DataLoader
    from .optimizers import Optimizer


class Sequential(Module):
    """
    A sequential container for neural network layers.

    Layers are added in order and executed sequentially during forward pass.
    This is the simplest way to build feedforward neural networks.

    ELI5:
        Think of Sequential like a factory assembly line. Each layer is a
        station that takes the product, does something to it, and passes it
        to the next station. The final product comes out at the end!

    Args:
        *layers: Variable number of layers to add.

    Example:
        >>> model = Sequential(
        ...     Linear(784, 256),
        ...     ReLU(),
        ...     Dropout(0.2),
        ...     Linear(256, 10)
        ... )
        >>>
        >>> x = Tensor(np.random.randn(32, 784))
        >>> output = model(x)  # Passes through all layers
        >>> print(output.shape)  # (32, 10)
    """

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers sequentially.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through all layers.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Tensor]:
        """
        Collect all parameters from all layers.

        Returns:
            List of all trainable parameters.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self, mode: bool = True) -> 'Sequential':
        """Set training mode for all layers."""
        self._training = mode
        for layer in self.layers:
            layer.train(mode)
        return self

    def eval(self) -> 'Sequential':
        """Set evaluation mode for all layers."""
        return self.train(False)

    def add(self, layer: Module) -> None:
        """Add a layer to the end of the sequence."""
        self.layers.append(layer)

    def __getitem__(self, idx: int) -> Module:
        """Get layer by index."""
        return self.layers[idx]

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self.layers)

    def __iter__(self) -> Iterator[Module]:
        """Iterate over layers."""
        return iter(self.layers)

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)


def train_epoch(
    model: Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: Optimizer,
    verbose: bool = False
) -> float:
    """
    Train the model for one epoch.

    This function handles the complete training loop for a single epoch:
    forward pass, loss computation, backward pass, and parameter update.

    Args:
        model: The neural network to train.
        dataloader: DataLoader providing batches of (inputs, targets).
        loss_fn: Loss function to minimize.
        optimizer: Optimizer for updating parameters.
        verbose: If True, print progress for each batch.

    Returns:
        Average loss over all batches.

    Example:
        >>> for epoch in range(10):
        ...     train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        ...     print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Convert to Tensor if needed
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs, requires_grad=True)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        num_batches += 1

        if verbose and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    metrics: Optional[List[Callable]] = None
) -> Tuple[float, dict]:
    """
    Evaluate the model on a dataset.

    Args:
        model: The neural network to evaluate.
        dataloader: DataLoader providing batches of (inputs, targets).
        loss_fn: Loss function to compute.
        metrics: Optional list of metric functions.

    Returns:
        Tuple of (average_loss, metrics_dict).

    Example:
        >>> val_loss, metrics = evaluate(model, val_loader, loss_fn)
        >>> print(f"Validation Loss: {val_loss:.4f}")
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        if not isinstance(inputs, Tensor):
            inputs = Tensor(inputs)
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Forward pass (no gradient computation needed)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

        # Store predictions and targets for metrics
        all_preds.append(outputs.data)
        all_targets.append(targets.data)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Compute metrics if provided
    metrics_dict = {}
    if metrics:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        for metric_fn in metrics:
            metric_name = metric_fn.__name__ if hasattr(metric_fn, '__name__') else str(metric_fn)
            metrics_dict[metric_name] = metric_fn(all_preds, all_targets)

    return avg_loss, metrics_dict


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model outputs of shape (batch_size, num_classes).
        targets: Target class indices of shape (batch_size,).

    Returns:
        Accuracy as a float between 0 and 1.

    Example:
        >>> preds = np.array([[0.1, 0.9], [0.8, 0.2]])  # Class probabilities
        >>> targets = np.array([1, 0])  # True classes
        >>> acc = accuracy(preds, targets)
        >>> print(f"Accuracy: {acc:.2%}")  # 100%
    """
    if predictions.ndim == 2:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions

    if targets.ndim == 2:
        target_classes = np.argmax(targets, axis=1)
    else:
        target_classes = targets

    return np.mean(pred_classes == target_classes)


def count_parameters(model: Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: The model to analyze.

    Returns:
        Total number of trainable parameters.

    Example:
        >>> model = Sequential(Linear(784, 256), Linear(256, 10))
        >>> print(f"Total params: {count_parameters(model):,}")
        >>> # 784*256 + 256 + 256*10 + 10 = 203,530
    """
    return sum(p.size for p in model.parameters())


def save_model(model: Module, path: str) -> None:
    """
    Save model parameters to a file.

    Args:
        model: The model to save.
        path: File path for saving.

    Example:
        >>> save_model(model, "model.npz")
    """
    params = {}
    for i, p in enumerate(model.parameters()):
        params[f"param_{i}"] = p.data
    np.savez(path, **params)


def load_model(model: Module, path: str) -> None:
    """
    Load model parameters from a file.

    Args:
        model: The model to load parameters into.
        path: File path to load from.

    Example:
        >>> load_model(model, "model.npz")
    """
    loaded = np.load(path)
    for i, p in enumerate(model.parameters()):
        p.data = loaded[f"param_{i}"]


# Note: DataLoader is defined in utils.py and imported from there
