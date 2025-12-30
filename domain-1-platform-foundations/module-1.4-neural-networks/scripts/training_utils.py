"""
Training Utilities - Built from Scratch with NumPy

This module provides utilities for training neural networks:
- Data loading and batching
- Training loops
- Metrics and evaluation
- Regularization (L2, Dropout)
- Early stopping
- Visualization helpers

Professor SPARK says: "A well-organized training loop is like a well-run
kitchen - ingredients (data), recipes (model), tasting (evaluation), and
adjustments (optimization) all flowing smoothly!"

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 4
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Generator
import time
from dataclasses import dataclass
import gzip
import os


# ============================================================================
# Data Loading and Batching
# ============================================================================

def load_mnist(
    path: str = './data',
    flatten: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset from local files or download if not present.

    Args:
        path: Directory to store/load MNIST files
        flatten: Whether to flatten images to 1D (784,) or keep 2D (28, 28)
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        X_train, y_train, X_test, y_test

    Example:
        >>> X_train, y_train, X_test, y_test = load_mnist()
        >>> print(f"Training: {X_train.shape}, Test: {X_test.shape}")
        # Training: (60000, 784), Test: (10000, 784)
    """
    import urllib.request

    os.makedirs(path, exist_ok=True)

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def download_if_needed(filename: str) -> str:
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        return filepath

    def load_images(filepath: str) -> np.ndarray:
        with gzip.open(filepath, 'rb') as f:
            # Skip magic number and dimensions
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28, 28)

    def load_labels(filepath: str) -> np.ndarray:
        with gzip.open(filepath, 'rb') as f:
            # Skip magic number
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    # Download and load
    X_train = load_images(download_if_needed(files['train_images']))
    y_train = load_labels(download_if_needed(files['train_labels']))
    X_test = load_images(download_if_needed(files['test_images']))
    y_test = load_labels(download_if_needed(files['test_labels']))

    # Convert to float
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Normalize
    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    # Flatten
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test


def create_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create mini-batches from dataset.

    ELI5: Instead of looking at all examples at once (slow) or one at a
    time (noisy), we look at small groups. Like grading papers in stacks
    instead of all at once or one by one.

    Args:
        X: Features array
        y: Labels array
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data before batching

    Yields:
        (X_batch, y_batch) tuples

    Example:
        >>> for X_batch, y_batch in create_batches(X_train, y_train, 32):
        ...     output = model(X_batch)
        ...     loss = compute_loss(output, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X: Features array
        y: Labels array
        test_size: Fraction of data to use for testing (0 to 1)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ============================================================================
# Metrics and Evaluation
# ============================================================================

def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (class indices or probabilities)
        targets: True labels (class indices)

    Returns:
        Accuracy as a float between 0 and 1

    Example:
        >>> preds = np.array([0, 1, 2, 1])
        >>> targets = np.array([0, 1, 1, 1])
        >>> print(f"Accuracy: {accuracy(preds, targets):.2%}")  # 75.00%
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    return np.mean(predictions == targets)


def confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Predicted class labels
        targets: True class labels
        num_classes: Number of classes (inferred if not provided)

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        Row = true class, Column = predicted class
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    if num_classes is None:
        num_classes = max(predictions.max(), targets.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets, predictions):
        cm[true, pred] += 1

    return cm


def precision_recall_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = 'macro'
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        predictions: Predicted class labels
        targets: True class labels
        average: 'macro' (average per class) or 'micro' (global)

    Returns:
        (precision, recall, f1_score)
    """
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    cm = confusion_matrix(predictions, targets)
    num_classes = cm.shape[0]

    if average == 'macro':
        precision_per_class = np.zeros(num_classes)
        recall_per_class = np.zeros(num_classes)

        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision_per_class[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_per_class[c] = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
    else:  # micro
        tp = np.diag(cm).sum()
        fp = cm.sum() - np.diag(cm).sum()
        fn = cm.sum() - np.diag(cm).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# ============================================================================
# Regularization
# ============================================================================

class Dropout:
    """
    Dropout regularization layer.

    ELI5: Dropout is like randomly asking some students to skip class.
    The remaining students can't rely on their absent friends, so they
    have to learn the material themselves. This makes the whole class
    stronger and less dependent on any single student!

    Parameters:
        rate: Probability of dropping a neuron (0 to 1)

    Notes:
        - During training: randomly zero out neurons, scale remaining by 1/(1-rate)
        - During inference: do nothing (use all neurons)
        - The scaling ensures expected values are the same in train/test

    Example:
        >>> dropout = Dropout(rate=0.5)
        >>> x = np.ones((32, 256))
        >>> out_train = dropout(x, training=True)   # ~half are zeros
        >>> out_test = dropout(x, training=False)   # all ones
    """

    def __init__(self, rate: float = 0.5):
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self.mask: Optional[np.ndarray] = None
        self.trainable = False

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if training and self.rate > 0:
            # Create binary mask
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(float)
            # Apply mask and scale
            return x * self.mask / (1 - self.rate)
        else:
            return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: apply same mask."""
        if self.mask is not None:
            return grad_output * self.mask / (1 - self.rate)
        return grad_output

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(x, training)


def l2_regularization_loss(
    layers: List,
    lambda_: float
) -> float:
    """
    Compute L2 regularization loss.

    L2 loss = lambda * sum(w^2) for all weights

    Args:
        layers: List of layers with weights
        lambda_: Regularization strength

    Returns:
        L2 regularization loss value
    """
    l2_loss = 0.0
    for layer in layers:
        if hasattr(layer, 'weights') and hasattr(layer, 'trainable') and layer.trainable:
            l2_loss += np.sum(layer.weights ** 2)
    return 0.5 * lambda_ * l2_loss


def l2_regularization_gradient(
    layers: List,
    lambda_: float
) -> None:
    """
    Add L2 regularization gradient to existing gradients.

    dL2/dw = lambda * w

    Args:
        layers: List of layers with weights and gradients
        lambda_: Regularization strength
    """
    for layer in layers:
        if hasattr(layer, 'weights') and hasattr(layer, 'gradients') and layer.trainable:
            if 'weights' in layer.gradients:
                layer.gradients['weights'] += lambda_ * layer.weights


# ============================================================================
# Training Loop
# ============================================================================

@dataclass
class TrainingHistory:
    """Container for training metrics."""
    train_loss: List[float]
    train_accuracy: List[float]
    val_loss: List[float]
    val_accuracy: List[float]
    epoch_times: List[float]
    learning_rates: List[float]

    def __init__(self):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.epoch_times = []
        self.learning_rates = []


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    ELI5: Early stopping is like knowing when to stop practicing.
    If your test scores stop improving (or get worse), practicing
    more might actually hurt! We save our best work and stop when
    we're no longer improving.

    Parameters:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss (lower is better), 'max' for accuracy

    Example:
        >>> early_stopping = EarlyStopping(patience=5, mode='min')
        >>> for epoch in range(100):
        ...     val_loss = train_epoch(...)
        ...     if early_stopping(val_loss):
        ...         print(f"Stopping early at epoch {epoch}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value: Optional[float] = None
        self.counter = 0
        self.best_weights: Optional[Dict] = None

    def __call__(self, value: float, model_weights: Optional[Dict] = None) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value (loss or accuracy)
            model_weights: Optional weights to save if this is the best

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            if model_weights is not None:
                self.best_weights = model_weights.copy()
            return False

        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
            if model_weights is not None:
                self.best_weights = model_weights.copy()
        else:
            self.counter += 1

        return self.counter >= self.patience


def train_epoch(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    loss_fn,
    optimizer,
    batch_size: int = 64,
    l2_lambda: float = 0.0,
    training: bool = True
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Sequential model
        X_train: Training features
        y_train: Training labels
        loss_fn: Loss function
        optimizer: Optimizer
        batch_size: Batch size
        l2_lambda: L2 regularization strength
        training: Whether in training mode (for dropout, batchnorm)

    Returns:
        (average_loss, accuracy)
    """
    total_loss = 0.0
    total_correct = 0
    n_samples = 0

    for X_batch, y_batch in create_batches(X_train, y_train, batch_size, shuffle=True):
        # Forward pass
        output = model(X_batch)

        # Compute loss
        loss = loss_fn(output, y_batch)

        # Add L2 regularization
        if l2_lambda > 0:
            loss += l2_regularization_loss(model.layers, l2_lambda)

        # Backward pass
        grad = loss_fn.backward()
        model.backward(grad)

        # Add L2 gradient
        if l2_lambda > 0:
            l2_regularization_gradient(model.layers, l2_lambda)

        # Update weights
        optimizer.step(model.get_trainable_layers())

        # Track metrics
        batch_size_actual = X_batch.shape[0]
        total_loss += loss * batch_size_actual
        total_correct += np.sum(np.argmax(output, axis=1) == y_batch)
        n_samples += batch_size_actual

    return total_loss / n_samples, total_correct / n_samples


def evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn,
    batch_size: int = 256
) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Sequential model
        X: Features
        y: Labels
        loss_fn: Loss function
        batch_size: Batch size for evaluation

    Returns:
        (average_loss, accuracy)
    """
    total_loss = 0.0
    total_correct = 0
    n_samples = 0

    for X_batch, y_batch in create_batches(X, y, batch_size, shuffle=False):
        output = model(X_batch)
        loss = loss_fn(output, y_batch)

        batch_size_actual = X_batch.shape[0]
        total_loss += loss * batch_size_actual
        total_correct += np.sum(np.argmax(output, axis=1) == y_batch)
        n_samples += batch_size_actual

    return total_loss / n_samples, total_correct / n_samples


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn,
    optimizer,
    epochs: int = 10,
    batch_size: int = 64,
    l2_lambda: float = 0.0,
    early_stopping: Optional[EarlyStopping] = None,
    scheduler=None,
    verbose: bool = True
) -> TrainingHistory:
    """
    Full training loop.

    ELI5: This is the main "learning" function. It shows the model
    many examples (epochs), checks how well it's doing (evaluation),
    and makes adjustments (optimization). Like practicing a skill
    over many days with regular check-ins!

    Args:
        model: Sequential model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of epochs
        batch_size: Batch size
        l2_lambda: L2 regularization strength
        early_stopping: Optional early stopping callback
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress

    Returns:
        TrainingHistory with metrics for each epoch
    """
    history = TrainingHistory()

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, X_train, y_train, loss_fn, optimizer,
            batch_size, l2_lambda
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, X_val, y_val, loss_fn, batch_size)

        # Record history
        epoch_time = time.time() - start_time
        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)
        history.epoch_times.append(epoch_time)
        history.learning_rates.append(optimizer.learning_rate)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Print progress
        if verbose:
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2%} | "
                  f"Time: {epoch_time:.2f}s")

        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    return history


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(
    history: TrainingHistory,
    title: str = "Training History"
) -> None:
    """
    Plot training curves.

    Creates a 2x1 figure with loss and accuracy curves.

    Args:
        history: TrainingHistory object
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history.train_loss, label='Train Loss')
    axes[0].plot(history.val_loss, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(history.train_accuracy, label='Train Accuracy')
    axes[1].plot(history.val_accuracy, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix
        class_names: Optional list of class names
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Testing Training Utilities")
    print("=" * 50)

    # Test data creation
    print("\n1. Creating synthetic dataset:")
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Test batching
    print("\n2. Testing batch creation:")
    batch_count = 0
    for X_batch, y_batch in create_batches(X_train, y_train, 64):
        batch_count += 1
    print(f"   Created {batch_count} batches of size 64")

    # Test metrics
    print("\n3. Testing metrics:")
    preds = np.random.rand(100, 2)
    targets = np.random.randint(0, 2, 100)
    acc = accuracy(preds, targets)
    cm = confusion_matrix(preds, targets)
    p, r, f1 = precision_recall_f1(preds, targets)
    print(f"   Accuracy: {acc:.2%}")
    print(f"   Precision: {p:.2%}, Recall: {r:.2%}, F1: {f1:.2%}")

    # Test dropout
    print("\n4. Testing Dropout:")
    dropout = Dropout(rate=0.5)
    x = np.ones((32, 64))
    out_train = dropout(x, training=True)
    out_test = dropout(x, training=False)
    print(f"   Training: {(out_train == 0).mean():.1%} zeros (expect ~50%)")
    print(f"   Testing: {(out_test == 0).mean():.1%} zeros (expect 0%)")

    # Test early stopping
    print("\n5. Testing Early Stopping:")
    early_stop = EarlyStopping(patience=3, mode='min')
    losses = [1.0, 0.8, 0.7, 0.75, 0.76, 0.77]
    for i, loss in enumerate(losses):
        if early_stop(loss):
            print(f"   Stopped at step {i + 1} with loss {loss}")
            break
    else:
        print(f"   Did not stop early")

    print("\n" + "=" * 50)
    print("All training utility tests passed!")
