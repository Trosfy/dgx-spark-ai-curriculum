"""
Utility Functions for MicroGrad+.

This module provides helper functions for common tasks like setting seeds,
monitoring memory, gradient checking, and data handling.

Example:
    >>> from micrograd_plus.utils import set_seed, gradient_check, get_memory_usage
    >>>
    >>> set_seed(42)  # Reproducible results
    >>> print(get_memory_usage())  # Memory stats
    >>>
    >>> # Verify gradients
    >>> x = Tensor([1, 2, 3], requires_grad=True)
    >>> is_correct = gradient_check(lambda t: (t ** 2).sum(), x)
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple, List, Union
import numpy as np
import os
import sys

__all__ = [
    'set_seed',
    'get_memory_usage',
    'numerical_gradient',
    'gradient_check',
    'check_all_gradients',
    'DataLoader',
    'one_hot',
    'normalize',
    'train_test_split',
    'ProgressBar',
    'plot_training_history',
    'visualize_computation_graph',
    'cleanup_notebook',
]


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Sets seeds for numpy's random number generator.

    Args:
        seed: The seed value to use.

    Example:
        >>> set_seed(42)
        >>> np.random.rand(3)  # Same output every time with seed 42
    """
    np.random.seed(seed)


def get_memory_usage() -> dict:
    """
    Get current memory usage statistics.

    Returns dictionary with memory info in MB. On DGX Spark, this includes
    unified memory which is shared between CPU and GPU.

    Returns:
        Dictionary with 'rss' (resident set size), 'vms' (virtual memory size),
        and 'percent' (percentage of total memory).

    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Memory used: {mem['rss']:.1f} MB ({mem['percent']:.1f}%)")
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        total_mem = psutil.virtual_memory().total

        return {
            'rss': mem_info.rss / (1024 ** 2),  # MB
            'vms': mem_info.vms / (1024 ** 2),  # MB
            'percent': (mem_info.rss / total_mem) * 100
        }
    except ImportError:
        # Fallback if psutil not available
        return {'rss': 0, 'vms': 0, 'percent': 0}


def numerical_gradient(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.

    This is useful for verifying that analytical gradients are correct.
    Uses central differences: (f(x+eps) - f(x-eps)) / (2*eps)

    ELI5:
        To check if you calculated the slope correctly, you can actually
        measure it by looking at two nearby points. Move a tiny bit left,
        measure. Move a tiny bit right, measure. The difference tells you
        the slope. This is slower but more reliable for checking.

    Args:
        f: Function that takes a numpy array and returns a scalar.
        x: Point at which to compute gradient.
        eps: Small perturbation for finite differences.

    Returns:
        Numerical gradient with same shape as x.

    Example:
        >>> def f(x): return (x ** 2).sum()
        >>> x = np.array([1., 2., 3.])
        >>> num_grad = numerical_gradient(f, x)
        >>> print(num_grad)  # [2., 4., 6.] (derivative of x^2 is 2x)
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        # f(x + eps)
        x[idx] = old_value + eps
        f_plus = f(x)

        # f(x - eps)
        x[idx] = old_value - eps
        f_minus = f(x)

        # Restore and compute gradient
        x[idx] = old_value
        grad[idx] = (f_plus - f_minus) / (2 * eps)

        it.iternext()

    return grad


def gradient_check(
    f: Callable,
    x: 'Tensor',
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> Tuple[bool, float]:
    """
    Verify that analytical gradients match numerical gradients.

    Computes gradients analytically via autograd and numerically via
    finite differences, then compares them.

    Args:
        f: Function that takes a Tensor and returns a scalar Tensor.
        x: Input tensor with requires_grad=True.
        eps: Perturbation for numerical gradient.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.

    Returns:
        Tuple of (passed: bool, max_error: float).

    Example:
        >>> from micrograd_plus import Tensor
        >>> x = Tensor([1., 2., 3.], requires_grad=True)
        >>> passed, error = gradient_check(lambda t: (t ** 2).sum(), x)
        >>> print(f"Gradient check {'passed' if passed else 'FAILED'}, max error: {error:.2e}")
    """
    from .tensor import Tensor

    # Compute analytical gradient
    x.zero_grad()
    y = f(x)
    y.backward()
    analytical_grad = x.grad.copy()

    # Compute numerical gradient
    def numpy_f(arr):
        t = Tensor(arr)
        return f(t).data.item()

    numerical_grad = numerical_gradient(numpy_f, x.data.copy(), eps)

    # Compare gradients
    diff = np.abs(analytical_grad - numerical_grad)
    denom = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad))
    relative_error = diff / np.maximum(denom, eps)

    max_error = np.max(relative_error)
    passed = np.allclose(analytical_grad, numerical_grad, atol=atol, rtol=rtol)

    return passed, max_error


def check_all_gradients(
    model: 'Module',
    loss_fn: Callable,
    sample_input: 'Tensor',
    sample_target: 'Tensor',
    eps: float = 1e-5
) -> List[Tuple[str, bool, float]]:
    """
    Check gradients for all parameters in a model.

    Args:
        model: The neural network model.
        loss_fn: Loss function.
        sample_input: Sample input tensor.
        sample_target: Sample target tensor.
        eps: Perturbation for numerical gradients.

    Returns:
        List of tuples (param_name, passed, max_error) for each parameter.

    Example:
        >>> results = check_all_gradients(model, loss_fn, sample_x, sample_y)
        >>> for name, passed, error in results:
        ...     status = "✓" if passed else "✗"
        ...     print(f"{status} {name}: error = {error:.2e}")
    """
    from .tensor import Tensor

    results = []
    params = model.parameters()

    for i, param in enumerate(params):
        # Zero all gradients
        for p in params:
            p.zero_grad()

        # Forward and backward
        output = model(sample_input)
        loss = loss_fn(output, sample_target)
        loss.backward()

        analytical_grad = param.grad.copy()

        # Numerical gradient
        def compute_loss(param_data):
            old_data = param.data.copy()
            param.data = param_data
            out = model(sample_input)
            l = loss_fn(out, sample_target)
            param.data = old_data
            return l.data.item()

        numerical_grad = numerical_gradient(compute_loss, param.data.copy(), eps)

        # Compare
        max_error = np.max(np.abs(analytical_grad - numerical_grad) /
                          (np.abs(analytical_grad) + np.abs(numerical_grad) + 1e-8))
        passed = np.allclose(analytical_grad, numerical_grad, atol=1e-4, rtol=1e-3)

        results.append((f"param_{i}", passed, max_error))

    return results


class DataLoader:
    """
    Simple data loader for batching and shuffling data.

    Args:
        X: Input data array.
        y: Target data array.
        batch_size: Samples per batch.
        shuffle: Whether to shuffle each epoch.

    Example:
        >>> loader = DataLoader(X_train, y_train, batch_size=32)
        >>> for batch_x, batch_y in loader:
        ...     # Training step
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot encoding.

    Args:
        labels: Array of integer class labels.
        num_classes: Total number of classes.

    Returns:
        One-hot encoded array of shape (len(labels), num_classes).

    Example:
        >>> labels = np.array([0, 1, 2, 1])
        >>> one_hot(labels, 3)
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [0, 1, 0]])
    """
    n_samples = len(labels)
    result = np.zeros((n_samples, num_classes), dtype=np.float32)
    result[np.arange(n_samples), labels.astype(int)] = 1.0
    return result


def normalize(X: np.ndarray, mean: Optional[np.ndarray] = None,
              std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data to zero mean and unit variance.

    Args:
        X: Data to normalize.
        mean: Optional precomputed mean (for test data).
        std: Optional precomputed std (for test data).

    Returns:
        Tuple of (normalized_X, mean, std).

    Example:
        >>> X_norm, mean, std = normalize(X_train)
        >>> X_test_norm, _, _ = normalize(X_test, mean, std)
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero

    return (X - mean) / std, mean, std


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.

    Args:
        X: Input features.
        y: Target values.
        test_size: Fraction of data for testing (0-1).
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class ProgressBar:
    """
    Simple progress bar for training loops.

    Args:
        total: Total number of iterations.
        prefix: String to display before the bar.
        length: Character length of the bar.
        ascii: Use ASCII characters (# and -) instead of Unicode blocks.
               Set to True if Unicode doesn't render in your terminal.

    Example:
        >>> pbar = ProgressBar(100, prefix='Training')
        >>> for i in range(100):
        ...     # Do work
        ...     pbar.update(i + 1)
        >>> pbar.finish()
    """

    def __init__(self, total: int, prefix: str = '', length: int = 30,
                 ascii: bool = False):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        # Use ASCII fallback if requested or if Unicode might not be supported
        self.fill_char = '#' if ascii else '█'
        self.empty_char = '-' if ascii else '░'

    def update(self, current: int, suffix: str = '') -> None:
        """Update the progress bar."""
        self.current = current
        percent = current / self.total
        filled = int(self.length * percent)
        bar = self.fill_char * filled + self.empty_char * (self.length - filled)
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent:.0%} {suffix}')
        sys.stdout.flush()

    def finish(self) -> None:
        """Complete the progress bar."""
        self.update(self.total)
        print()


def plot_training_history(
    history: dict,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.

    Example:
        >>> history = {'train_loss': [...], 'val_loss': [...], 'train_acc': [...]}
        >>> plot_training_history(history)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    if 'train_acc' in history or 'val_acc' in history:
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc')
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_computation_graph(tensor: 'Tensor', filename: str = 'graph') -> None:
    """
    Visualize the computation graph of a tensor.

    Requires graphviz to be installed.

    Args:
        tensor: The output tensor whose graph to visualize.
        filename: Output filename (without extension).

    Example:
        >>> x = Tensor([1, 2], requires_grad=True)
        >>> y = x ** 2
        >>> z = y.sum()
        >>> visualize_computation_graph(z, 'my_graph')
    """
    try:
        from graphviz import Digraph
    except ImportError:
        print("graphviz not available. Install with: pip install graphviz")
        return

    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})

    # Collect all nodes
    nodes = set()
    edges = set()

    def trace(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                trace(child)

    trace(tensor)

    # Add nodes
    for n in nodes:
        uid = str(id(n))
        label = f"{n.shape}\n{n._op}" if n._op else f"{n.shape}"
        dot.node(uid, label, shape='record')

    # Add edges
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))

    dot.render(filename, view=True)


def cleanup_notebook(namespace: dict = None) -> None:
    """
    Clean up common notebook variables to free memory.

    This function properly cleans up training-related variables that may
    consume memory. Call at the end of notebooks.

    Args:
        namespace: The namespace dict to clean (typically globals()).
                  If None, only runs garbage collection.

    Example:
        >>> # At the end of a notebook cell:
        >>> cleanup_notebook(globals())
        Cleanup complete!
    """
    import gc

    # Variables commonly created during training
    cleanup_vars = [
        'model', 'X_train', 'X_test', 'y_train', 'y_test',
        'X_tensor', 'y_tensor', 'train_loader', 'test_loader',
        'optimizer', 'loss_fn', 'history', 'X_train_subset',
        'y_train_subset', 'X_train_norm', 'X_test_norm',
        'train_loss', 'test_loss', 'logits', 'predictions'
    ]

    if namespace is not None:
        for var_name in cleanup_vars:
            if var_name in namespace:
                try:
                    del namespace[var_name]
                except (KeyError, NameError):
                    pass

    gc.collect()
    print("Cleanup complete!")
