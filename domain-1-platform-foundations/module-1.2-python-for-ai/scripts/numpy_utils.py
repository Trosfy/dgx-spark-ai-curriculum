"""
NumPy Utilities for Machine Learning
=====================================

Production-ready NumPy utilities for common ML operations.
Optimized for performance and memory efficiency on DGX Spark.

This module is part of the DGX Spark AI Curriculum - Module 1.2.

Features:
- Vectorized distance computations
- Efficient similarity calculations
- Softmax and other activation functions
- Batch operations with memory awareness
- Einsum-based tensor operations

Example Usage:
    >>> from numpy_utils import pairwise_distances, cosine_similarity, softmax
    >>>
    >>> # Compute pairwise distances
    >>> dists = pairwise_distances(embeddings)
    >>>
    >>> # Compute cosine similarity matrix
    >>> sims = cosine_similarity(queries, keys)
    >>>
    >>> # Apply softmax with temperature
    >>> probs = softmax(logits, temperature=0.7)

Author: Professor SPARK
Date: 2024
"""

from typing import Optional, Tuple, Union, Literal
import numpy as np

__all__ = [
    # Distance functions
    'pairwise_distances',
    'pairwise_distances_chunked',
    'euclidean_distance',

    # Similarity functions
    'cosine_similarity',
    'dot_product_similarity',

    # Activation functions
    'softmax',
    'log_softmax',
    'relu',
    'gelu',
    'sigmoid',

    # Normalization
    'l2_normalize',
    'batch_normalize',
    'layer_normalize',

    # Tensor operations
    'attention_scores',
    'batch_matmul',
    'outer_product_batch',

    # Utilities
    'one_hot',
    'top_k',
    'check_contiguous',
    'estimate_memory_usage',
]


# =============================================================================
# Distance Functions
# =============================================================================

def pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: Literal['euclidean', 'squared_euclidean'] = 'euclidean'
) -> np.ndarray:
    """
    Compute pairwise distances between all points efficiently.

    Uses the formula: ||a-b||² = ||a||² + ||b||² - 2*a·b
    which is faster than explicit subtraction for large arrays.

    Args:
        X: First set of points, shape (n_samples, n_features)
        Y: Second set of points, shape (m_samples, n_features).
           If None, computes distances within X.
        metric: Distance metric - 'euclidean' or 'squared_euclidean'

    Returns:
        Distance matrix of shape (n_samples, m_samples)

    Example:
        >>> points = np.random.randn(100, 64)
        >>> dists = pairwise_distances(points)
        >>> print(dists.shape)  # (100, 100)

    Memory: O(n*m) for the output matrix
    Time: O(n*m*d) for n,m points of dimension d
    """
    if Y is None:
        Y = X

    # Compute squared norms
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y ** 2, axis=1)  # (m,)

    # ||X - Y||² = ||X||² + ||Y||² - 2*X·Y
    sq_distances = X_sq + Y_sq - 2 * (X @ Y.T)

    # Handle numerical errors (small negatives from floating point)
    sq_distances = np.maximum(sq_distances, 0)

    if metric == 'euclidean':
        return np.sqrt(sq_distances)
    return sq_distances


def pairwise_distances_chunked(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    chunk_size: int = 1000,
    metric: Literal['euclidean', 'squared_euclidean'] = 'euclidean'
) -> np.ndarray:
    """
    Memory-efficient pairwise distances computed in chunks.

    Useful when the full distance matrix doesn't fit in memory.

    Args:
        X: First set of points, shape (n_samples, n_features)
        Y: Second set of points, shape (m_samples, n_features)
        chunk_size: Number of X samples to process at once
        metric: Distance metric

    Returns:
        Distance matrix of shape (n_samples, m_samples)

    Example:
        >>> # For very large datasets
        >>> dists = pairwise_distances_chunked(large_X, large_Y, chunk_size=500)
    """
    if Y is None:
        Y = X

    n_samples = X.shape[0]
    m_samples = Y.shape[0]

    # Pre-allocate output
    distances = np.empty((n_samples, m_samples), dtype=X.dtype)

    # Precompute Y squared norms
    Y_sq = np.sum(Y ** 2, axis=1)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk = X[start:end]

        X_sq = np.sum(X_chunk ** 2, axis=1, keepdims=True)
        sq_dists = X_sq + Y_sq - 2 * (X_chunk @ Y.T)
        sq_dists = np.maximum(sq_dists, 0)

        if metric == 'euclidean':
            distances[start:end] = np.sqrt(sq_dists)
        else:
            distances[start:end] = sq_dists

    return distances


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute element-wise Euclidean distance between corresponding vectors.

    Args:
        a: Array of shape (..., n_features)
        b: Array of shape (..., n_features)

    Returns:
        Array of distances with shape (...)

    Example:
        >>> a = np.random.randn(32, 64)
        >>> b = np.random.randn(32, 64)
        >>> dists = euclidean_distance(a, b)  # Shape: (32,)
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


# =============================================================================
# Similarity Functions
# =============================================================================

def cosine_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute pairwise cosine similarity.

    cosine_sim(a, b) = (a · b) / (||a|| * ||b||)

    Args:
        X: First set of vectors, shape (n_samples, n_features)
        Y: Second set of vectors, shape (m_samples, n_features).
           If None, computes similarity within X.
        eps: Small constant for numerical stability

    Returns:
        Similarity matrix of shape (n_samples, m_samples)
        Values range from -1 (opposite) to 1 (identical)

    Example:
        >>> embeddings = np.random.randn(100, 768)
        >>> sims = cosine_similarity(embeddings)
        >>> print(np.diag(sims))  # All ~1.0 (self-similarity)
    """
    if Y is None:
        Y = X

    # Normalize to unit length
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + eps)

    return X_norm @ Y_norm.T


def dot_product_similarity(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute pairwise dot product similarity.

    Faster than cosine but not normalized.

    Args:
        X: First set of vectors, shape (n_samples, n_features)
        Y: Second set of vectors. If None, uses X.

    Returns:
        Similarity matrix of shape (n_samples, m_samples)
    """
    if Y is None:
        Y = X
    return X @ Y.T


# =============================================================================
# Activation Functions
# =============================================================================

def softmax(
    x: np.ndarray,
    axis: int = -1,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Numerically stable softmax function.

    softmax(x)_i = exp(x_i / T) / sum(exp(x_j / T))

    Args:
        x: Input array
        axis: Axis along which to apply softmax
        temperature: Scaling factor (lower = sharper, higher = smoother)

    Returns:
        Probability distribution (sums to 1 along axis)

    Example:
        >>> logits = np.random.randn(32, 10)
        >>> probs = softmax(logits)
        >>> print(probs.sum(axis=-1))  # All ~1.0
    """
    x_scaled = x / temperature
    x_max = np.max(x_scaled, axis=axis, keepdims=True)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(
    x: np.ndarray,
    axis: int = -1
) -> np.ndarray:
    """
    Numerically stable log-softmax function.

    More numerically stable than log(softmax(x)) for loss computation.

    Args:
        x: Input array
        axis: Axis along which to apply log-softmax

    Returns:
        Log probabilities (log of softmax output)
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return x - logsumexp


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation.

    ReLU(x) = max(0, x)
    """
    return np.maximum(0, x)


def gelu(x: np.ndarray, approximate: bool = True) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    Used in BERT, GPT, and modern transformers.

    Args:
        x: Input array
        approximate: If True (default), use faster tanh approximation.
                    If False, requires scipy for exact computation.

    Returns:
        GELU-activated values

    Note:
        The exact computation (approximate=False) requires scipy.
        Install with: pip install scipy
    """
    if approximate:
        # tanh approximation (faster, no scipy required)
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    else:
        # Exact computation using error function
        try:
            from scipy.special import erf
        except ImportError:
            raise ImportError(
                "scipy is required for exact GELU computation. "
                "Install with: pip install scipy, or use approximate=True"
            )
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    sigma(x) = 1 / (1 + exp(-x))

    Numerically stable implementation.
    """
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


# =============================================================================
# Normalization Functions
# =============================================================================

def l2_normalize(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 norm = 1).

    Args:
        x: Input array
        axis: Axis along which to normalize
        eps: Small constant for numerical stability

    Returns:
        Normalized array where ||x||_2 = 1 along axis

    Example:
        >>> embeddings = np.random.randn(100, 768)
        >>> normalized = l2_normalize(embeddings)
        >>> norms = np.linalg.norm(normalized, axis=1)
        >>> print(np.allclose(norms, 1.0))  # True
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def batch_normalize(
    x: np.ndarray,
    axis: int = 0,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Batch normalization (normalize across batch dimension).

    Transforms to mean=0, std=1 per feature.

    Args:
        x: Input array of shape (batch_size, features)
        axis: Batch axis
        eps: Small constant for numerical stability

    Returns:
        Batch-normalized array
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


def layer_normalize(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Layer normalization (normalize across feature dimension).

    Each sample is normalized independently.

    Args:
        x: Input array
        axis: Feature axis (usually -1)
        eps: Small constant for numerical stability

    Returns:
        Layer-normalized array
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


# =============================================================================
# Tensor Operations (einsum-based)
# =============================================================================

def attention_scores(
    Q: np.ndarray,
    K: np.ndarray,
    scale: bool = True
) -> np.ndarray:
    """
    Compute attention scores using einsum.

    Supports standard and multi-head attention shapes.

    Args:
        Q: Query tensor
           - (seq, dim) for single-head
           - (batch, seq, dim) for batched
           - (batch, heads, seq, dim) for multi-head
        K: Key tensor (same shape as Q)
        scale: Whether to scale by sqrt(d_k)

    Returns:
        Attention scores of appropriate shape

    Example:
        >>> Q = np.random.randn(8, 12, 64, 64)  # (batch, heads, seq, dim)
        >>> K = np.random.randn(8, 12, 64, 64)
        >>> scores = attention_scores(Q, K)
        >>> print(scores.shape)  # (8, 12, 64, 64)
    """
    ndim = Q.ndim

    if ndim == 2:
        # (seq, dim) @ (dim, seq) -> (seq, seq)
        scores = np.einsum('qd,kd->qk', Q, K)
        d_k = Q.shape[-1]
    elif ndim == 3:
        # (batch, seq, dim)
        scores = np.einsum('bqd,bkd->bqk', Q, K)
        d_k = Q.shape[-1]
    elif ndim == 4:
        # (batch, heads, seq, dim)
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K)
        d_k = Q.shape[-1]
    else:
        raise ValueError(f"Unsupported input dimension: {ndim}")

    if scale:
        scores = scores / np.sqrt(d_k)

    return scores


def batch_matmul(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Batch matrix multiplication using einsum.

    Handles various input shapes flexibly.

    Args:
        A: First tensor
        B: Second tensor

    Returns:
        Batch matrix product

    Example:
        >>> A = np.random.randn(32, 64, 128)
        >>> B = np.random.randn(32, 128, 64)
        >>> C = batch_matmul(A, B)
        >>> print(C.shape)  # (32, 64, 64)
    """
    if A.ndim == 3 and B.ndim == 3:
        return np.einsum('bik,bkj->bij', A, B)
    elif A.ndim == 4 and B.ndim == 4:
        return np.einsum('bhik,bhkj->bhij', A, B)
    else:
        # Fall back to numpy's matmul which handles broadcasting
        return A @ B


def outer_product_batch(
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    Compute outer product for each pair of vectors in a batch.

    Args:
        a: First batch of vectors, shape (batch, dim_a)
        b: Second batch of vectors, shape (batch, dim_b)

    Returns:
        Batch of outer products, shape (batch, dim_a, dim_b)

    Example:
        >>> a = np.random.randn(32, 64)
        >>> b = np.random.randn(32, 128)
        >>> outer = outer_product_batch(a, b)
        >>> print(outer.shape)  # (32, 64, 128)
    """
    return np.einsum('bi,bj->bij', a, b)


# =============================================================================
# Utility Functions
# =============================================================================

def one_hot(
    labels: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Convert integer labels to one-hot encoding using broadcasting.

    Args:
        labels: Integer labels, shape (n_samples,) or (n_samples, 1)
        num_classes: Number of classes

    Returns:
        One-hot encoded array of shape (n_samples, num_classes)

    Example:
        >>> labels = np.array([0, 2, 1, 3])
        >>> one_hot_labels = one_hot(labels, num_classes=4)
        >>> print(one_hot_labels)
        [[1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]]
    """
    labels = labels.ravel()
    classes = np.arange(num_classes)
    return (labels[:, np.newaxis] == classes).astype(np.float32)


def top_k(
    x: np.ndarray,
    k: int,
    axis: int = -1,
    largest: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k values and indices along an axis.

    Uses argpartition for O(n) complexity instead of O(n log n) full sort.

    Args:
        x: Input array
        k: Number of top elements to find
        axis: Axis along which to find top-k
        largest: If True, find k largest; if False, find k smallest

    Returns:
        Tuple of (values, indices) for top-k elements

    Example:
        >>> x = np.random.randn(10, 100)
        >>> values, indices = top_k(x, k=5)
        >>> print(values.shape)  # (10, 5)
    """
    if not largest:
        x = -x

    # Use argpartition (O(n)) instead of argsort (O(n log n))
    indices = np.argpartition(x, -k, axis=axis)
    indices = np.take(indices, np.arange(-k, 0), axis=axis)

    # Get values
    values = np.take_along_axis(x, indices, axis=axis)

    # Sort within top-k
    sorted_within = np.argsort(-values, axis=axis)
    indices = np.take_along_axis(indices, sorted_within, axis=axis)
    values = np.take_along_axis(values, sorted_within, axis=axis)

    if not largest:
        values = -values

    return values, indices


def check_contiguous(arr: np.ndarray) -> dict:
    """
    Check memory layout of a NumPy array.

    Contiguous arrays are faster for most operations.

    Args:
        arr: NumPy array to check

    Returns:
        Dictionary with memory layout information

    Example:
        >>> x = np.random.randn(100, 100)
        >>> print(check_contiguous(x))
        {'c_contiguous': True, 'f_contiguous': False, 'writeable': True}
    """
    return {
        'c_contiguous': arr.flags['C_CONTIGUOUS'],
        'f_contiguous': arr.flags['F_CONTIGUOUS'],
        'writeable': arr.flags['WRITEABLE'],
        'aligned': arr.flags['ALIGNED'],
        'shape': arr.shape,
        'strides': arr.strides,
        'dtype': str(arr.dtype),
        'nbytes': arr.nbytes,
        'nbytes_human': _format_bytes(arr.nbytes)
    }


def estimate_memory_usage(
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
    n_arrays: int = 1
) -> str:
    """
    Estimate memory usage for arrays.

    Helpful for planning large computations on DGX Spark.

    Args:
        shape: Array shape
        dtype: Data type
        n_arrays: Number of arrays of this shape

    Returns:
        Human-readable memory estimate

    Example:
        >>> print(estimate_memory_usage((10000, 10000), np.float32, 3))
        "3 arrays of (10000, 10000) float32: 1.12 GB total"
    """
    itemsize = np.dtype(dtype).itemsize
    single_size = np.prod(shape) * itemsize
    total_size = single_size * n_arrays

    single_str = _format_bytes(single_size)
    total_str = _format_bytes(total_size)

    return f"{n_arrays} arrays of {shape} {np.dtype(dtype).name}: {total_str} total ({single_str} each)"


def _format_bytes(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.2f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024**2:.2f} MB"
    else:
        return f"{n_bytes / 1024**3:.2f} GB"


if __name__ == "__main__":
    print("NumPy Utilities Demo")
    print("=" * 50)

    np.random.seed(42)

    # Demo pairwise distances
    print("\n1. Pairwise distances:")
    points = np.random.randn(100, 64).astype(np.float32)
    dists = pairwise_distances(points)
    print(f"   Input: {points.shape}")
    print(f"   Output: {dists.shape}")
    print(f"   Diagonal (self-distances): {dists.diagonal()[:5]}")

    # Demo cosine similarity
    print("\n2. Cosine similarity:")
    sims = cosine_similarity(points)
    print(f"   Output: {sims.shape}")
    print(f"   Diagonal (self-similarity): {sims.diagonal()[:5].round(4)}")

    # Demo softmax
    print("\n3. Softmax:")
    logits = np.random.randn(32, 10)
    probs = softmax(logits)
    print(f"   Input: {logits.shape}")
    print(f"   Probabilities sum: {probs.sum(axis=-1)[:5].round(4)}")

    # Demo attention scores
    print("\n4. Attention scores:")
    Q = np.random.randn(8, 12, 64, 64).astype(np.float32)
    K = np.random.randn(8, 12, 64, 64).astype(np.float32)
    scores = attention_scores(Q, K)
    print(f"   Q shape: {Q.shape}")
    print(f"   K shape: {K.shape}")
    print(f"   Scores shape: {scores.shape}")

    # Demo memory estimation
    print("\n5. Memory estimation:")
    print(f"   {estimate_memory_usage((10000, 10000), np.float32, 3)}")

    print("\n" + "=" * 50)
    print("Demo complete!")
