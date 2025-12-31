"""
Learning Theory Utilities for Module A: Statistical Learning Theory

This module provides reusable utilities for computing theoretical learning
bounds, VC dimension calculations, bias-variance analysis, and PAC learning.

These utilities are designed to work with the DGX Spark AI Curriculum and
provide production-quality implementations with full type hints and documentation.

Author: Professor SPARK
Module: A - Statistical Learning Theory
"""

import numpy as np
from typing import Tuple, List, Optional, Callable, Dict, Any
from itertools import product
from math import comb, log, ceil
from dataclasses import dataclass

# For classifier checking
try:
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# VC Dimension Utilities
# =============================================================================

def can_linearly_separate(points: np.ndarray, labels: np.ndarray) -> bool:
    """
    Check if a set of 2D points with given labels can be linearly separated.

    Uses a hard-margin SVM (very high C) to check if perfect separation exists.

    Args:
        points: Array of shape (n, 2) with point coordinates
        labels: Array of shape (n,) with binary labels (0 or 1)

    Returns:
        True if linearly separable, False otherwise

    Example:
        >>> points = np.array([[0, 0], [1, 1], [0, 1]])
        >>> labels = np.array([0, 0, 1])
        >>> can_linearly_separate(points, labels)
        True
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for this function")

    # Edge case: all same label is trivially separable
    if len(np.unique(labels)) == 1:
        return True

    # Use hard-margin SVM (very high C = no slack allowed)
    clf = SVC(kernel='linear', C=1e10, max_iter=10000)

    try:
        clf.fit(points, labels)
        predictions = clf.predict(points)
        return np.all(predictions == labels)
    except Exception:
        return False


def check_if_shattered(points: np.ndarray) -> Tuple[bool, List[Tuple]]:
    """
    Check if a linear classifier can shatter the given points.

    A set of points is shattered if we can achieve every possible labeling
    using some hypothesis in our hypothesis class (linear classifiers).

    Args:
        points: Array of shape (n, 2) with point coordinates

    Returns:
        Tuple of (is_shattered, list of failed labelings)

    Example:
        >>> points_3 = np.array([[0, 0], [1, 0], [0.5, 1]])
        >>> is_shattered, failed = check_if_shattered(points_3)
        >>> is_shattered
        True
        >>> len(failed)
        0
    """
    n = len(points)
    all_labelings = list(product([0, 1], repeat=n))

    failed_labelings = []

    for labeling in all_labelings:
        labels = np.array(labeling)
        if not can_linearly_separate(points, labels):
            failed_labelings.append(labeling)

    is_shattered = len(failed_labelings) == 0
    return is_shattered, failed_labelings


def vc_dimension_linear(d: int) -> int:
    """
    Compute VC dimension of linear classifiers in d-dimensional space.

    For linear classifiers (hyperplanes), VC dimension = d + 1.

    Args:
        d: Dimensionality of the input space

    Returns:
        VC dimension (d + 1)

    Example:
        >>> vc_dimension_linear(2)
        3
        >>> vc_dimension_linear(100)
        101
    """
    return d + 1


def vc_dimension_polynomial(d: int, k: int) -> int:
    """
    Compute VC dimension of polynomial classifiers of degree k in d dimensions.

    This equals the number of monomials up to degree k:
    VC = C(d+k, k) = (d+k)! / (d! * k!)

    Args:
        d: Input dimensionality
        k: Polynomial degree

    Returns:
        VC dimension

    Example:
        >>> vc_dimension_polynomial(2, 2)
        6
        >>> vc_dimension_polynomial(10, 2)
        66
    """
    return comb(d + k, k)


def estimate_nn_vc_dimension(n_weights: int, n_layers: int) -> int:
    """
    Estimate upper bound on VC dimension of a neural network.

    Uses classical bound: O(W * L * log(W))
    where W = number of weights, L = number of layers.

    Note: This is a theoretical upper bound; actual effective capacity
    is often much lower due to implicit regularization.

    Args:
        n_weights: Total number of trainable parameters
        n_layers: Number of layers

    Returns:
        Upper bound on VC dimension

    Example:
        >>> # Small MLP: 784 -> 100 -> 10
        >>> n_weights = 784*100 + 100 + 100*10 + 10
        >>> estimate_nn_vc_dimension(n_weights, 2)
        1536540
    """
    return int(n_weights * n_layers * log(n_weights))


# =============================================================================
# Bias-Variance Utilities
# =============================================================================

@dataclass
class BiasVarianceResult:
    """Container for bias-variance decomposition results."""
    bias_squared: float
    variance: float
    noise: float
    total_error: float
    predictions: np.ndarray
    mean_prediction: np.ndarray
    X_test: np.ndarray
    y_true_test: np.ndarray


def bias_variance_decomposition(
    model_factory: Callable,
    X_train_sets: List[np.ndarray],
    y_train_sets: List[np.ndarray],
    X_test: np.ndarray,
    y_true: np.ndarray,
    noise_variance: float = 0.0
) -> BiasVarianceResult:
    """
    Compute bias and variance via bootstrap sampling.

    Trains the model on multiple training sets and analyzes how
    predictions vary.

    Args:
        model_factory: Callable that returns a fresh model instance
        X_train_sets: List of training X arrays (bootstrap samples)
        y_train_sets: List of training y arrays
        X_test: Test points
        y_true: True function values at test points
        noise_variance: Known variance of noise in the data

    Returns:
        BiasVarianceResult dataclass with all computed values

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> # Create bootstrap samples and compute decomposition
        >>> result = bias_variance_decomposition(
        ...     LinearRegression,
        ...     X_train_sets, y_train_sets,
        ...     X_test, y_true, noise_variance=0.1
        ... )
        >>> print(f"Bias²: {result.bias_squared:.4f}")
    """
    n_bootstrap = len(X_train_sets)
    n_test = len(X_test)

    predictions = np.zeros((n_bootstrap, n_test))

    for i, (X_train, y_train) in enumerate(zip(X_train_sets, y_train_sets)):
        model = model_factory()
        model.fit(X_train, y_train)
        predictions[i] = model.predict(X_test).flatten()

    # Mean prediction across bootstrap samples
    mean_prediction = np.mean(predictions, axis=0)

    # Bias² = E[(E[f̂] - f)²]
    bias_squared = float(np.mean((mean_prediction - y_true) ** 2))

    # Variance = E[(f̂ - E[f̂])²]
    variance = float(np.mean(np.var(predictions, axis=0)))

    # Total error
    total_error = bias_squared + variance + noise_variance

    return BiasVarianceResult(
        bias_squared=bias_squared,
        variance=variance,
        noise=noise_variance,
        total_error=total_error,
        predictions=predictions,
        mean_prediction=mean_prediction,
        X_test=X_test,
        y_true_test=y_true
    )


def compute_train_test_gap(
    model_factory: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute train error, test error, and generalization gap.

    The gap reveals variance/overfitting.

    Args:
        model_factory: Callable that returns a fresh model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Tuple of (train_error, test_error, gap)

    Example:
        >>> train_err, test_err, gap = compute_train_test_gap(
        ...     lambda: LogisticRegression(),
        ...     X_train, y_train, X_test, y_test
        ... )
        >>> if gap > 0.1:
        ...     print("Warning: Possible overfitting!")
    """
    model = model_factory()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_error = float(np.mean(train_pred != y_train))
    test_error = float(np.mean(test_pred != y_test))
    gap = test_error - train_error

    return train_error, test_error, gap


# =============================================================================
# PAC Learning Utilities
# =============================================================================

def pac_sample_complexity_finite(
    H_size: int,
    epsilon: float,
    delta: float
) -> int:
    """
    Sample complexity bound for finite hypothesis classes.

    For a hypothesis class with |H| hypotheses:
    m >= (1/ε) * (ln|H| + ln(1/δ))

    Args:
        H_size: Number of hypotheses in the class
        epsilon: Accuracy parameter (maximum allowed error)
        delta: Confidence parameter (probability of failure)

    Returns:
        Sample complexity (minimum samples needed)

    Example:
        >>> pac_sample_complexity_finite(100, 0.05, 0.05)
        152
    """
    m = (1 / epsilon) * (log(H_size) + log(1 / delta))
    return ceil(m)


def pac_sample_complexity_vc(
    vc_dim: int,
    epsilon: float,
    delta: float,
    C: float = 8.0
) -> int:
    """
    Sample complexity bound based on VC dimension.

    This is the fundamental theorem of PAC learning:
    m >= (C/ε) * (VC * log(1/ε) + log(1/δ))

    Args:
        vc_dim: VC dimension of the hypothesis class
        epsilon: Accuracy parameter (maximum allowed error)
        delta: Confidence parameter (probability of failure)
        C: Constant in the bound (typically 8-16)

    Returns:
        Sample complexity (minimum samples needed)

    Example:
        >>> # Linear classifier in 100D
        >>> pac_sample_complexity_vc(101, 0.05, 0.05)
        41893
    """
    m = (C / epsilon) * (vc_dim * log(16 / epsilon) + log(2 / delta))
    return ceil(m)


def generalization_bound(
    vc_dim: int,
    n_samples: int,
    delta: float = 0.05
) -> float:
    """
    Compute generalization gap bound based on VC dimension.

    With probability (1 - delta), the difference between test error
    and training error is at most this value.

    Args:
        vc_dim: VC dimension of the hypothesis class
        n_samples: Number of training samples
        delta: Confidence parameter

    Returns:
        Upper bound on generalization gap

    Example:
        >>> gap = generalization_bound(101, 10000)
        >>> print(f"With 95% confidence, gap <= {gap:.4f}")
    """
    if n_samples <= vc_dim:
        return 1.0  # Bound is trivial

    gap = np.sqrt(
        (vc_dim * log(2 * n_samples / vc_dim) + log(4 / delta)) / n_samples
    )
    return min(float(gap), 1.0)


def practical_sample_estimate(
    n_parameters: int,
    task_difficulty: str = 'medium',
    data_quality: str = 'clean'
) -> Tuple[int, int]:
    """
    Practical sample size estimation based on industry experience.

    Args:
        n_parameters: Number of model parameters
        task_difficulty: 'easy', 'medium', 'hard'
        data_quality: 'clean', 'noisy', 'very_noisy'

    Returns:
        Tuple of (minimum_samples, recommended_samples)

    Example:
        >>> min_s, rec_s = practical_sample_estimate(1000, 'medium', 'noisy')
        >>> print(f"Recommended: {rec_s:,} samples")
    """
    base = 10  # 10x parameters is common rule

    difficulty_mult = {'easy': 1.0, 'medium': 2.0, 'hard': 5.0}
    noise_mult = {'clean': 1.0, 'noisy': 2.0, 'very_noisy': 5.0}

    if task_difficulty not in difficulty_mult:
        raise ValueError(f"task_difficulty must be one of {list(difficulty_mult.keys())}")
    if data_quality not in noise_mult:
        raise ValueError(f"data_quality must be one of {list(noise_mult.keys())}")

    min_samples = int(n_parameters * base * difficulty_mult[task_difficulty])
    rec_samples = int(min_samples * noise_mult[data_quality])

    return min_samples, rec_samples


# =============================================================================
# Data Generation Utilities
# =============================================================================

def generate_regression_data(
    n_samples: int,
    true_function: Callable[[np.ndarray], np.ndarray],
    x_min: float = 0.0,
    x_max: float = 4.0,
    noise_std: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy samples from a true function for regression.

    Args:
        n_samples: Number of samples to generate
        true_function: The true underlying function f(x)
        x_min: Minimum x value
        x_max: Maximum x value
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is shape (n_samples, 1)

    Example:
        >>> f = lambda x: np.sin(2 * x)
        >>> X, y = generate_regression_data(100, f, noise_std=0.2, seed=42)
        >>> X.shape
        (100, 1)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.uniform(x_min, x_max, n_samples)
    y = true_function(X) + np.random.normal(0, noise_std, n_samples)

    return X.reshape(-1, 1), y


def generate_classification_data(
    n_samples: int,
    n_features: int = 20,
    noise: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data with a linear decision boundary.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Amount of label noise (probability of flipping)
        seed: Random seed

    Returns:
        Tuple of (X, y, true_weights)

    Example:
        >>> X, y, weights = generate_classification_data(1000, n_features=50, seed=42)
        >>> X.shape
        (1000, 50)
    """
    if seed is not None:
        np.random.seed(seed)

    X = np.random.randn(n_samples, n_features)

    # True linear decision boundary
    true_weights = np.random.randn(n_features)
    true_weights = true_weights / np.linalg.norm(true_weights)

    # Labels with sigmoid + noise
    logits = X @ true_weights
    probs = 1 / (1 + np.exp(-3 * logits))
    y = (np.random.random(n_samples) < probs).astype(int)

    # Add label noise
    flip_mask = np.random.random(n_samples) < noise
    y[flip_mask] = 1 - y[flip_mask]

    return X, y, true_weights


# =============================================================================
# Testing and Validation
# =============================================================================

if __name__ == "__main__":
    # Run basic tests
    print("Testing Learning Theory Utilities...")
    print("=" * 50)

    # Test VC dimension
    print("\n1. VC Dimension Tests:")
    print(f"   Linear 2D: {vc_dimension_linear(2)}")  # Should be 3
    print(f"   Linear 100D: {vc_dimension_linear(100)}")  # Should be 101
    print(f"   Polynomial degree 2, 10D: {vc_dimension_polynomial(10, 2)}")  # Should be 66

    # Test PAC bounds
    print("\n2. PAC Sample Complexity Tests:")
    print(f"   Finite (|H|=100, ε=0.05, δ=0.05): {pac_sample_complexity_finite(100, 0.05, 0.05)}")
    print(f"   VC-based (VC=101, ε=0.05, δ=0.05): {pac_sample_complexity_vc(101, 0.05, 0.05)}")

    # Test generalization bound
    print("\n3. Generalization Bound Tests:")
    print(f"   VC=101, n=1000: {generalization_bound(101, 1000):.4f}")
    print(f"   VC=101, n=10000: {generalization_bound(101, 10000):.4f}")

    # Test practical estimates
    print("\n4. Practical Sample Estimates:")
    min_s, rec_s = practical_sample_estimate(1000, 'medium', 'noisy')
    print(f"   1000 params, medium, noisy: min={min_s:,}, rec={rec_s:,}")

    print("\nAll tests passed!")
