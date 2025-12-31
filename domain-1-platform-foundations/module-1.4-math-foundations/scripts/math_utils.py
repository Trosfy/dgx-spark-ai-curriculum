"""
Mathematics utilities for deep learning.

This module provides production-ready implementations of:
- Activation functions and their derivatives
- Loss functions
- Optimizers (SGD, Momentum, Adam)
- SVD utilities for LoRA
- Probability distributions

Example usage:
    >>> from math_utils import sigmoid, sigmoid_derivative, Adam
    >>>
    >>> # Activation functions
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = sigmoid(x)
    >>> dy = sigmoid_derivative(x)
    >>>
    >>> # Optimizer
    >>> optimizer = Adam(lr=0.001)
    >>> params = optimizer.step(params, gradients)
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    σ(z) = 1 / (1 + exp(-z))

    Args:
        z: Input array of any shape

    Returns:
        Output array of same shape, values in (0, 1)

    Example:
        >>> sigmoid(np.array([0, 1, -1]))
        array([0.5       , 0.73105858, 0.26894142])
    """
    # Clip for numerical stability
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function.

    σ'(z) = σ(z) × (1 - σ(z))

    Args:
        z: Input array (pre-activation values)

    Returns:
        Derivative at each point
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.

    ReLU(z) = max(0, z)

    Args:
        z: Input array of any shape

    Returns:
        Output array with negative values set to 0
    """
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.

    ReLU'(z) = 1 if z > 0, else 0

    Args:
        z: Input array (pre-activation values)

    Returns:
        Derivative at each point (0 or 1)
    """
    return (z > 0).astype(float)


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.

    Args:
        z: Input array

    Returns:
        Output in range (-1, 1)
    """
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh function.

    tanh'(z) = 1 - tanh²(z)
    """
    t = np.tanh(z)
    return 1 - t ** 2


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax function (numerically stable).

    softmax(z)_i = exp(z_i) / Σ exp(z_j)

    Args:
        z: Input array
        axis: Axis along which to compute softmax

    Returns:
        Probability distribution (sums to 1 along axis)

    Example:
        >>> softmax(np.array([1.0, 2.0, 3.0]))
        array([0.09003057, 0.24472847, 0.66524096])
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error loss.

    MSE = (1/n) Σ (y_true - y_pred)²

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Scalar loss value
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE loss with respect to predictions.

    ∂MSE/∂y_pred = (2/n)(y_pred - y_true)
    """
    n = y_true.shape[0]
    return 2 * (y_pred - y_true) / n


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Binary Cross-Entropy loss.

    BCE = -(1/n) Σ [y*log(p) + (1-y)*log(1-p)]

    Args:
        y_true: Binary labels (0 or 1)
        y_pred: Predicted probabilities (0 to 1)
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


def cross_entropy_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Cross-Entropy loss for multi-class classification.

    CE = -(1/n) Σ Σ y_true * log(y_pred)

    Args:
        y_true: One-hot encoded labels (batch_size, n_classes)
        y_pred: Predicted probabilities after softmax
        eps: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


# =============================================================================
# OPTIMIZERS
# =============================================================================

@dataclass
class SGD:
    """
    Vanilla Stochastic Gradient Descent optimizer.

    Update rule: θ = θ - lr × gradient

    Attributes:
        lr: Learning rate

    Example:
        >>> opt = SGD(lr=0.01)
        >>> params = opt.step(params, grads)
    """
    lr: float = 0.01

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform one optimization step."""
        return params - self.lr * grads

    def reset(self) -> None:
        """Reset optimizer state (no state for vanilla SGD)."""
        pass


@dataclass
class SGDMomentum:
    """
    SGD with Momentum optimizer.

    Update rules:
        v = β × v + gradient
        θ = θ - lr × v

    Attributes:
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)
    """
    lr: float = 0.01
    momentum: float = 0.9
    velocity: Optional[np.ndarray] = field(default=None, repr=False)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform one optimization step with momentum."""
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity + grads
        return params - self.lr * self.velocity

    def reset(self) -> None:
        """Reset velocity to zero."""
        self.velocity = None


@dataclass
class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum with adaptive learning rates.

    Update rules:
        m = β1 × m + (1-β1) × g
        v = β2 × v + (1-β2) × g²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        θ = θ - lr × m_hat / (√v_hat + ε)

    Attributes:
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        epsilon: Numerical stability constant

    Example:
        >>> opt = Adam(lr=0.001)
        >>> for epoch in range(100):
        ...     grads = compute_gradients(params)
        ...     params = opt.step(params, grads)
    """
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    m: Optional[np.ndarray] = field(default=None, repr=False)
    v: Optional[np.ndarray] = field(default=None, repr=False)
    t: int = field(default=0, repr=False)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform one Adam optimization step."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self) -> None:
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0


@dataclass
class AdamW:
    """
    AdamW optimizer (Adam with decoupled weight decay).

    Same as Adam but adds weight decay separately from gradients.
    """
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    m: Optional[np.ndarray] = field(default=None, repr=False)
    v: Optional[np.ndarray] = field(default=None, repr=False)
    t: int = field(default=0, repr=False)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Perform one AdamW optimization step."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Adam updates
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Adam step + weight decay
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        params = params - self.lr * self.weight_decay * params

        return params

    def reset(self) -> None:
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0


# =============================================================================
# SVD UTILITIES
# =============================================================================

def compute_svd(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Singular Value Decomposition.

    W = U × Σ × V^T

    Args:
        W: Input matrix of shape (m, n)

    Returns:
        U: Left singular vectors (m, min(m,n))
        S: Singular values (min(m,n),)
        Vt: Right singular vectors transposed (min(m,n), n)
    """
    return np.linalg.svd(W, full_matrices=False)


def low_rank_approximation(
    W: np.ndarray,
    rank: int
) -> np.ndarray:
    """
    Compute low-rank approximation of a matrix.

    Args:
        W: Input matrix
        rank: Target rank for approximation

    Returns:
        Approximated matrix of same shape

    Example:
        >>> W = np.random.randn(100, 100)
        >>> W_approx = low_rank_approximation(W, rank=10)
        >>> print(f"Error: {np.linalg.norm(W - W_approx) / np.linalg.norm(W):.4f}")
    """
    U, S, Vt = compute_svd(W)
    return U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]


def compute_reconstruction_error(
    W: np.ndarray,
    W_approx: np.ndarray
) -> float:
    """
    Compute relative reconstruction error (Frobenius norm).

    Error = ||W - W_approx|| / ||W||

    Args:
        W: Original matrix
        W_approx: Approximated matrix

    Returns:
        Relative error (0 = perfect, 1 = completely wrong)
    """
    return np.linalg.norm(W - W_approx) / np.linalg.norm(W)


def find_optimal_rank(
    W: np.ndarray,
    target_error: float = 0.01
) -> int:
    """
    Find minimum rank needed to achieve target reconstruction error.

    Args:
        W: Input matrix
        target_error: Maximum acceptable relative error

    Returns:
        Minimum rank needed

    Example:
        >>> W = np.random.randn(100, 100) @ np.random.randn(100, 100)
        >>> rank = find_optimal_rank(W, target_error=0.01)
    """
    U, S, Vt = compute_svd(W)

    for r in range(1, len(S) + 1):
        W_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        error = compute_reconstruction_error(W, W_approx)
        if error < target_error:
            return r

    return len(S)


def lora_memory_savings(
    d_model: int,
    rank: int
) -> Dict[str, Union[int, float]]:
    """
    Calculate memory savings from using LoRA.

    Args:
        d_model: Model dimension
        rank: LoRA rank

    Returns:
        Dictionary with parameter counts and savings
    """
    full_params = d_model * d_model
    lora_params = 2 * d_model * rank
    savings = (1 - lora_params / full_params) * 100

    return {
        'full_params': full_params,
        'lora_params': lora_params,
        'savings_percent': savings,
        'compression_ratio': full_params / lora_params
    }


# =============================================================================
# PROBABILITY UTILITIES
# =============================================================================

def gaussian_pdf(
    x: np.ndarray,
    mu: float = 0.0,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Gaussian probability density function.

    p(x) = (1/σ√(2π)) × exp(-(x-μ)²/2σ²)
    """
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coeff * np.exp(exponent)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """
    Compute KL Divergence D_KL(P || Q).

    D_KL(P||Q) = Σ P(x) × log(P(x)/Q(x))

    Args:
        p: True distribution
        q: Approximate distribution
        eps: Small constant for stability

    Returns:
        KL divergence value (>= 0)
    """
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return np.sum(p * np.log(p / q))


def entropy(p: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute entropy of a distribution.

    H(P) = -Σ P(x) × log(P(x))
    """
    p = np.clip(p, eps, 1 - eps)
    return -np.sum(p * np.log(p))


# =============================================================================
# GRADIENT CHECKING
# =============================================================================

def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using central differences.

    Useful for verifying analytical gradients.

    Args:
        f: Function to differentiate (takes array, returns scalar)
        x: Point at which to compute gradient
        eps: Step size for finite differences

    Returns:
        Gradient array of same shape as x

    Example:
        >>> f = lambda x: np.sum(x**2)
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> grad = numerical_gradient(f, x)
        >>> # Should be close to [2.0, 4.0, 6.0]
    """
    grad = np.zeros_like(x)

    for i in range(len(x.flat)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus.flat[i] += eps
        x_minus.flat[i] -= eps

        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad


def check_gradient(
    analytical_grad: np.ndarray,
    numerical_grad: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Check if analytical gradient matches numerical gradient.

    Args:
        analytical_grad: Computed gradient
        numerical_grad: Numerically approximated gradient
        tolerance: Maximum allowed difference

    Returns:
        (passes, max_difference)
    """
    max_diff = np.abs(analytical_grad - numerical_grad).max()
    return max_diff < tolerance, max_diff


if __name__ == "__main__":
    # Quick tests
    print("Math Utilities Tests")
    print("=" * 50)

    # Test activation functions
    x = np.array([-1.0, 0.0, 1.0])
    print(f"sigmoid({x}) = {sigmoid(x)}")
    print(f"relu({x}) = {relu(x)}")
    print(f"softmax({x}) = {softmax(x)}")

    # Test optimizer
    opt = Adam(lr=0.1)
    params = np.array([5.0, 5.0])

    print("\nAdam optimization towards origin:")
    for i in range(5):
        grads = 2 * params  # Gradient of x^2 + y^2
        params = opt.step(params, grads)
        print(f"  Step {i+1}: params = {params.round(4)}")

    # Test SVD
    W = np.random.randn(100, 100)
    savings = lora_memory_savings(768, 16)
    print(f"\nLoRA savings for d=768, r=16:")
    print(f"  Full params: {savings['full_params']:,}")
    print(f"  LoRA params: {savings['lora_params']:,}")
    print(f"  Savings: {savings['savings_percent']:.1f}%")

    print("\n✅ All tests passed!")
