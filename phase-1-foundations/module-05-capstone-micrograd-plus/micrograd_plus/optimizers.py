"""
Optimizers for MicroGrad+.

This module implements optimization algorithms that update neural network
parameters based on computed gradients to minimize the loss function.

ELI5 Explanation:
    Imagine you're trying to roll a ball to the lowest point in a hilly landscape.
    An optimizer is like your strategy for pushing the ball:
    - SGD: Push the ball in the direction of steepest descent
    - Momentum: Remember which way you've been pushing and keep going that way
    - Adam: Be smart about how hard to push based on recent terrain and history

Example:
    >>> from micrograd_plus import Tensor, SGD, Adam
    >>>
    >>> # Parameters to optimize
    >>> weights = Tensor(np.random.randn(10, 5), requires_grad=True)
    >>> bias = Tensor(np.zeros(5), requires_grad=True)
    >>>
    >>> # Create optimizer
    >>> optimizer = Adam([weights, bias], lr=0.001)
    >>>
    >>> # Training loop
    >>> for epoch in range(100):
    ...     # Forward pass
    ...     loss = compute_loss(weights, bias, data)
    ...
    ...     # Backward pass
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...
    ...     # Update parameters
    ...     optimizer.step()
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from .tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers.

    Provides common functionality for managing parameters and gradients.

    Args:
        params: List of Tensor objects to optimize.
        lr: Learning rate (step size for updates).

    Subclasses should implement:
        - step(): Update parameters using their gradients
    """

    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

        # Validate parameters
        for i, p in enumerate(self.params):
            if not isinstance(p, Tensor):
                raise TypeError(f"Param {i} must be a Tensor, got {type(p)}")
            if not p.requires_grad:
                print(f"Warning: Param {i} has requires_grad=False, it won't be updated")

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters to zero.

        Should be called before each backward pass to prevent gradient accumulation.

        Example:
            >>> optimizer.zero_grad()
            >>> loss.backward()
            >>> optimizer.step()
        """
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def step(self) -> None:
        """
        Update parameters based on gradients.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.lr

    def set_lr(self, lr: float) -> None:
        """Set learning rate."""
        self.lr = lr


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with optional momentum.

    Update rule:
        Without momentum: θ = θ - lr * ∇θ
        With momentum:    v = μ * v + ∇θ
                         θ = θ - lr * v

    ELI5:
        Imagine you're sledding down a hill. Without momentum, you stop and
        recalculate direction at every point. With momentum, you build up
        speed and "coast" through flat areas, which helps you escape
        small bumps (local minima) and reach the bottom faster!

    Args:
        params: List of parameters to optimize.
        lr: Learning rate (default: 0.01).
        momentum: Momentum factor (default: 0). Use 0.9 for most cases.
        weight_decay: L2 regularization factor (default: 0).
        nesterov: Whether to use Nesterov momentum (default: False).

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>>
        >>> # Training step
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(params, lr)

        if momentum < 0:
            raise ValueError(f"Momentum must be >= 0, got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be >= 0, got {weight_decay}")
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Velocity buffers for momentum
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        """
        Perform one optimization step.

        Updates parameters in-place based on their gradients.
        """
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Apply momentum
            if self.momentum != 0:
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(p.data)

                v = self.velocities[i]
                v = self.momentum * v + grad
                self.velocities[i] = v

                if self.nesterov:
                    # Nesterov momentum: look ahead
                    grad = grad + self.momentum * v
                else:
                    grad = v

            # Update parameter
            p.data = p.data - self.lr * grad

    def __repr__(self) -> str:
        return (f"SGD(lr={self.lr}, momentum={self.momentum}, "
                f"weight_decay={self.weight_decay}, nesterov={self.nesterov})")


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Adam combines the ideas of momentum (using past gradients) and RMSprop
    (adapting learning rate per parameter). It maintains both first moment
    (mean) and second moment (variance) estimates of the gradients.

    Update rules:
        m = β₁ * m + (1 - β₁) * g          (first moment estimate)
        v = β₂ * v + (1 - β₂) * g²         (second moment estimate)
        m̂ = m / (1 - β₁ᵗ)                   (bias correction)
        v̂ = v / (1 - β₂ᵗ)                   (bias correction)
        θ = θ - lr * m̂ / (√v̂ + ε)          (parameter update)

    ELI5:
        Imagine you're learning to throw darts:
        - The "first moment" (m) tracks which direction you're usually missing
        - The "second moment" (v) tracks how consistent your throws are
        Adam uses both to adjust your aim: it corrects your average direction
        while being careful about dimensions where you're inconsistent.

    Args:
        params: List of parameters to optimize.
        lr: Learning rate (default: 0.001).
        betas: Coefficients for computing running averages (default: (0.9, 0.999)).
        eps: Small constant for numerical stability (default: 1e-8).
        weight_decay: L2 regularization factor (default: 0).

    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>>
        >>> for epoch in range(num_epochs):
        ...     optimizer.zero_grad()
        ...     loss = compute_loss()
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta1 must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta2 must be in [0, 1), got {betas[1]}")
        if eps <= 0:
            raise ValueError(f"Epsilon must be > 0, got {eps}")

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # State: first moment (m), second moment (v), and timestep
        self.m: Dict[int, np.ndarray] = {}  # First moment
        self.v: Dict[int, np.ndarray] = {}  # Second moment
        self.t = 0  # Timestep

    def step(self) -> None:
        """
        Perform one optimization step.

        Updates parameters using Adam algorithm.
        """
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Apply weight decay (decoupled, like AdamW)
            if self.weight_decay != 0:
                p.data = p.data - self.lr * self.weight_decay * p.data

            # Initialize moment buffers if needed
            if i not in self.m:
                self.m[i] = np.zeros_like(p.data)
                self.v[i] = np.zeros_like(p.data)

            m = self.m[i]
            v = self.v[i]

            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            self.m[i] = m

            # Update biased second moment estimate
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            self.v[i] = v

            # Bias correction
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)

            # Update parameter
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self) -> str:
        return (f"Adam(lr={self.lr}, betas={self.betas}, "
                f"eps={self.eps}, weight_decay={self.weight_decay})")


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    This is a variant of Adam that applies weight decay correctly by
    decoupling it from the gradient update. This often leads to better
    generalization than standard Adam with L2 regularization.

    The key difference from Adam:
        - Adam: adds weight_decay * param to gradient before moment updates
        - AdamW: subtracts lr * weight_decay * param directly from parameters

    Args:
        params: List of parameters to optimize.
        lr: Learning rate (default: 0.001).
        betas: Coefficients for running averages (default: (0.9, 0.999)).
        eps: Small constant for numerical stability (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 0.01).

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        super().__init__(params, lr)

        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self) -> None:
        """Perform one optimization step."""
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Initialize moment buffers if needed
            if i not in self.m:
                self.m[i] = np.zeros_like(p.data)
                self.v[i] = np.zeros_like(p.data)

            m = self.m[i]
            v = self.v[i]

            # Update moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            self.m[i] = m
            self.v[i] = v

            # Bias correction
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)

            # Update with decoupled weight decay
            p.data = p.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)

    def __repr__(self) -> str:
        return (f"AdamW(lr={self.lr}, betas={self.betas}, "
                f"eps={self.eps}, weight_decay={self.weight_decay})")


class RMSprop(Optimizer):
    """
    RMSprop optimizer.

    RMSprop adapts the learning rate for each parameter based on the recent
    history of gradients. It divides the learning rate by a running average
    of recent gradient magnitudes.

    Update rules:
        v = α * v + (1 - α) * g²
        θ = θ - lr * g / (√v + ε)

    ELI5:
        Some dimensions might have huge gradients, others tiny ones.
        RMSprop keeps a running average of how big gradients are for each
        dimension, then divides by that. This way, dimensions with big
        gradients take smaller steps, and dimensions with small gradients
        take bigger steps. Everyone moves at a similar pace!

    Args:
        params: List of parameters to optimize.
        lr: Learning rate (default: 0.01).
        alpha: Smoothing constant (default: 0.99).
        eps: Small constant for numerical stability (default: 1e-8).
        weight_decay: L2 regularization factor (default: 0).
        momentum: Momentum factor (default: 0).

    Example:
        >>> optimizer = RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0
    ):
        super().__init__(params, lr)

        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.v: Dict[int, np.ndarray] = {}  # Running average of squared gradients
        self.buffer: Dict[int, np.ndarray] = {}  # Momentum buffer

    def step(self) -> None:
        """Perform one optimization step."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.copy()

            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Initialize running average
            if i not in self.v:
                self.v[i] = np.zeros_like(p.data)
                if self.momentum > 0:
                    self.buffer[i] = np.zeros_like(p.data)

            v = self.v[i]

            # Update running average of squared gradients
            v = self.alpha * v + (1 - self.alpha) * (grad ** 2)
            self.v[i] = v

            # Compute update
            if self.momentum > 0:
                buf = self.buffer[i]
                buf = self.momentum * buf + grad / (np.sqrt(v) + self.eps)
                self.buffer[i] = buf
                p.data = p.data - self.lr * buf
            else:
                p.data = p.data - self.lr * grad / (np.sqrt(v) + self.eps)

    def __repr__(self) -> str:
        return (f"RMSprop(lr={self.lr}, alpha={self.alpha}, "
                f"eps={self.eps}, weight_decay={self.weight_decay}, momentum={self.momentum})")


class LRScheduler:
    """
    Base class for learning rate schedulers.

    Schedulers adjust the learning rate during training, which can help
    convergence and final performance.

    Args:
        optimizer: The optimizer whose learning rate to adjust.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.step_count = 0

    def step(self) -> None:
        """Update the learning rate."""
        raise NotImplementedError

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


class StepLR(LRScheduler):
    """
    Decay learning rate by gamma every step_size epochs.

    Args:
        optimizer: The optimizer.
        step_size: Period of learning rate decay.
        gamma: Multiplicative factor of learning rate decay (default: 0.1).

    Example:
        >>> scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        >>> for epoch in range(100):
        ...     train()
        ...     scheduler.step()  # lr *= 0.1 at epochs 10, 20, 30, ...
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        """Decay learning rate if at step boundary."""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    """
    Decay learning rate by gamma every epoch.

    Args:
        optimizer: The optimizer.
        gamma: Multiplicative factor of learning rate decay.

    Example:
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> # lr at epoch n = initial_lr * 0.95^n
    """

    def __init__(self, optimizer: Optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        """Decay learning rate."""
        self.step_count += 1
        self.optimizer.lr = self.initial_lr * (self.gamma ** self.step_count)


class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler.

    Learning rate follows a cosine curve from initial_lr to min_lr.

    Args:
        optimizer: The optimizer.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate (default: 0).

    Example:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
        >>> # lr follows cosine curve over 100 epochs
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self) -> None:
        """Update learning rate following cosine curve."""
        self.step_count += 1
        self.optimizer.lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                            (1 + np.cos(np.pi * self.step_count / self.T_max)) / 2
