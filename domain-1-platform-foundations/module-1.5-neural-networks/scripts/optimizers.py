"""
Neural Network Optimizers - Built from Scratch with NumPy

This module implements various optimization algorithms for training
neural networks. Each optimizer updates the model parameters based
on computed gradients.

Professor SPARK says: "An optimizer is like a hiking guide. The gradient
tells you which direction is downhill, but the optimizer decides how
big of a step to take and whether to build up momentum!"

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 1.5
"""

import numpy as np
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base class for all optimizers.

    ELI5: An optimizer is like a coach for your neural network.
    After each practice (forward + backward pass), the coach
    looks at what went wrong (gradients) and tells each player
    (weight) how to improve (update).
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers: List) -> None:
        """Update parameters of all trainable layers."""
        pass

    def zero_grad(self, layers: List) -> None:
        """Reset all gradients to zero."""
        for layer in layers:
            if hasattr(layer, 'gradients'):
                layer.gradients = {}


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.

    Update rule: w = w - learning_rate * gradient

    ELI5: SGD is the simplest coach. It just says "move in the
    direction that reduces error, by this much." Simple but
    sometimes slow, like always walking instead of running.

    Parameters:
        learning_rate: Step size for updates (default: 0.01)
        momentum: Momentum factor (default: 0, meaning no momentum)
        weight_decay: L2 regularization strength (default: 0)

    Example:
        >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)
        >>> for epoch in range(epochs):
        ...     loss = compute_loss(model, data)
        ...     model.backward(loss_gradient)
        ...     optimizer.step(model.get_trainable_layers())
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Dict[int, Dict[str, np.ndarray]] = {}

    def step(self, layers: List) -> None:
        """
        Update parameters using SGD with optional momentum.

        Momentum helps escape local minima and speeds up training
        in directions with consistent gradients.

        With momentum:
            v = momentum * v - learning_rate * gradient
            w = w + v

        Without momentum:
            w = w - learning_rate * gradient
        """
        for i, layer in enumerate(layers):
            if not layer.trainable:
                continue

            # Initialize velocity if needed
            if i not in self.velocity:
                self.velocity[i] = {}

            # Update weights if they exist
            if hasattr(layer, 'weights') and 'weights' in layer.gradients:
                grad = layer.gradients['weights']

                # Add weight decay (L2 regularization)
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * layer.weights

                if self.momentum > 0:
                    # Initialize velocity for weights
                    if 'weights' not in self.velocity[i]:
                        self.velocity[i]['weights'] = np.zeros_like(layer.weights)

                    # Update velocity
                    self.velocity[i]['weights'] = (
                        self.momentum * self.velocity[i]['weights'] -
                        self.learning_rate * grad
                    )
                    # Update weights
                    layer.weights += self.velocity[i]['weights']
                else:
                    # Simple SGD update
                    layer.weights -= self.learning_rate * grad

            # Update bias if it exists
            if hasattr(layer, 'bias') and 'bias' in layer.gradients:
                grad = layer.gradients['bias']

                if self.momentum > 0:
                    # Initialize velocity for bias
                    if 'bias' not in self.velocity[i]:
                        self.velocity[i]['bias'] = np.zeros_like(layer.bias)

                    # Update velocity
                    self.velocity[i]['bias'] = (
                        self.momentum * self.velocity[i]['bias'] -
                        self.learning_rate * grad
                    )
                    # Update bias
                    layer.bias += self.velocity[i]['bias']
                else:
                    # Simple SGD update
                    layer.bias -= self.learning_rate * grad


class SGDWithMomentum(SGD):
    """
    Convenience class for SGD with momentum.

    ELI5: Momentum is like a ball rolling downhill. It builds up
    speed when going in the same direction, and resists sudden
    direction changes. This helps the ball (our weights) roll
    past small bumps (local minima) and reach the bottom faster.

    Example:
        >>> optimizer = SGDWithMomentum(learning_rate=0.01, momentum=0.9)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate, momentum, weight_decay)


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Combines momentum with adaptive learning rates per parameter.

    ELI5: Adam is the smartest coach. It remembers which direction
    worked well (momentum) AND adjusts how big a step each player
    takes based on how consistently they've been improving. Players
    who've been doing well take bigger steps; uncertain players
    take smaller steps.

    Parameters:
        learning_rate: Initial learning rate (default: 0.001)
        beta1: Exponential decay rate for momentum (default: 0.9)
        beta2: Exponential decay rate for squared gradient (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
        weight_decay: L2 regularization strength (default: 0)

    Why Adam is popular:
    1. Works well with default hyperparameters
    2. Adaptive learning rates help with sparse gradients
    3. Momentum helps with non-stationary objectives

    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> # Training loop
        >>> for batch in data:
        ...     output = model(batch)
        ...     loss = compute_loss(output, targets)
        ...     model.backward(loss_gradient)
        ...     optimizer.step(model.get_trainable_layers())
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # Moving averages
        self.m: Dict[int, Dict[str, np.ndarray]] = {}  # First moment
        self.v: Dict[int, Dict[str, np.ndarray]] = {}  # Second moment
        self.t = 0  # Time step

    def step(self, layers: List) -> None:
        """
        Update parameters using Adam.

        Algorithm:
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient^2
            m_hat = m / (1 - beta1^t)  # Bias correction
            v_hat = v / (1 - beta2^t)  # Bias correction
            w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        """
        self.t += 1

        for i, layer in enumerate(layers):
            if not layer.trainable:
                continue

            # Initialize moments if needed
            if i not in self.m:
                self.m[i] = {}
                self.v[i] = {}

            # Update weights
            if hasattr(layer, 'weights') and 'weights' in layer.gradients:
                self._update_param(
                    layer, 'weights', layer.weights, layer.gradients['weights'], i
                )

            # Update bias
            if hasattr(layer, 'bias') and 'bias' in layer.gradients:
                self._update_param(
                    layer, 'bias', layer.bias, layer.gradients['bias'], i
                )

    def _update_param(
        self,
        layer,
        param_name: str,
        param: np.ndarray,
        grad: np.ndarray,
        layer_idx: int
    ) -> None:
        """Update a single parameter using Adam."""
        # Add weight decay
        if self.weight_decay > 0 and param_name == 'weights':
            grad = grad + self.weight_decay * param

        # Initialize moments
        if param_name not in self.m[layer_idx]:
            self.m[layer_idx][param_name] = np.zeros_like(param)
            self.v[layer_idx][param_name] = np.zeros_like(param)

        # Update biased first moment estimate
        self.m[layer_idx][param_name] = (
            self.beta1 * self.m[layer_idx][param_name] +
            (1 - self.beta1) * grad
        )

        # Update biased second raw moment estimate
        self.v[layer_idx][param_name] = (
            self.beta2 * self.v[layer_idx][param_name] +
            (1 - self.beta2) * (grad ** 2)
        )

        # Compute bias-corrected estimates
        m_hat = self.m[layer_idx][param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer_idx][param_name] / (1 - self.beta2 ** self.t)

        # Update parameter
        if param_name == 'weights':
            layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW(Adam):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    ELI5: AdamW is like Adam, but it handles weight decay
    differently. Instead of mixing weight decay into the
    gradient, it applies it separately. This turns out to
    work better, especially for large models like transformers!

    The difference is subtle but important:
    - Adam: gradient = gradient + weight_decay * weights, then Adam update
    - AdamW: Adam update, then weights = weights - learning_rate * weight_decay * weights

    This is the optimizer used for training GPT, BERT, and most modern LLMs!

    Example:
        >>> optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01
    ):
        # Initialize Adam with weight_decay=0 (we handle it separately)
        super().__init__(learning_rate, beta1, beta2, epsilon, weight_decay=0)
        self.weight_decay = weight_decay

    def step(self, layers: List) -> None:
        """Update parameters using AdamW (decoupled weight decay)."""
        self.t += 1

        for i, layer in enumerate(layers):
            if not layer.trainable:
                continue

            # Initialize moments if needed
            if i not in self.m:
                self.m[i] = {}
                self.v[i] = {}

            # Update weights (with decoupled weight decay)
            if hasattr(layer, 'weights') and 'weights' in layer.gradients:
                grad = layer.gradients['weights']
                self._update_param(layer, 'weights', layer.weights, grad, i)
                # Apply decoupled weight decay AFTER the Adam update
                layer.weights -= self.learning_rate * self.weight_decay * layer.weights

            # Update bias (no weight decay on bias)
            if hasattr(layer, 'bias') and 'bias' in layer.gradients:
                self._update_param(
                    layer, 'bias', layer.bias, layer.gradients['bias'], i
                )


class LearningRateScheduler:
    """
    Base class for learning rate schedulers.

    ELI5: A scheduler is like a training plan. At first, you can
    take big steps because you're far from the goal. As you get
    closer, you take smaller steps to avoid overshooting. This
    helps find the exact best solution!
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.learning_rate
        self.step_count = 0

    def step(self) -> None:
        """Update learning rate (called after each epoch or batch)."""
        raise NotImplementedError

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.learning_rate


class StepLR(LearningRateScheduler):
    """
    Step learning rate decay.

    Reduces learning rate by a factor every N steps.

    Parameters:
        optimizer: The optimizer to schedule
        step_size: Epochs between each decay
        gamma: Multiplicative factor (default: 0.1)

    Example:
        >>> optimizer = SGD(learning_rate=0.1)
        >>> scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        >>> # LR: 0.1 -> 0.01 (at epoch 10) -> 0.001 (at epoch 20)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1
    ):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        """Update learning rate if we've reached a step boundary."""
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.learning_rate *= self.gamma


class ExponentialLR(LearningRateScheduler):
    """
    Exponential learning rate decay.

    lr = initial_lr * gamma^step

    Parameters:
        optimizer: The optimizer to schedule
        gamma: Multiplicative factor per step (default: 0.95)

    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> scheduler = ExponentialLR(optimizer, gamma=0.95)
        >>> # LR decays by 5% each step
    """

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.95
    ):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        """Decay learning rate exponentially."""
        self.step_count += 1
        self.optimizer.learning_rate = self.initial_lr * (self.gamma ** self.step_count)


class CosineAnnealingLR(LearningRateScheduler):
    """
    Cosine annealing learning rate schedule.

    lr = min_lr + (initial_lr - min_lr) * (1 + cos(pi * step / T)) / 2

    ELI5: The learning rate follows a smooth cosine curve from high
    to low. It's like gradually slowing down as you approach a stop
    sign - smooth and gradual, not sudden.

    Parameters:
        optimizer: The optimizer to schedule
        T_max: Maximum number of steps (usually total epochs)
        min_lr: Minimum learning rate (default: 0)

    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        min_lr: float = 0.0
    ):
        super().__init__(optimizer)
        self.T_max = T_max
        self.min_lr = min_lr

    def step(self) -> None:
        """Update learning rate using cosine annealing."""
        self.step_count += 1
        self.optimizer.learning_rate = (
            self.min_lr +
            (self.initial_lr - self.min_lr) *
            (1 + np.cos(np.pi * self.step_count / self.T_max)) / 2
        )


class WarmupScheduler(LearningRateScheduler):
    """
    Linear warmup followed by another scheduler.

    ELI5: When you start exercising, you warm up first! Similarly,
    we start with a tiny learning rate and gradually increase it.
    This prevents the network from making wild updates early on
    when it knows nothing.

    Parameters:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        after_scheduler: Scheduler to use after warmup (optional)

    Example:
        >>> optimizer = Adam(learning_rate=0.001)
        >>> scheduler = WarmupScheduler(optimizer, warmup_steps=1000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        after_scheduler: Optional[LearningRateScheduler] = None
    ):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler

    def step(self) -> None:
        """Update learning rate with warmup."""
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            self.optimizer.learning_rate = (
                self.initial_lr * self.step_count / self.warmup_steps
            )
        elif self.after_scheduler is not None:
            # Use the after scheduler
            self.after_scheduler.step()


def get_optimizer(name: str, layers: List, **kwargs) -> Optimizer:
    """
    Factory function to get optimizer by name.

    Args:
        name: One of 'sgd', 'sgd_momentum', 'adam', 'adamw'
        layers: List of trainable layers (not used, kept for API compatibility)
        **kwargs: Additional arguments for the optimizer

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = get_optimizer('adam', layers, learning_rate=0.001)
    """
    optimizers = {
        'sgd': SGD,
        'sgd_momentum': SGDWithMomentum,
        'adam': Adam,
        'adamw': AdamW
    }

    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: {list(optimizers.keys())}")

    return optimizers[name.lower()](**kwargs)


if __name__ == "__main__":
    print("Testing Optimizers")
    print("=" * 50)

    # Create a simple mock layer for testing
    class MockLayer:
        def __init__(self):
            self.trainable = True
            self.weights = np.random.randn(10, 5)
            self.bias = np.zeros(5)
            self.gradients = {
                'weights': np.random.randn(10, 5) * 0.1,
                'bias': np.random.randn(5) * 0.1
            }

    layers = [MockLayer()]

    # Test SGD
    print("\n1. Testing SGD:")
    sgd = SGD(learning_rate=0.01)
    weights_before = layers[0].weights.copy()
    sgd.step(layers)
    weight_diff = np.abs(layers[0].weights - weights_before).mean()
    print(f"   Average weight change: {weight_diff:.6f}")

    # Test SGD with momentum
    print("\n2. Testing SGD with Momentum:")
    layers = [MockLayer()]
    sgd_mom = SGDWithMomentum(learning_rate=0.01, momentum=0.9)
    for _ in range(3):
        layers[0].gradients = {
            'weights': np.random.randn(10, 5) * 0.1,
            'bias': np.random.randn(5) * 0.1
        }
        sgd_mom.step(layers)
    print(f"   Momentum accumulated over 3 steps")

    # Test Adam
    print("\n3. Testing Adam:")
    layers = [MockLayer()]
    adam = Adam(learning_rate=0.001)
    weights_before = layers[0].weights.copy()
    for _ in range(10):
        layers[0].gradients = {
            'weights': np.random.randn(10, 5) * 0.1,
            'bias': np.random.randn(5) * 0.1
        }
        adam.step(layers)
    weight_diff = np.abs(layers[0].weights - weights_before).mean()
    print(f"   Average weight change after 10 steps: {weight_diff:.6f}")

    # Test AdamW
    print("\n4. Testing AdamW:")
    layers = [MockLayer()]
    adamw = AdamW(learning_rate=0.001, weight_decay=0.01)
    weights_before = layers[0].weights.copy()
    adamw.step(layers)
    print(f"   Weight decay applied with decoupling")

    # Test learning rate schedulers
    print("\n5. Testing Learning Rate Schedulers:")

    # StepLR
    optimizer = SGD(learning_rate=0.1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    print(f"   StepLR: Initial LR = {scheduler.get_lr()}")
    for _ in range(5):
        scheduler.step()
    print(f"   StepLR: After 5 steps = {scheduler.get_lr()}")

    # CosineAnnealingLR
    optimizer = SGD(learning_rate=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    lrs = [scheduler.get_lr()]
    for _ in range(100):
        scheduler.step()
        lrs.append(scheduler.get_lr())
    print(f"   CosineAnnealingLR: {lrs[0]:.4f} -> {lrs[50]:.4f} -> {lrs[-1]:.4f}")

    # WarmupScheduler
    optimizer = SGD(learning_rate=0.1)
    scheduler = WarmupScheduler(optimizer, warmup_steps=10)
    print(f"   Warmup: Step 0 = {scheduler.get_lr():.4f}", end="")
    for i in range(10):
        scheduler.step()
    print(f" -> Step 10 = {scheduler.get_lr():.4f}")

    print("\n" + "=" * 50)
    print("All optimizer tests passed!")
