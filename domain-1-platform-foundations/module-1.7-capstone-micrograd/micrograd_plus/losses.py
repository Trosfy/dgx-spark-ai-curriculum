"""
Loss Functions for MicroGrad+.

This module implements common loss functions used for training neural networks.
Each loss function computes a scalar that measures how far the predictions
are from the targets.

ELI5 Explanation:
    Imagine you're learning to throw darts at a bullseye. The loss function
    is like your coach telling you how far off your throw was. A good loss
    function tells you not just "you missed" but "you missed by this much
    in this direction" so you can adjust your aim!

Example:
    >>> from micrograd_plus import Tensor, MSELoss
    >>>
    >>> # Predictions and targets
    >>> pred = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> target = Tensor([[1.5, 2.5], [2.5, 3.5]])
    >>>
    >>> # Compute loss
    >>> loss_fn = MSELoss()
    >>> loss = loss_fn(pred, target)
    >>> print(loss)  # Mean squared difference
    >>>
    >>> # Backpropagate to get gradients
    >>> loss.backward()
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from .tensor import Tensor


class Loss:
    """
    Base class for all loss functions.

    All loss functions inherit from this class and implement the __call__ method.

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').
    """

    def __init__(self, reduction: str = 'mean'):
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute the loss. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Make the loss callable."""
        return self.forward(predictions, targets)


class MSELoss(Loss):
    """
    Mean Squared Error Loss.

    Computes: L = mean((pred - target)^2)

    This is the standard loss for regression problems where you want to
    predict continuous values.

    ELI5:
        Imagine you're guessing people's heights. For each guess, you
        calculate (your guess - actual height)^2. The squaring punishes
        big mistakes more than small ones. Then you take the average of
        all these squared errors. Lower is better!

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').
            - 'mean': Average over all elements (default)
            - 'sum': Sum of all elements
            - 'none': Return loss for each element

    Example:
        >>> loss_fn = MSELoss()
        >>> pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> target = Tensor([1.5, 2.0, 2.5])
        >>> loss = loss_fn(pred, target)
        >>> print(loss)  # Tensor(0.1667) = mean([0.25, 0, 0.25])
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE loss.

        Args:
            pred: Predicted values (any shape).
            target: Target values (same shape as pred).

        Returns:
            Scalar loss (or per-element if reduction='none').
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)

        diff = pred - target
        squared = diff ** 2

        if self.reduction == 'mean':
            return squared.mean()
        elif self.reduction == 'sum':
            return squared.sum()
        else:
            return squared

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss for multi-class classification.

    Expects raw logits (not softmax) as predictions and class indices as targets.
    Computes: L = -log(softmax(logits)[target_class])

    This is the standard loss for classification problems where each example
    belongs to exactly one class.

    ELI5:
        Imagine you're on a game show guessing which door has the prize.
        You say "I'm 80% sure it's door 1, 15% door 2, 5% door 3".
        If the prize was behind door 1, you get a small penalty (-log(0.8) ≈ 0.22).
        If it was behind door 3, you get a huge penalty (-log(0.05) ≈ 3.0).
        The loss punishes confident wrong answers severely!

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> # Batch of 2 samples, 3 classes
        >>> logits = Tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True)
        >>> targets = Tensor([0, 1])  # First sample is class 0, second is class 1
        >>> loss = loss_fn(logits, targets)
        >>> loss.backward()
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Raw predictions of shape (batch_size, num_classes).
            targets: Class indices of shape (batch_size,) with values in [0, num_classes-1].

        Returns:
            Scalar loss (or per-sample if reduction='none').
        """
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # Compute log-softmax for numerical stability
        log_probs = logits.log_softmax(axis=1)

        # Get log probabilities for target classes
        target_indices = targets.data.astype(np.int32)

        # Create one-hot encoding
        one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot[np.arange(batch_size), target_indices] = 1.0

        # Negative log likelihood: -sum(one_hot * log_probs)
        nll = -(log_probs * Tensor(one_hot)).sum(axis=1)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class BCELoss(Loss):
    """
    Binary Cross-Entropy Loss.

    Expects probabilities (after sigmoid) as predictions.
    For raw logits, use BCEWithLogitsLoss instead.

    Computes: L = -(target * log(pred) + (1-target) * log(1-pred))

    Use this for binary classification or multi-label classification.

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
        >>> loss_fn = BCELoss()
        >>> pred = Tensor([0.8, 0.4, 0.1], requires_grad=True)  # Probabilities
        >>> target = Tensor([1, 0, 0])  # Binary labels
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute binary cross-entropy loss.

        Args:
            pred: Predicted probabilities in [0, 1].
            target: Binary targets (0 or 1).

        Returns:
            Scalar loss (or per-element if reduction='none').
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Clamp for numerical stability
        eps = 1e-7
        pred_clamped = Tensor(np.clip(pred.data, eps, 1 - eps),
                               requires_grad=pred.requires_grad,
                               _children=(pred,), _op='clamp')

        def _backward():
            if pred.requires_grad:
                pred.grad += pred_clamped.grad * ((pred.data > eps) & (pred.data < 1-eps)).astype(np.float32)
        pred_clamped._backward = _backward

        # BCE formula
        loss = -(target * pred_clamped.log() + (1 - target) * (1 - pred_clamped).log())

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"BCELoss(reduction='{self.reduction}')"


class BCEWithLogitsLoss(Loss):
    """
    Binary Cross-Entropy Loss with built-in sigmoid.

    More numerically stable than applying sigmoid then BCELoss.
    Expects raw logits as predictions.

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
        >>> loss_fn = BCEWithLogitsLoss()
        >>> logits = Tensor([2.0, -1.0, -3.0], requires_grad=True)  # Raw scores
        >>> target = Tensor([1, 0, 0])  # Binary labels
        >>> loss = loss_fn(logits, target)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute BCE with logits.

        Args:
            logits: Raw predictions (any real number).
            target: Binary targets (0 or 1).

        Returns:
            Scalar loss (or per-element if reduction='none').
        """
        if not isinstance(target, Tensor):
            target = Tensor(target)

        # Numerically stable formula:
        # max(logits, 0) - logits * target + log(1 + exp(-|logits|))

        # Use relu to get max(logits, 0) with gradient support
        max_val = logits.relu()

        # For stability, we compute log(1 + exp(-|x|))
        # Use abs() and exp() which have gradient support
        abs_logits = logits.abs()
        neg_abs = abs_logits * (-1)
        exp_neg_abs = neg_abs.exp()
        one_plus_exp = exp_neg_abs + 1.0
        stable_exp = one_plus_exp.log()

        loss = max_val - logits * target + stable_exp

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"BCEWithLogitsLoss(reduction='{self.reduction}')"


class L1Loss(Loss):
    """
    Mean Absolute Error (L1) Loss.

    Computes: L = mean(|pred - target|)

    Less sensitive to outliers than MSE because it doesn't square the error.

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
        >>> loss_fn = L1Loss()
        >>> pred = Tensor([1.0, 2.0, 5.0], requires_grad=True)
        >>> target = Tensor([1.5, 2.0, 2.5])
        >>> loss = loss_fn(pred, target)
        >>> print(loss)  # Tensor(1.0) = mean([0.5, 0, 2.5])
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute L1 loss."""
        if not isinstance(target, Tensor):
            target = Tensor(target)

        loss = (pred - target).abs()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"L1Loss(reduction='{self.reduction}')"


class HuberLoss(Loss):
    """
    Huber Loss (Smooth L1 Loss).

    Combination of L1 and L2 loss - L2 for small errors, L1 for large errors.
    This makes it robust to outliers while still being differentiable everywhere.

    Computes:
        - 0.5 * (pred - target)^2              if |pred - target| < delta
        - delta * |pred - target| - 0.5*delta^2  otherwise

    Args:
        delta: Threshold where loss transitions from L2 to L1 (default: 1.0).
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
        >>> loss_fn = HuberLoss(delta=1.0)
        >>> pred = Tensor([1.0, 2.0, 10.0], requires_grad=True)
        >>> target = Tensor([1.5, 2.0, 2.5])
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute Huber loss."""
        if not isinstance(target, Tensor):
            target = Tensor(target)

        diff = pred - target
        abs_diff = diff.abs()

        # L2 part: 0.5 * diff^2
        l2_loss = 0.5 * diff ** 2
        # L1 part: delta * |diff| - 0.5 * delta^2
        l1_loss = self.delta * abs_diff - 0.5 * self.delta ** 2

        # Use L2 where |diff| < delta, L1 otherwise
        mask = (abs_diff.data < self.delta).astype(np.float32)
        loss = Tensor(l2_loss.data * mask + l1_loss.data * (1 - mask),
                      requires_grad=pred.requires_grad,
                      _children=(l2_loss, l1_loss),
                      _op='huber')

        def _backward():
            # Gradient for L2: diff, for L1: delta * sign(diff)
            l2_grad = diff.data * mask * loss.grad
            l1_grad = self.delta * np.sign(diff.data) * (1 - mask) * loss.grad
            if l2_loss.requires_grad:
                l2_loss.grad += l2_grad
            if l1_loss.requires_grad:
                l1_loss.grad += l1_grad
        loss._backward = _backward

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def __repr__(self) -> str:
        return f"HuberLoss(delta={self.delta}, reduction='{self.reduction}')"


class NLLLoss(Loss):
    """
    Negative Log Likelihood Loss.

    Expects log-probabilities (output of LogSoftmax) as input.
    For raw logits, use CrossEntropyLoss instead.

    Args:
        reduction: How to reduce the loss ('mean', 'sum', or 'none').

    Example:
        >>> from micrograd_plus import LogSoftmax
        >>> log_softmax = LogSoftmax()
        >>> loss_fn = NLLLoss()
        >>>
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> log_probs = log_softmax(logits)
        >>> target = Tensor([0])
        >>> loss = loss_fn(log_probs, target)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction)

    def forward(self, log_probs: Tensor, targets: Tensor) -> Tensor:
        """Compute negative log likelihood loss."""
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        batch_size = log_probs.shape[0]
        num_classes = log_probs.shape[1]

        target_indices = targets.data.astype(np.int32)

        # Create one-hot encoding
        one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot[np.arange(batch_size), target_indices] = 1.0

        # NLL: -sum(one_hot * log_probs)
        nll = -(log_probs * Tensor(one_hot)).sum(axis=1)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

    def __repr__(self) -> str:
        return f"NLLLoss(reduction='{self.reduction}')"
