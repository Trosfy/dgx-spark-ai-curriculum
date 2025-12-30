"""
Neural Network Layers for MicroGrad+.

This module implements common neural network layers that work with the
Tensor autograd system. Each layer is a callable that transforms input
tensors and maintains trainable parameters.

ELI5 Explanation:
    Think of neural network layers like stations in a factory assembly line.
    Each station takes a product, does something to it (painting, bending,
    adding parts), and passes it to the next station. The "weights" of a
    layer are like the settings on each machine - we adjust them during
    training to make the final product (prediction) come out right!

Example:
    >>> from micrograd_plus import Tensor, Linear, ReLU
    >>>
    >>> # Create a simple network
    >>> layer1 = Linear(10, 5)
    >>> activation = ReLU()
    >>> layer2 = Linear(5, 2)
    >>>
    >>> # Forward pass
    >>> x = Tensor(np.random.randn(3, 10))  # batch of 3, 10 features
    >>> h = activation(layer1(x))
    >>> output = layer2(h)
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
from .tensor import Tensor


class Module:
    """
    Base class for all neural network layers.

    This provides common functionality for managing parameters, training mode,
    and layer organization.

    Subclasses should implement:
        - forward(self, x: Tensor) -> Tensor: The forward computation
        - parameters(self) -> List[Tensor]: Return trainable parameters

    Example:
        >>> class MyLayer(Module):
        ...     def __init__(self, in_features, out_features):
        ...         super().__init__()
        ...         self.weight = Tensor(np.random.randn(in_features, out_features) * 0.1,
        ...                              requires_grad=True)
        ...     def forward(self, x):
        ...         return x @ self.weight
        ...     def parameters(self):
        ...         return [self.weight]
    """

    def __init__(self):
        self._training = True

    @property
    def training(self) -> bool:
        """Return whether the module is in training mode."""
        return self._training

    def train(self, mode: bool = True) -> 'Module':
        """
        Set the module to training mode.

        This affects layers like Dropout and BatchNorm that behave
        differently during training vs evaluation.

        Args:
            mode: True for training, False for evaluation.

        Returns:
            self (for chaining).
        """
        self._training = mode
        return self

    def eval(self) -> 'Module':
        """Set the module to evaluation mode."""
        return self.train(False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, x: Tensor) -> Tensor:
        """Make the layer callable."""
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        """
        Return a list of all trainable parameters.

        Returns:
            List of Tensor objects with requires_grad=True.
        """
        return []

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Linear(Module):
    """
    Fully connected (dense) layer: y = xW + b.

    This is the fundamental building block of neural networks. It computes
    a weighted sum of inputs plus a bias term.

    ELI5:
        Imagine each input is a vote, and each weight is how important that
        vote is. We add up all the votes multiplied by their importance,
        then add a starting bonus (bias). The result is our "score" for
        each output feature.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term (default: True).

    Attributes:
        weight: The weight matrix of shape (in_features, out_features).
        bias: The bias vector of shape (out_features,), or None if disabled.

    Example:
        >>> layer = Linear(784, 256)  # MNIST input to hidden layer
        >>> x = Tensor(np.random.randn(32, 784))  # batch of 32 images
        >>> y = layer(x)  # shape: (32, 256)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier/Glorot initialization
        # This helps gradients flow better through deep networks
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.randn(in_features, out_features).astype(np.float32) * scale,
            requires_grad=True
        )

        if bias:
            self.bias = Tensor(
                np.zeros(out_features, dtype=np.float32),
                requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass: y = xW + b.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self) -> List[Tensor]:
        """Return weight and bias parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class ReLU(Module):
    """
    Rectified Linear Unit activation: ReLU(x) = max(0, x).

    This is the most popular activation function in modern neural networks.
    It simply passes positive values through and blocks negative values.

    ELI5:
        Imagine a gate that only lets positive numbers through. Negative
        numbers become zero. This is like a bouncer at a club who only
        lets in guests with "positive vibes"!

    Why it works:
        - Simple and fast to compute
        - Avoids vanishing gradient problem (gradient is 0 or 1)
        - Creates sparse representations (many zeros)

    Example:
        >>> relu = ReLU()
        >>> x = Tensor([-2, -1, 0, 1, 2])
        >>> relu(x)  # Tensor([0., 0., 0., 1., 2.])
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU activation."""
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """
    Sigmoid activation: σ(x) = 1 / (1 + e^(-x)).

    Squashes any input to a value between 0 and 1. Useful for probabilities.

    ELI5:
        Imagine a smooth ramp that goes from 0 to 1. Very negative numbers
        slide down to nearly 0, very positive numbers climb to nearly 1,
        and zero gives you exactly 0.5 in the middle.

    Common uses:
        - Binary classification output layer
        - Gate mechanisms (like in LSTMs)
        - Anywhere you need probabilities

    Example:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor([-2, 0, 2])
        >>> sigmoid(x)  # Tensor([0.12, 0.5, 0.88])
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation."""
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """
    Hyperbolic tangent activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)).

    Similar to sigmoid but outputs values between -1 and 1.

    ELI5:
        Like sigmoid, but the ramp goes from -1 to +1 instead of 0 to 1.
        This means the average output is around 0, which can help training.

    Example:
        >>> tanh = Tanh()
        >>> x = Tensor([-2, 0, 2])
        >>> tanh(x)  # Tensor([-0.96, 0., 0.96])
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply tanh activation."""
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Module):
    """
    Softmax activation: softmax(x)_i = e^(x_i) / Σ e^(x_j).

    Converts a vector of numbers into a probability distribution that sums to 1.

    ELI5:
        Imagine you have test scores and want to convert them to percentages
        of "how much better" each student did. Softmax does this, but in a
        way where higher scores get exponentially more "percentage points".
        The result always adds up to 100% (or 1.0).

    Args:
        axis: The axis along which to compute softmax (default: -1, last axis).

    Example:
        >>> softmax = Softmax()
        >>> logits = Tensor([[1, 2, 3]])  # Raw scores
        >>> probs = softmax(logits)
        >>> print(probs)  # [[0.09, 0.24, 0.67]]
        >>> print(probs.sum())  # 1.0
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """Apply softmax activation."""
        return x.softmax(axis=self.axis)

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"


class LogSoftmax(Module):
    """
    Log-softmax: log(softmax(x)).

    Computes log of softmax in a numerically stable way. Often used with
    negative log likelihood loss.

    Args:
        axis: The axis along which to compute log-softmax (default: -1).

    Example:
        >>> log_softmax = LogSoftmax()
        >>> logits = Tensor([[1, 2, 3]])
        >>> log_probs = log_softmax(logits)
        >>> print(log_probs)  # [[-2.41, -1.41, -0.41]]
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        """Apply log-softmax."""
        return x.log_softmax(axis=self.axis)

    def __repr__(self) -> str:
        return f"LogSoftmax(axis={self.axis})"


class Dropout(Module):
    """
    Dropout regularization layer.

    During training, randomly sets elements to zero with probability p.
    The remaining elements are scaled by 1/(1-p) to maintain expected values.
    During evaluation, this layer does nothing.

    ELI5:
        Imagine a team where sometimes random members call in sick. The rest
        of the team learns to pick up the slack. This makes the team more
        robust - no single person is too important. That's dropout for neurons!

    Args:
        p: Probability of dropping each element (default: 0.5).

    Example:
        >>> dropout = Dropout(p=0.2)
        >>> x = Tensor([[1, 2, 3, 4, 5]])
        >>>
        >>> dropout.train()  # Training mode
        >>> y = dropout(x)  # Some values become 0, others scaled up
        >>>
        >>> dropout.eval()  # Eval mode
        >>> y = dropout(x)  # Returns x unchanged
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Apply dropout during training."""
        if not self.training or self.p == 0:
            return x

        # Create mask: 1 with probability (1-p), 0 with probability p
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)

        # Apply dropout with inverted scaling
        out = Tensor(
            x.data * mask * scale,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op='dropout'
        )

        def _backward():
            if x.requires_grad:
                x.grad += mask * scale * out.grad
        out._backward = _backward

        return out

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class BatchNorm(Module):
    """
    Batch Normalization layer.

    Normalizes inputs to have zero mean and unit variance across the batch,
    then applies a learnable scale (gamma) and shift (beta).

    During training, uses batch statistics. During evaluation, uses running
    averages computed during training.

    ELI5:
        Imagine students taking a test. Some tests are harder than others.
        BatchNorm is like grading on a curve - it adjusts everyone's scores
        so the average is always 0 and spread is always the same. Then it
        lets the network learn what the "real" average and spread should be.

    Args:
        num_features: Number of features (channels) in the input.
        eps: Small constant for numerical stability (default: 1e-5).
        momentum: Momentum for running statistics (default: 0.1).

    Example:
        >>> bn = BatchNorm(64)  # For 64 features
        >>> x = Tensor(np.random.randn(32, 64))  # batch of 32
        >>> y = bn(x)  # Normalized output
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)

        # Running statistics (not parameters, but state)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        """Apply batch normalization."""
        if self.training:
            # Compute batch statistics
            batch_mean = x.data.mean(axis=0)
            batch_var = x.data.var(axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_centered = x.data - mean
        std = np.sqrt(var + self.eps)
        x_norm = x_centered / std

        # Scale and shift
        out_data = self.gamma.data * x_norm + self.beta.data

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
            _children=(x, self.gamma, self.beta),
            _op='batchnorm'
        )

        # Store for backward
        def _backward():
            N = x.data.shape[0]

            if self.gamma.requires_grad:
                self.gamma.grad += (x_norm * out.grad).sum(axis=0)
            if self.beta.requires_grad:
                self.beta.grad += out.grad.sum(axis=0)
            if x.requires_grad:
                dx_norm = out.grad * self.gamma.data
                dvar = (dx_norm * x_centered * -0.5 * (var + self.eps) ** (-1.5)).sum(axis=0)
                dmean = (dx_norm * -1 / std).sum(axis=0) + dvar * (-2 * x_centered).mean(axis=0)
                x.grad += dx_norm / std + dvar * 2 * x_centered / N + dmean / N

        out._backward = _backward

        return out

    def parameters(self) -> List[Tensor]:
        """Return gamma and beta parameters."""
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return f"BatchNorm(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum})"


class LayerNorm(Module):
    """
    Layer Normalization.

    Unlike BatchNorm, this normalizes across features (not batch dimension).
    More suitable for sequence models and transformers.

    Args:
        normalized_shape: Size of the last dimension to normalize.
        eps: Small constant for numerical stability.

    Example:
        >>> ln = LayerNorm(256)
        >>> x = Tensor(np.random.randn(32, 10, 256))  # (batch, seq, features)
        >>> y = ln(x)  # Normalized across last dimension
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization."""
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)

        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad or self.gamma.requires_grad,
            _children=(x, self.gamma, self.beta),
            _op='layernorm'
        )

        def _backward():
            N = self.normalized_shape
            x_centered = x.data - mean
            std = np.sqrt(var + self.eps)

            if self.gamma.requires_grad:
                self.gamma.grad += (x_norm * out.grad).sum(axis=tuple(range(x.ndim - 1)))
            if self.beta.requires_grad:
                self.beta.grad += out.grad.sum(axis=tuple(range(x.ndim - 1)))
            if x.requires_grad:
                dx_norm = out.grad * self.gamma.data
                dvar = (dx_norm * x_centered * -0.5 * (var + self.eps) ** (-1.5)).sum(axis=-1, keepdims=True)
                dmean = (dx_norm * -1 / std).sum(axis=-1, keepdims=True) + dvar * (-2 * x_centered).mean(axis=-1, keepdims=True)
                x.grad += dx_norm / std + dvar * 2 * x_centered / N + dmean / N

        out._backward = _backward

        return out

    def parameters(self) -> List[Tensor]:
        """Return gamma and beta parameters."""
        return [self.gamma, self.beta]

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps})"


class Flatten(Module):
    """
    Flatten layer - reshapes input to 2D (batch_size, features).

    Args:
        start_dim: First dimension to flatten (default: 1, keep batch dim).

    Example:
        >>> flatten = Flatten()
        >>> x = Tensor(np.random.randn(32, 3, 28, 28))  # (batch, channels, height, width)
        >>> y = flatten(x)  # (32, 2352) where 2352 = 3*28*28
    """

    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: Tensor) -> Tensor:
        """Flatten the tensor."""
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim})"


class Embedding(Module):
    """
    Embedding layer - maps discrete tokens to continuous vectors.

    ELI5:
        Imagine a dictionary where each word has a list of numbers that describe
        its meaning. "Cat" might be [0.8, 0.2, 0.9, ...] meaning "furry, small,
        cute, ...". This layer is that dictionary - give it a word ID, get back
        a vector that captures its meaning.

    Args:
        num_embeddings: Size of vocabulary (number of unique tokens).
        embedding_dim: Dimension of embedding vectors.

    Example:
        >>> emb = Embedding(10000, 256)  # 10k vocab, 256-dim embeddings
        >>> token_ids = np.array([[1, 5, 3, 2], [4, 7, 8, 0]])  # (batch, seq_len)
        >>> vectors = emb(Tensor(token_ids))  # (2, 4, 256)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize with standard normal
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Look up embeddings for input indices."""
        indices = x.data.astype(np.int32)
        out_data = self.weight.data[indices]

        out = Tensor(
            out_data,
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,),
            _op='embedding'
        )

        def _backward():
            if self.weight.requires_grad:
                np.add.at(self.weight.grad, indices.flatten(), out.grad.reshape(-1, self.embedding_dim))
        out._backward = _backward

        return out

    def parameters(self) -> List[Tensor]:
        """Return embedding weights."""
        return [self.weight]

    def __repr__(self) -> str:
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
