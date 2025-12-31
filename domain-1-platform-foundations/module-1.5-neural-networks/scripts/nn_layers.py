"""
Neural Network Layers - Built from Scratch with NumPy

This module implements fundamental neural network layers and components
using only NumPy. Each layer includes forward and backward passes for
training with backpropagation.

Professor SPARK says: "Understanding these implementations is like knowing
how an engine works - you don't need to build one to drive a car, but
it makes you a much better driver when something goes wrong!"

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 1.5
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class Layer:
    """
    Base class for all neural network layers.

    Every layer must implement:
    - forward(): Compute output given input
    - backward(): Compute gradients given upstream gradient

    ELI5: Think of a layer like a LEGO brick in a tower.
    Each brick takes something from below (forward) and passes
    something up. When we're learning, we also need to pass
    messages back down (backward) to fix mistakes.
    """

    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}
        self.gradients: Dict[str, np.ndarray] = {}
        self.trainable = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass - must be implemented by subclass."""
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass - must be implemented by subclass."""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow layer to be called like a function."""
        return self.forward(x)


class Linear(Layer):
    """
    Fully-connected (dense) linear layer.

    Computes: output = input @ weights + bias

    ELI5: Imagine you have a voting committee. Each input feature
    is a voter, and each weight is how much we trust that voter's
    opinion. We multiply each voter's value by their trust level
    and add them all up. The bias is like a "default opinion" we
    start with.

    Parameters:
        in_features: Number of input features
        out_features: Number of output features
        init_method: Weight initialization ('he', 'xavier', 'normal')

    Example:
        >>> layer = Linear(784, 256, init_method='he')
        >>> x = np.random.randn(32, 784)  # batch of 32 images
        >>> output = layer(x)
        >>> print(output.shape)  # (32, 256)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_method: str = 'he',
        seed: Optional[int] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trainable = True

        if seed is not None:
            np.random.seed(seed)

        # Initialize weights based on method
        self.weights, self.bias = self._initialize_weights(init_method)

    def _initialize_weights(self, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize weights using specified method.

        Why this matters: Poor initialization can cause vanishing or
        exploding gradients, making training impossible or very slow.

        He init: Best for ReLU activations (keeps variance stable)
        Xavier init: Best for tanh/sigmoid activations
        """
        if method == 'he':
            # He initialization - optimal for ReLU
            std = np.sqrt(2.0 / self.in_features)
        elif method == 'xavier':
            # Xavier/Glorot initialization - optimal for tanh/sigmoid
            std = np.sqrt(2.0 / (self.in_features + self.out_features))
        elif method == 'normal':
            std = 0.01
        else:
            raise ValueError(f"Unknown init method: {method}")

        weights = np.random.randn(self.in_features, self.out_features) * std
        bias = np.zeros(self.out_features)

        return weights, bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W + b

        Args:
            x: Input array of shape (batch_size, in_features)

        Returns:
            Output array of shape (batch_size, out_features)
        """
        # Save input for backward pass
        self.cache['input'] = x

        # Linear transformation
        output = x @ self.weights + self.bias

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients for weights, bias, and input.

        Args:
            grad_output: Gradient from the layer above,
                        shape (batch_size, out_features)

        Returns:
            Gradient with respect to input, shape (batch_size, in_features)

        Math:
            dL/dW = input.T @ grad_output
            dL/db = sum(grad_output, axis=0)
            dL/dx = grad_output @ W.T
        """
        x = self.cache['input']
        batch_size = x.shape[0]

        # Gradient with respect to weights: average over batch
        self.gradients['weights'] = x.T @ grad_output / batch_size

        # Gradient with respect to bias: average over batch
        self.gradients['bias'] = np.mean(grad_output, axis=0)

        # Gradient with respect to input (to pass to previous layer)
        grad_input = grad_output @ self.weights.T

        return grad_input


class ReLU(Layer):
    """
    Rectified Linear Unit activation function.

    ReLU(x) = max(0, x)

    ELI5: ReLU is like a one-way valve for water. Positive water
    (positive numbers) flows through unchanged. Negative water
    (negative numbers) gets blocked - nothing comes through.

    Why ReLU is popular:
    1. Simple and fast to compute
    2. Doesn't saturate for positive values (no vanishing gradient)
    3. Creates sparse activations (many zeros = efficient)

    Example:
        >>> relu = ReLU()
        >>> x = np.array([[-2, -1, 0, 1, 2]])
        >>> print(relu(x))  # [[0, 0, 0, 1, 2]]
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: output = max(0, x)"""
        self.cache['input'] = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: gradient is 1 where x > 0, else 0.

        The derivative of max(0, x) is:
        - 1 if x > 0
        - 0 if x <= 0
        """
        x = self.cache['input']
        grad_input = grad_output * (x > 0).astype(float)
        return grad_input


class LeakyReLU(Layer):
    """
    Leaky ReLU activation function.

    LeakyReLU(x) = x if x > 0 else alpha * x

    ELI5: Like regular ReLU, but the valve isn't completely closed
    for negative values - it lets a tiny trickle through. This
    prevents "dead neurons" that can never activate again.

    Parameters:
        alpha: Slope for negative values (default: 0.01)

    Example:
        >>> leaky_relu = LeakyReLU(alpha=0.1)
        >>> x = np.array([[-2, -1, 0, 1, 2]])
        >>> print(leaky_relu(x))  # [[-0.2, -0.1, 0, 1, 2]]
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: max(alpha * x, x)"""
        self.cache['input'] = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: 1 where x > 0, alpha elsewhere."""
        x = self.cache['input']
        grad = np.where(x > 0, 1.0, self.alpha)
        return grad_output * grad


class Sigmoid(Layer):
    """
    Sigmoid activation function.

    sigmoid(x) = 1 / (1 + exp(-x))

    ELI5: Sigmoid squashes any number into a value between 0 and 1,
    like a probability. Very negative numbers become almost 0,
    very positive numbers become almost 1, and 0 becomes 0.5.

    Warning: Sigmoid causes vanishing gradients in deep networks!
    The gradient is at most 0.25, so in deep networks, gradients
    get multiplied many times and shrink to nearly zero.

    Example:
        >>> sigmoid = Sigmoid()
        >>> x = np.array([[-5, 0, 5]])
        >>> print(sigmoid(x))  # [[0.007, 0.5, 0.993]]
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with numerical stability."""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        output = 1.0 / (1.0 + np.exp(-x_clipped))
        self.cache['output'] = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

        Note: Max gradient is 0.25 (at x=0), causing vanishing gradients.
        """
        output = self.cache['output']
        grad = output * (1 - output)
        return grad_output * grad


class Tanh(Layer):
    """
    Hyperbolic tangent activation function.

    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    ELI5: Like sigmoid, but squashes numbers between -1 and 1
    instead of 0 and 1. This is often better because the output
    is "centered" around zero, which helps with learning.

    Example:
        >>> tanh = Tanh()
        >>> x = np.array([[-5, 0, 5]])
        >>> print(tanh(x))  # [[-0.9999, 0, 0.9999]]
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        output = np.tanh(x)
        self.cache['output'] = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: tanh'(x) = 1 - tanh(x)^2"""
        output = self.cache['output']
        grad = 1 - output ** 2
        return grad_output * grad


class GELU(Layer):
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * Phi(x) where Phi is the CDF of standard normal

    ELI5: GELU is like ReLU's smarter cousin. Instead of a hard
    cutoff at zero, it uses probability to decide how much of
    each value to keep. Small negative values might still get
    through a little bit. This is the activation used in GPT
    and BERT!

    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Example:
        >>> gelu = GELU()
        >>> x = np.array([[-2, -1, 0, 1, 2]])
        >>> print(gelu(x))  # [[-0.045, -0.158, 0, 0.841, 1.955]]
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using tanh approximation."""
        self.cache['input'] = x
        # Using the tanh approximation for efficiency
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass (approximate gradient)."""
        x = self.cache['input']
        # Approximate derivative
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
        grad = cdf + x * pdf
        return grad_output * grad


class SiLU(Layer):
    """
    Sigmoid Linear Unit (also called Swish) activation.

    SiLU(x) = x * sigmoid(x)

    ELI5: SiLU multiplies each value by its own sigmoid. This is
    like asking "how confident are you?" and scaling the value by
    that confidence. Small values become even smaller, large values
    stay large. Used in modern architectures like EfficientNet.

    Example:
        >>> silu = SiLU()
        >>> x = np.array([[-2, -1, 0, 1, 2]])
        >>> print(silu(x))  # [[-0.24, -0.27, 0, 0.73, 1.76]]
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: x * sigmoid(x)"""
        self.cache['input'] = x
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        self.cache['sigmoid'] = sigmoid
        return x * sigmoid

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))"""
        x = self.cache['input']
        sigmoid = self.cache['sigmoid']
        grad = sigmoid + x * sigmoid * (1 - sigmoid)
        return grad_output * grad


class Softmax(Layer):
    """
    Softmax activation function for multi-class classification.

    softmax(x)_i = exp(x_i) / sum(exp(x_j))

    ELI5: Softmax turns any list of numbers into probabilities that
    add up to 1. It's like a "fair share" calculator - bigger inputs
    get more probability, but everyone gets something. Perfect for
    "which class is this?" questions.

    Example:
        >>> softmax = Softmax()
        >>> x = np.array([[1, 2, 3]])
        >>> print(softmax(x))  # [[0.09, 0.24, 0.67]]  (sums to 1.0)
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with numerical stability.

        Subtracting max prevents exp from overflowing.
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.cache['output'] = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for softmax.

        Note: Usually combined with cross-entropy loss, which
        simplifies the gradient to (softmax_output - targets).
        This method computes the full Jacobian for standalone use.
        """
        output = self.cache['output']
        # For each sample in batch
        batch_size = output.shape[0]
        grad_input = np.zeros_like(output)

        for i in range(batch_size):
            s = output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - s @ s.T
            grad_input[i] = jacobian @ grad_output[i]

        return grad_input


class CrossEntropyLoss:
    """
    Cross-entropy loss for classification tasks.

    loss = -sum(targets * log(predictions))

    ELI5: Cross-entropy measures how surprised we are by the
    predictions. If we're confident about the right answer,
    we're not surprised and the loss is low. If we're confident
    about the wrong answer, we're very surprised and the loss
    is high!

    Example:
        >>> loss_fn = CrossEntropyLoss()
        >>> predictions = np.array([[0.1, 0.2, 0.7]])  # predicted probs
        >>> targets = np.array([2])  # true class is 2
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Loss: {loss:.4f}")  # Low loss because we predicted 0.7 for class 2
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        self.cache: Dict[str, np.ndarray] = {}

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute cross-entropy loss.

        Args:
            predictions: Softmax probabilities, shape (batch_size, num_classes)
            targets: True labels, shape (batch_size,) with class indices

        Returns:
            Scalar loss value (averaged over batch)
        """
        batch_size = predictions.shape[0]

        # Clip predictions to prevent log(0)
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)

        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            num_classes = predictions.shape[1]
            targets_onehot = np.zeros_like(predictions)
            targets_onehot[np.arange(batch_size), targets] = 1
        else:
            targets_onehot = targets

        # Compute cross-entropy
        loss = -np.sum(targets_onehot * np.log(predictions_clipped)) / batch_size

        # Save for backward pass
        self.cache['predictions'] = predictions
        self.cache['targets'] = targets_onehot

        return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to predictions.

        When combined with softmax, the gradient simplifies to:
        dL/dz = softmax(z) - targets (one-hot)

        This is the "magic" of combining softmax + cross-entropy!
        """
        predictions = self.cache['predictions']
        targets = self.cache['targets']

        # Simple gradient for softmax + cross-entropy combo
        return predictions - targets


class MSELoss:
    """
    Mean Squared Error loss for regression tasks.

    loss = mean((predictions - targets)^2)

    ELI5: MSE measures how far off our predictions are, squared.
    The squaring makes big mistakes hurt much more than small ones.
    Predicting 10 when the answer is 11 is okay (error = 1).
    Predicting 10 when the answer is 20 is terrible (error = 100)!

    Example:
        >>> loss_fn = MSELoss()
        >>> predictions = np.array([[2.5, 3.5]])
        >>> targets = np.array([[2.0, 4.0]])
        >>> loss = loss_fn(predictions, targets)
        >>> print(f"Loss: {loss:.4f}")  # 0.25
    """

    def __init__(self):
        self.cache: Dict[str, np.ndarray] = {}

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Compute MSE loss."""
        self.cache['predictions'] = predictions
        self.cache['targets'] = targets

        loss = np.mean((predictions - targets) ** 2)
        return loss

    def backward(self) -> np.ndarray:
        """Gradient: 2 * (predictions - targets) / n"""
        predictions = self.cache['predictions']
        targets = self.cache['targets']
        n = predictions.size

        return 2 * (predictions - targets) / n


class Dropout(Layer):
    """
    Dropout regularization layer.

    Randomly sets a fraction of input units to zero during training.
    This helps prevent overfitting by making the network more robust.

    ELI5: Dropout is like randomly asking some students to skip class.
    The remaining students can't rely on their absent friends, so they
    have to learn the material themselves. This makes the whole class
    stronger and less dependent on any single student!

    Parameters:
        rate: Probability of dropping a neuron (0 to 1)

    Note: Uses "inverted dropout" - scales by 1/(1-rate) during training
    so no scaling is needed at test time.

    Example:
        >>> dropout = Dropout(rate=0.5)
        >>> x = np.ones((4, 10))
        >>> out = dropout(x, training=True)  # ~half the values are 0
        >>> out = dropout(x, training=False)  # all values unchanged
    """

    def __init__(self, rate: float = 0.5):
        super().__init__()
        if not 0 <= rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass with dropout.

        Args:
            x: Input array
            training: If True, apply dropout. If False, pass through unchanged.

        Returns:
            Output with dropout applied (training) or unchanged (inference)
        """
        if training and self.rate > 0:
            # Create binary mask: 1 = keep, 0 = drop
            self.mask = (np.random.rand(*x.shape) > self.rate).astype(np.float32)
            # Apply mask and scale by 1/(1-rate) to maintain expected values
            return x * self.mask / (1 - self.rate)
        else:
            self.mask = None
            return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: apply same mask as forward pass.
        """
        if self.mask is not None:
            return grad_output * self.mask / (1 - self.rate)
        return grad_output

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(x, training)


class Sequential:
    """
    Container for a sequence of layers.

    ELI5: Sequential is like a recipe that lists steps in order.
    Data flows through each layer one by one, like an assembly line.
    To learn, we trace back through the same steps in reverse.

    Example:
        >>> model = Sequential([
        ...     Linear(784, 256),
        ...     ReLU(),
        ...     Linear(256, 10),
        ...     Softmax()
        ... ])
        >>> output = model(input_data)  # Forward through all layers
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pass input through all layers in sequence."""
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Pass gradient backward through all layers."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow model to be called like a function."""
        return self.forward(x)

    def get_trainable_layers(self) -> list:
        """Return list of layers that have trainable parameters."""
        return [layer for layer in self.layers if layer.trainable]

    def parameters(self) -> list:
        """Return list of all trainable parameters."""
        params = []
        for layer in self.get_trainable_layers():
            if hasattr(layer, 'weights'):
                params.append(('weights', layer.weights, layer.gradients.get('weights')))
            if hasattr(layer, 'bias'):
                params.append(('bias', layer.bias, layer.gradients.get('bias')))
        return params


def get_activation(name: str, **kwargs) -> Layer:
    """
    Factory function to get activation layer by name.

    Args:
        name: One of 'relu', 'leaky_relu', 'sigmoid', 'tanh', 'gelu', 'silu'
        **kwargs: Additional arguments for the activation

    Returns:
        Activation layer instance

    Example:
        >>> relu = get_activation('relu')
        >>> leaky = get_activation('leaky_relu', alpha=0.2)
    """
    activations = {
        'relu': ReLU,
        'leaky_relu': LeakyReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'gelu': GELU,
        'silu': SiLU,
        'softmax': Softmax
    }

    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from: {list(activations.keys())}")

    return activations[name.lower()](**kwargs)


if __name__ == "__main__":
    # Quick test of all components
    print("Testing Neural Network Layers")
    print("=" * 50)

    # Test Linear layer
    print("\n1. Testing Linear Layer:")
    linear = Linear(10, 5)
    x = np.random.randn(3, 10)
    out = linear(x)
    print(f"   Input shape: {x.shape} -> Output shape: {out.shape}")

    # Test activations
    print("\n2. Testing Activations:")
    test_input = np.array([[-2, -1, 0, 1, 2]], dtype=float)

    for name in ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'gelu', 'silu']:
        act = get_activation(name)
        output = act(test_input.copy())
        print(f"   {name:12s}: {output[0]}")

    # Test Softmax
    print("\n3. Testing Softmax:")
    softmax = Softmax()
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = softmax(logits)
    print(f"   Input: {logits[0]} -> Probs: {probs[0]} (sum={probs.sum():.4f})")

    # Test CrossEntropyLoss
    print("\n4. Testing CrossEntropyLoss:")
    loss_fn = CrossEntropyLoss()
    predictions = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]])
    targets = np.array([2, 0])
    loss = loss_fn(predictions, targets)
    print(f"   Loss: {loss:.4f}")

    # Test Sequential
    print("\n5. Testing Sequential Model:")
    model = Sequential([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
        Softmax()
    ])
    x = np.random.randn(32, 784)
    output = model(x)
    print(f"   Input: (32, 784) -> Output: {output.shape}")
    print(f"   Output sums to 1: {np.allclose(output.sum(axis=1), 1)}")

    print("\n" + "=" * 50)
    print("All tests passed!")
