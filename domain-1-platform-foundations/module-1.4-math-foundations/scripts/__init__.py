"""
Module 1.4: Mathematics for Deep Learning - Scripts Package

This package provides reusable utilities for mathematical operations
commonly used in deep learning, including:

- Activation functions and their derivatives (sigmoid, relu, tanh, softmax)
- Loss functions (MSE, cross-entropy, binary cross-entropy)
- Optimizers (SGD, SGD with Momentum, Adam, AdamW)
- SVD utilities for understanding LoRA
- Probability distribution functions
- Gradient checking utilities

Example Usage:
    >>> from scripts.math_utils import sigmoid, relu, Adam
    >>> from scripts.visualization_utils import plot_loss_landscape
    >>>
    >>> # Use activation functions
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = sigmoid(x)
    >>>
    >>> # Use optimizer
    >>> opt = Adam(lr=0.001)
    >>> params = opt.step(params, gradients)
    >>>
    >>> # Plot loss landscape
    >>> rosenbrock = lambda x, y: (1-x)**2 + 100*(y-x**2)**2
    >>> plot_loss_landscape(rosenbrock)

Available modules:
- math_utils: Mathematical helper functions (activations, losses, optimizers)
- visualization_utils: Visualization tools for loss landscapes, training curves
"""

from .math_utils import *
from .visualization_utils import *
