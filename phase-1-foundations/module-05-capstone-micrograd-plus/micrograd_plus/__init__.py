"""
MicroGrad+ - A Tiny Autograd Engine with Neural Network Support.

This is an educational implementation of automatic differentiation and neural
networks, inspired by Andrej Karpathy's micrograd but extended with additional
features for learning purposes.

Example:
    >>> from micrograd_plus import Tensor, nn
    >>>
    >>> # Create tensors with gradients
    >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> w = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    >>>
    >>> # Forward pass
    >>> y = x @ w
    >>> loss = y.sum()
    >>>
    >>> # Backward pass
    >>> loss.backward()
    >>> print(x.grad)

Author: DGX Spark AI Curriculum
Version: 1.0.0
"""

from .tensor import Tensor
from .layers import (
    Module, Linear, ReLU, Sigmoid, Tanh, Softmax, LogSoftmax,
    Dropout, BatchNorm, LayerNorm, Flatten, Embedding
)
from .losses import (
    Loss, MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,
    L1Loss, HuberLoss, NLLLoss
)
from .optimizers import (
    Optimizer, SGD, Adam, AdamW, RMSprop,
    LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR
)
from .nn import (
    Sequential, train_epoch, evaluate, accuracy,
    count_parameters, save_model, load_model
)
from .utils import (
    set_seed,
    get_memory_usage,
    numerical_gradient,
    gradient_check,
    check_all_gradients,
    DataLoader,
    one_hot,
    normalize,
    train_test_split,
    ProgressBar,
    plot_training_history,
    visualize_computation_graph,
)

__version__ = "1.0.0"
__author__ = "DGX Spark AI Curriculum"

__all__ = [
    # Core
    "Tensor",
    # Layers
    "Module",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LogSoftmax",
    "Dropout",
    "BatchNorm",
    "LayerNorm",
    "Flatten",
    "Embedding",
    # Losses
    "Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "L1Loss",
    "HuberLoss",
    "NLLLoss",
    # Optimizers
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    "LRScheduler",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    # NN utilities
    "Sequential",
    "train_epoch",
    "evaluate",
    "accuracy",
    "count_parameters",
    "save_model",
    "load_model",
    # Utils
    "set_seed",
    "get_memory_usage",
    "numerical_gradient",
    "gradient_check",
    "check_all_gradients",
    "DataLoader",
    "one_hot",
    "normalize",
    "train_test_split",
    # Optional visualization utilities (require matplotlib/graphviz)
    "ProgressBar",
    "plot_training_history",
    "visualize_computation_graph",
]
