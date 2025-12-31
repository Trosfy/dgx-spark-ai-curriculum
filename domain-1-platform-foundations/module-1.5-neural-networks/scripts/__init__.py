"""
Module 1.5: Neural Network Fundamentals - Scripts Package

This package contains utility scripts for the Neural Network Fundamentals module.

Available modules:
- nn_layers: Neural network layer implementations (Linear, ReLU, Sigmoid, etc.)
- training_utils: Training helper functions (data loading, batching, metrics)
- optimizers: Optimizer implementations (SGD, Adam, AdamW, schedulers)
- normalization: Normalization layer implementations (BatchNorm, LayerNorm, RMSNorm)

Example usage:
    from nn_layers import Linear, ReLU, Softmax, Sequential
    from optimizers import Adam, AdamW
    from normalization import BatchNorm, LayerNorm, RMSNorm
    from training_utils import load_mnist, create_batches, accuracy
"""

from .nn_layers import *
from .training_utils import *
from .optimizers import *
from .normalization import *
