"""
Custom Activation Functions - Production-Ready Implementations

This module provides custom activation functions with autograd support.

Activations:
    - SwishFunction / swish: Self-gated activation (SiLU)
    - MishFunction / mish: Self-regularized non-monotonic activation
    - HardSwishFunction / hard_swish: Efficient Swish approximation
    - StarReLUFunction / star_relu: Star-ReLU from MetaFormer

All functions have:
    - Custom forward and backward implementations
    - Gradient verification
    - GPU support

Example:
    >>> from custom_activations import swish, mish, Swish, Mish
    >>> x = torch.randn(10, requires_grad=True)
    >>> y = swish(x)  # Functional API
    >>> layer = Swish()  # Module API
    >>> z = layer(x)

Author: DGX Spark AI Curriculum
"""

__all__ = [
    'SwishFunction',
    'swish',
    'Swish',
    'MishFunction',
    'mish',
    'Mish',
    'HardSwishFunction',
    'hard_swish',
    'HardSwish',
    'StarReLUFunction',
    'star_relu',
    'StarReLU',
    'verify_gradients',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, gradcheck
from typing import Optional


class SwishFunction(Function):
    """
    Custom autograd function for Swish (SiLU) activation.

    Swish(x) = x * sigmoid(x)

    Paper: "Searching for Activation Functions" (Ramachandran et al., 2017)
    https://arxiv.org/abs/1710.05941
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Compute Swish(x) = x * sigmoid(x)."""
        sigmoid_x = torch.sigmoid(x)
        result = x * sigmoid_x
        ctx.save_for_backward(x, sigmoid_x)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient: d/dx Swish(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        """
        x, sigmoid_x = ctx.saved_tensors
        grad_input = sigmoid_x * (1 + x * (1 - sigmoid_x))
        return grad_output * grad_input


class MishFunction(Function):
    """
    Custom autograd function for Mish activation.

    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    Paper: "Mish: A Self Regularized Non-Monotonic Activation Function"
    https://arxiv.org/abs/1908.08681
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Compute Mish(x) = x * tanh(softplus(x))."""
        softplus_x = F.softplus(x)
        tanh_softplus = torch.tanh(softplus_x)
        result = x * tanh_softplus
        ctx.save_for_backward(x, tanh_softplus)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient using chain rule:
        d/dx Mish(x) = tanh(sp) + x * sech²(sp) * sigmoid(x)
        """
        x, tanh_sp = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        sech2_sp = 1 - tanh_sp ** 2
        grad_input = tanh_sp + x * sech2_sp * sigmoid_x
        return grad_output * grad_input


class HardSwishFunction(Function):
    """
    Custom autograd function for Hard Swish activation.

    HardSwish(x) = x * ReLU6(x + 3) / 6

    This is a computationally efficient approximation of Swish,
    used in MobileNetV3.

    Paper: "Searching for MobileNetV3" (Howard et al., 2019)
    https://arxiv.org/abs/1905.02244
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Compute HardSwish(x) = x * ReLU6(x + 3) / 6."""
        ctx.save_for_backward(x)
        return x * F.relu6(x + 3) / 6

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Compute piecewise gradient:
        - x < -3: 0
        - -3 <= x <= 3: (2x + 3) / 6
        - x > 3: 1
        """
        x, = ctx.saved_tensors

        grad_input = torch.zeros_like(x)
        # Middle region: -3 <= x <= 3
        mask_mid = (x >= -3) & (x <= 3)
        grad_input[mask_mid] = (2 * x[mask_mid] + 3) / 6
        # Upper region: x > 3
        grad_input[x > 3] = 1

        return grad_output * grad_input


class StarReLUFunction(Function):
    """
    Custom autograd function for Star-ReLU activation.

    StarReLU(x) = s * ReLU(x)² + b

    where s and b are learnable parameters.
    Default: s = 0.8944, b = -0.4472 (approximates GELU)

    Paper: "MetaFormer Baselines for Vision" (Yu et al., 2022)
    https://arxiv.org/abs/2210.13452
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: float = 0.8944,
        bias: float = -0.4472
    ) -> torch.Tensor:
        """Compute StarReLU(x) = scale * ReLU(x)² + bias."""
        relu_x = F.relu(x)
        result = scale * relu_x ** 2 + bias
        ctx.save_for_backward(x)
        ctx.scale = scale
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Compute gradient: 2 * scale * ReLU(x) * (x > 0)
        """
        x, = ctx.saved_tensors
        scale = ctx.scale

        grad_input = 2 * scale * F.relu(x) * (x > 0).float()
        # Return None for scale and bias (not differentiating w.r.t. them here)
        return grad_output * grad_input, None, None


# Functional API
def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Swish activation: x * sigmoid(x).

    Also known as SiLU (Sigmoid Linear Unit).

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return SwishFunction.apply(x)


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Mish activation: x * tanh(softplus(x)).

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return MishFunction.apply(x)


def hard_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Hard Swish activation: x * ReLU6(x + 3) / 6.

    Efficient approximation of Swish.

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return HardSwishFunction.apply(x)


def star_relu(
    x: torch.Tensor,
    scale: float = 0.8944,
    bias: float = -0.4472
) -> torch.Tensor:
    """
    Apply Star-ReLU activation: scale * ReLU(x)² + bias.

    Args:
        x: Input tensor
        scale: Scaling factor (default: 0.8944)
        bias: Bias term (default: -0.4472)

    Returns:
        Activated tensor
    """
    return StarReLUFunction.apply(x, scale, bias)


# Module API
class Swish(nn.Module):
    """Swish activation as an nn.Module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


class Mish(nn.Module):
    """Mish activation as an nn.Module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x)


class HardSwish(nn.Module):
    """Hard Swish activation as an nn.Module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x)


class StarReLU(nn.Module):
    """
    Star-ReLU activation as an nn.Module.

    Args:
        scale: Scaling factor (default: 0.8944)
        bias: Bias term (default: -0.4472)
        learnable: Whether scale and bias are learnable (default: False)
    """

    def __init__(
        self,
        scale: float = 0.8944,
        bias: float = -0.4472,
        learnable: bool = False
    ):
        super().__init__()

        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale))
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer('scale', torch.tensor(scale))
            self.register_buffer('bias', torch.tensor(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.relu(x) ** 2 + self.bias


def verify_gradients(verbose: bool = True) -> bool:
    """
    Verify all custom gradients using numerical differentiation.

    Args:
        verbose: Print results

    Returns:
        True if all gradients pass verification
    """
    all_passed = True
    functions = [
        ('Swish', SwishFunction.apply),
        ('Mish', MishFunction.apply),
        ('HardSwish', HardSwishFunction.apply),
    ]

    for name, fn in functions:
        x = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
        try:
            passed = gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
            if verbose:
                print(f"{name}: {'PASSED' if passed else 'FAILED'}")
            all_passed = all_passed and passed
        except Exception as e:
            if verbose:
                print(f"{name}: FAILED ({e})")
            all_passed = False

    return all_passed


if __name__ == '__main__':
    print("Testing custom activation functions...")
    print("=" * 50)

    # Verify gradients
    print("\nGradient verification:")
    all_passed = verify_gradients()
    print(f"\nAll tests passed: {all_passed}")

    # Test on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTesting on {device}...")

    x = torch.randn(10, 10, device=device, requires_grad=True)

    for name, fn in [('swish', swish), ('mish', mish), ('hard_swish', hard_swish)]:
        y = fn(x)
        y.sum().backward()
        print(f"{name}: output shape={y.shape}, grad shape={x.grad.shape}")
        x.grad = None

    print("\nAll tests completed!")
