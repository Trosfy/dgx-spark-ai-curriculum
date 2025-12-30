"""
Quantization Utilities for DGX Spark

This module provides core quantization functions for neural network weights.
Optimized for DGX Spark's 128GB unified memory and Blackwell FP4 support.

Example:
    >>> import torch
    >>> from quantization_utils import symmetric_quantize, dequantize
    >>>
    >>> weights = torch.randn(4, 4)
    >>> quantized, scale = symmetric_quantize(weights, bits=4)
    >>> reconstructed = dequantize(quantized, scale)
    >>> error = (weights - reconstructed).abs().mean()
    >>> print(f"Mean quantization error: {error:.6f}")
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def symmetric_quantize(
    tensor: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    axis: int = -1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization of a tensor.

    Maps values to [-2^(bits-1), 2^(bits-1)-1] range symmetrically around zero.

    Args:
        tensor: Input float tensor to quantize
        bits: Number of bits (4 or 8 typically)
        per_channel: If True, compute scale per channel
        axis: Axis for per-channel quantization

    Returns:
        quantized: Integer tensor
        scale: Scale factor(s) for dequantization

    Example:
        >>> weights = torch.randn(64, 64)
        >>> q, scale = symmetric_quantize(weights, bits=8)
        >>> print(f"Quantized range: [{q.min()}, {q.max()}]")
    """
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)

    if per_channel:
        # Compute scale per channel
        max_val = tensor.abs().amax(dim=axis, keepdim=True)
    else:
        # Global scale
        max_val = tensor.abs().max()

    # Avoid division by zero
    max_val = torch.clamp(max_val, min=1e-10)

    # Compute scale
    scale = max_val / qmax

    # Quantize
    quantized = torch.round(tensor / scale).clamp(qmin, qmax)

    return quantized.to(torch.int8), scale


def asymmetric_quantize(
    tensor: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False,
    axis: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization with zero-point.

    Maps values to [0, 2^bits - 1] range, useful for tensors with non-symmetric
    distributions (like activations after ReLU).

    Args:
        tensor: Input float tensor to quantize
        bits: Number of bits (4 or 8 typically)
        per_channel: If True, compute scale per channel
        axis: Axis for per-channel quantization

    Returns:
        quantized: Unsigned integer tensor
        scale: Scale factor(s) for dequantization
        zero_point: Zero point offset(s)

    Example:
        >>> activations = torch.relu(torch.randn(32, 64))  # Non-negative
        >>> q, scale, zp = asymmetric_quantize(activations, bits=8)
        >>> print(f"Zero point: {zp}")
    """
    qmax = 2 ** bits - 1
    qmin = 0

    if per_channel:
        min_val = tensor.amin(dim=axis, keepdim=True)
        max_val = tensor.amax(dim=axis, keepdim=True)
    else:
        min_val = tensor.min()
        max_val = tensor.max()

    # Compute scale
    scale = (max_val - min_val) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-10)

    # Compute zero point
    zero_point = torch.round(-min_val / scale).clamp(qmin, qmax)

    # Quantize
    quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)

    return quantized.to(torch.uint8), scale, zero_point


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Dequantize a tensor back to floating point.

    Args:
        quantized: Quantized integer tensor
        scale: Scale factor(s) from quantization
        zero_point: Zero point (for asymmetric quantization)

    Returns:
        Dequantized float tensor

    Example:
        >>> q, scale = symmetric_quantize(weights, bits=4)
        >>> reconstructed = dequantize(q, scale)
    """
    if zero_point is not None:
        return (quantized.float() - zero_point.float()) * scale.float()
    else:
        return quantized.float() * scale.float()


def get_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    metric: str = 'mse'
) -> float:
    """
    Calculate quantization error between original and quantized tensor.

    Args:
        original: Original float tensor
        quantized: Quantized tensor
        scale: Scale factor from quantization
        zero_point: Zero point (for asymmetric)
        metric: Error metric ('mse', 'mae', 'max')

    Returns:
        Error value as float

    Example:
        >>> error = get_quantization_error(weights, q, scale, metric='mse')
        >>> print(f"MSE: {error:.6f}")
    """
    reconstructed = dequantize(quantized, scale, zero_point)
    diff = original - reconstructed

    if metric == 'mse':
        return diff.pow(2).mean().item()
    elif metric == 'mae':
        return diff.abs().mean().item()
    elif metric == 'max':
        return diff.abs().max().item()
    elif metric == 'rmse':
        return diff.pow(2).mean().sqrt().item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_optimal_scale(
    tensor: torch.Tensor,
    bits: int = 8,
    method: str = 'minmax'
) -> torch.Tensor:
    """
    Compute optimal scale factor for quantization.

    Args:
        tensor: Input tensor
        bits: Number of bits
        method: Scale computation method
            - 'minmax': Use min/max values
            - 'percentile': Use 99.9th percentile
            - 'mse': Minimize MSE (grid search)

    Returns:
        Optimal scale factor
    """
    qmax = 2 ** (bits - 1) - 1

    if method == 'minmax':
        return tensor.abs().max() / qmax

    elif method == 'percentile':
        # Use 99.9th percentile to be robust to outliers
        flat = tensor.abs().flatten()
        percentile_val = torch.quantile(flat, 0.999)
        return percentile_val / qmax

    elif method == 'mse':
        # Grid search for optimal scale
        max_val = tensor.abs().max()
        best_scale = max_val / qmax
        best_mse = float('inf')

        for scale_factor in np.linspace(0.5, 1.5, 50):
            scale = (max_val * scale_factor) / qmax
            quantized = torch.round(tensor / scale).clamp(-qmax-1, qmax)
            reconstructed = quantized * scale
            mse = (tensor - reconstructed).pow(2).mean().item()

            if mse < best_mse:
                best_mse = mse
                best_scale = scale

        return best_scale

    else:
        raise ValueError(f"Unknown method: {method}")


def simulate_quantization(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True
) -> torch.Tensor:
    """
    Simulate quantization without actually converting to integers.

    Useful for quantization-aware training (QAT).

    Args:
        tensor: Input tensor
        bits: Number of bits
        symmetric: Use symmetric quantization

    Returns:
        Simulated quantized tensor (still float)
    """
    if symmetric:
        q, scale = symmetric_quantize(tensor, bits)
        return dequantize(q, scale)
    else:
        q, scale, zp = asymmetric_quantize(tensor, bits)
        return dequantize(q, scale, zp)


if __name__ == "__main__":
    # Example usage
    print("Quantization Utils Demo")
    print("=" * 50)

    # Create sample weights
    torch.manual_seed(42)
    weights = torch.randn(64, 64)

    print(f"Original weights: shape={weights.shape}, dtype={weights.dtype}")
    print(f"Original range: [{weights.min():.3f}, {weights.max():.3f}]")

    # INT8 quantization
    q8, scale8 = symmetric_quantize(weights, bits=8)
    error8 = get_quantization_error(weights, q8, scale8, metric='rmse')
    print(f"\nINT8: RMSE={error8:.6f}")

    # INT4 quantization
    q4, scale4 = symmetric_quantize(weights, bits=4)
    error4 = get_quantization_error(weights, q4, scale4, metric='rmse')
    print(f"INT4: RMSE={error4:.6f}")

    print(f"\nINT4 has {error4/error8:.1f}x more error than INT8")
