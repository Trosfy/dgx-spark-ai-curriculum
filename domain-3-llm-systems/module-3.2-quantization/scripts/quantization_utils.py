"""
Quantization Utilities for DGX Spark

This module provides core quantization functions for neural network weights,
including support for FP8 (E4M3/E5M2) and simulated FP4 formats optimized
for DGX Spark's Blackwell architecture.

Features:
- Symmetric and asymmetric quantization
- FP8 format simulation (E4M3 for inference, E5M2 for training)
- FP4 format simulation with micro-block scaling
- Per-channel and per-group quantization
- Calibration utilities for optimal scale finding

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
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass


# =============================================================================
# FP8 Format Constants (Blackwell Native)
# =============================================================================

@dataclass
class FP8Format:
    """Configuration for FP8 floating-point format."""
    name: str
    exponent_bits: int
    mantissa_bits: int
    bias: int
    max_value: float
    min_positive: float

# E4M3: 4 exponent bits, 3 mantissa bits - optimized for inference
FP8_E4M3 = FP8Format(
    name="E4M3",
    exponent_bits=4,
    mantissa_bits=3,
    bias=7,
    max_value=448.0,      # 1.875 * 2^8
    min_positive=2**-9    # Subnormal minimum
)

# E5M2: 5 exponent bits, 2 mantissa bits - optimized for training (larger range)
FP8_E5M2 = FP8Format(
    name="E5M2",
    exponent_bits=5,
    mantissa_bits=2,
    bias=15,
    max_value=57344.0,    # 1.75 * 2^15
    min_positive=2**-16   # Subnormal minimum
)


# =============================================================================
# Basic Quantization Functions
# =============================================================================

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
        max_val = tensor.abs().amax(dim=axis, keepdim=True)
    else:
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


# =============================================================================
# FP8 Quantization (Blackwell Native)
# =============================================================================

def quantize_to_fp8(
    tensor: torch.Tensor,
    format: FP8Format = FP8_E4M3,
    scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 format (simulated for non-Blackwell GPUs).

    On Blackwell GPUs, this maps to native FP8 tensor cores.

    Args:
        tensor: Input tensor (FP16/FP32/BF16)
        format: FP8 format (E4M3 for inference, E5M2 for training)
        scale: Optional pre-computed scale factor

    Returns:
        quantized: Simulated FP8 tensor (stored as FP16/BF16)
        scale: Scale factor used

    Example:
        >>> weights = torch.randn(64, 64, dtype=torch.float16)
        >>> fp8_weights, scale = quantize_to_fp8(weights, format=FP8_E4M3)
        >>> print(f"Max value after FP8: {fp8_weights.abs().max():.1f}")
    """
    # Compute scale if not provided
    if scale is None:
        max_val = tensor.abs().max()
        scale = max_val / format.max_value
        scale = torch.clamp(scale, min=1e-10)

    # Scale tensor to FP8 range
    scaled = tensor / scale

    # Clip to FP8 range
    clipped = torch.clamp(scaled, -format.max_value, format.max_value)

    # Simulate reduced precision by rounding to representable values
    # In real FP8, this happens in hardware
    n_mantissa_bits = format.mantissa_bits
    multiplier = 2 ** n_mantissa_bits

    # Round to nearest representable value
    quantized = torch.round(clipped * multiplier) / multiplier

    return quantized, scale


def dequantize_from_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Dequantize from FP8 back to original scale.

    Args:
        tensor: FP8-quantized tensor
        scale: Scale factor from quantization

    Returns:
        Dequantized tensor
    """
    return tensor * scale


# =============================================================================
# FP4 Quantization (Blackwell Exclusive - NVFP4)
# =============================================================================

@dataclass
class FP4Config:
    """Configuration for NVFP4 micro-block scaling."""
    block_size: int = 16      # Micro-block size (16 is typical)
    num_mantissa_bits: int = 2  # 2 mantissa bits for FP4
    num_exponent_bits: int = 1  # 1 exponent bit for FP4
    use_dual_scaling: bool = True  # Enable dual-level scaling

# NVFP4: 4-bit floating point with micro-block scaling
# Format: 1 sign + 1 exponent + 2 mantissa = 4 bits
# Representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
NVFP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_to_fp4(
    tensor: torch.Tensor,
    block_size: int = 16,
    use_dual_scaling: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize tensor to NVFP4 format with micro-block scaling.

    NVFP4 is exclusive to Blackwell architecture and provides 3.5x memory
    reduction compared to FP16 with minimal quality loss.

    Args:
        tensor: Input tensor (FP16/BF16)
        block_size: Size of micro-blocks for fine-grained scaling
        use_dual_scaling: Enable dual-level scaling (block + tensor)

    Returns:
        quantized: Simulated FP4 tensor
        block_scales: Per-block scale factors
        tensor_scale: Optional tensor-level scale (if dual scaling)

    Example:
        >>> weights = torch.randn(64, 64, dtype=torch.float16)
        >>> fp4_weights, block_scales, tensor_scale = quantize_to_fp4(weights)
        >>> compression = weights.numel() * 2 / (fp4_weights.numel() * 0.5)
        >>> print(f"Compression ratio: {compression:.1f}x")
    """
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()

    # Pad to multiple of block_size
    n = tensor_flat.numel()
    pad_size = (block_size - n % block_size) % block_size
    if pad_size > 0:
        tensor_flat = torch.nn.functional.pad(tensor_flat, (0, pad_size))

    # Reshape into blocks
    n_blocks = tensor_flat.numel() // block_size
    blocks = tensor_flat.view(n_blocks, block_size)

    # Tensor-level scale (coarse)
    tensor_scale = None
    if use_dual_scaling:
        tensor_scale = blocks.abs().max()
        blocks = blocks / torch.clamp(tensor_scale, min=1e-10)

    # Block-level scales (fine-grained)
    block_max = blocks.abs().amax(dim=1, keepdim=True)
    block_scales = block_max / 6.0  # 6.0 is max representable in FP4
    block_scales = torch.clamp(block_scales, min=1e-10)

    # Normalize blocks
    normalized = blocks / block_scales

    # Quantize to nearest FP4 value
    fp4_values = NVFP4_VALUES.to(tensor.device)
    signs = torch.sign(normalized)
    abs_vals = normalized.abs()

    # Find nearest FP4 value for each element
    distances = (abs_vals.unsqueeze(-1) - fp4_values.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)
    quantized = signs * fp4_values[indices]

    # Reshape back, removing padding
    quantized = quantized.flatten()[:n].view(original_shape)
    block_scales = block_scales.squeeze()

    return quantized, block_scales, tensor_scale


def dequantize_from_fp4(
    quantized: torch.Tensor,
    block_scales: torch.Tensor,
    tensor_scale: Optional[torch.Tensor] = None,
    block_size: int = 16
) -> torch.Tensor:
    """
    Dequantize from NVFP4 back to FP16/BF16.

    Args:
        quantized: FP4-quantized tensor
        block_scales: Per-block scale factors
        tensor_scale: Tensor-level scale (if dual scaling was used)
        block_size: Block size used during quantization

    Returns:
        Dequantized tensor
    """
    original_shape = quantized.shape
    quantized_flat = quantized.flatten()

    # Pad to match block structure
    n = quantized_flat.numel()
    pad_size = (block_size - n % block_size) % block_size
    if pad_size > 0:
        quantized_flat = torch.nn.functional.pad(quantized_flat, (0, pad_size))

    # Reshape and apply block scales
    n_blocks = quantized_flat.numel() // block_size
    blocks = quantized_flat.view(n_blocks, block_size)
    dequantized = blocks * block_scales.view(-1, 1)

    # Apply tensor scale
    if tensor_scale is not None:
        dequantized = dequantized * tensor_scale

    # Remove padding and reshape
    dequantized = dequantized.flatten()[:n].view(original_shape)

    return dequantized


# =============================================================================
# Per-Group Quantization (for GPTQ/AWQ style)
# =============================================================================

def group_quantize(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group-wise symmetric quantization (GPTQ/AWQ style).

    Divides weights into groups and computes per-group scales
    for better accuracy than per-tensor quantization.

    Args:
        tensor: Input weight tensor [out_features, in_features]
        bits: Number of bits (typically 4)
        group_size: Size of each quantization group

    Returns:
        quantized: Quantized tensor
        scales: Per-group scale factors
        zeros: Zero points (for asymmetric, otherwise None)

    Example:
        >>> weights = torch.randn(4096, 4096)
        >>> q, scales, zeros = group_quantize(weights, bits=4, group_size=128)
        >>> print(f"Scales shape: {scales.shape}")  # [4096, 32]
    """
    out_features, in_features = tensor.shape

    # Ensure in_features is divisible by group_size
    assert in_features % group_size == 0, \
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"

    n_groups = in_features // group_size
    qmax = 2 ** (bits - 1) - 1

    # Reshape for group-wise operation
    tensor_grouped = tensor.view(out_features, n_groups, group_size)

    # Compute per-group scales
    group_max = tensor_grouped.abs().amax(dim=2)  # [out_features, n_groups]
    scales = group_max / qmax
    scales = torch.clamp(scales, min=1e-10)

    # Quantize
    quantized = torch.round(
        tensor_grouped / scales.unsqueeze(-1)
    ).clamp(-qmax-1, qmax)

    # Reshape back
    quantized = quantized.view(out_features, in_features).to(torch.int8)

    return quantized, scales, None


def group_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    group_size: int = 128
) -> torch.Tensor:
    """
    Dequantize group-quantized tensor.

    Args:
        quantized: Group-quantized tensor
        scales: Per-group scales
        zeros: Per-group zero points (optional)
        group_size: Group size used during quantization

    Returns:
        Dequantized tensor
    """
    out_features, in_features = quantized.shape
    n_groups = in_features // group_size

    # Reshape
    q_grouped = quantized.view(out_features, n_groups, group_size).float()

    # Apply zero point if present
    if zeros is not None:
        q_grouped = q_grouped - zeros.unsqueeze(-1)

    # Apply scales
    dequantized = q_grouped * scales.unsqueeze(-1)

    return dequantized.view(out_features, in_features)


# =============================================================================
# Calibration Utilities
# =============================================================================

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
        flat = tensor.abs().flatten()
        percentile_val = torch.quantile(flat.float(), 0.999)
        return percentile_val / qmax

    elif method == 'mse':
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
        metric: Error metric ('mse', 'mae', 'max', 'rmse')

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


# =============================================================================
# Comparison and Analysis
# =============================================================================

def compare_quantization_methods(
    tensor: torch.Tensor,
    methods: list = ['int8', 'int4', 'fp8_e4m3', 'fp4']
) -> Dict[str, Dict[str, float]]:
    """
    Compare different quantization methods on the same tensor.

    Args:
        tensor: Input tensor to quantize
        methods: List of methods to compare

    Returns:
        Dict with error metrics for each method

    Example:
        >>> weights = torch.randn(64, 64)
        >>> results = compare_quantization_methods(weights)
        >>> for method, metrics in results.items():
        ...     print(f"{method}: RMSE={metrics['rmse']:.6f}")
    """
    results = {}

    for method in methods:
        if method == 'int8':
            q, s = symmetric_quantize(tensor, bits=8)
            recon = dequantize(q, s)
        elif method == 'int4':
            q, s = symmetric_quantize(tensor, bits=4)
            recon = dequantize(q, s)
        elif method == 'fp8_e4m3':
            q, s = quantize_to_fp8(tensor, format=FP8_E4M3)
            recon = dequantize_from_fp8(q, s)
        elif method == 'fp8_e5m2':
            q, s = quantize_to_fp8(tensor, format=FP8_E5M2)
            recon = dequantize_from_fp8(q, s)
        elif method == 'fp4':
            q, bs, ts = quantize_to_fp4(tensor)
            recon = dequantize_from_fp4(q, bs, ts)
        else:
            continue

        diff = tensor - recon
        results[method] = {
            'mse': diff.pow(2).mean().item(),
            'rmse': diff.pow(2).mean().sqrt().item(),
            'mae': diff.abs().mean().item(),
            'max_error': diff.abs().max().item(),
            'snr_db': 10 * np.log10(
                tensor.pow(2).mean().item() / (diff.pow(2).mean().item() + 1e-10)
            )
        }

    return results


if __name__ == "__main__":
    print("Quantization Utils Demo")
    print("=" * 60)

    # Create sample weights
    torch.manual_seed(42)
    weights = torch.randn(64, 64)

    print(f"Original weights: shape={weights.shape}, dtype={weights.dtype}")
    print(f"Original range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Compare all methods
    print("\nComparing quantization methods:")
    print("-" * 60)

    results = compare_quantization_methods(weights)
    for method, metrics in results.items():
        print(f"{method:12s}: RMSE={metrics['rmse']:.6f}, SNR={metrics['snr_db']:.1f} dB")

    # Demo FP4 specifically
    print("\n" + "=" * 60)
    print("NVFP4 Demo (Blackwell Exclusive)")
    print("=" * 60)

    fp4_q, block_scales, tensor_scale = quantize_to_fp4(weights)
    fp4_recon = dequantize_from_fp4(fp4_q, block_scales, tensor_scale)

    # Memory calculation
    original_bytes = weights.numel() * 4  # FP32
    fp4_bytes = weights.numel() * 0.5 + block_scales.numel() * 2  # FP4 + scales

    print(f"Original size: {original_bytes} bytes")
    print(f"FP4 size: {fp4_bytes:.0f} bytes")
    print(f"Compression ratio: {original_bytes / fp4_bytes:.1f}x")
    print(f"RMSE: {(weights - fp4_recon).pow(2).mean().sqrt():.6f}")
