"""
LoRA Utilities for LLM Fine-Tuning
==================================

This module provides utilities for working with LoRA (Low-Rank Adaptation) layers.

Author: DGX Spark AI Curriculum
Module: 10 - Large Language Model Fine-Tuning
"""

__all__ = [
    'LoRALayer',
    'add_lora_to_model',
    'count_trainable_parameters',
    'get_lora_state_dict',
    'load_lora_state_dict',
    'analyze_lora_updates',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class LoRALayer(nn.Module):
    """
    Production-ready LoRA layer implementation.

    Implements Low-Rank Adaptation as described in:
    "LoRA: Low-Rank Adaptation of Large Language Models"
    (https://arxiv.org/abs/2106.09685)

    Args:
        original_layer: The nn.Linear layer to adapt
        rank: Rank of the low-rank decomposition (r)
        alpha: Scaling factor for LoRA updates
        dropout: Dropout probability for LoRA path

    Example:
        >>> linear = nn.Linear(4096, 4096)
        >>> lora = LoRALayer(linear, rank=16, alpha=32)
        >>> x = torch.randn(8, 4096)
        >>> output = lora(x)  # Shape: (8, 4096)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA matrices
        # A: projection to low-rank space (in_features -> rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        # B: projection from low-rank space (rank -> out_features)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        # B stays zero - important for stable training start!

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (batch, ..., in_features)

        Returns:
            Output tensor of shape (batch, ..., out_features)
        """
        # Original path
        result = self.original_layer(x)

        # LoRA path
        # Apply dropout to input
        lora_x = self.dropout(x)
        # Down projection: x @ A.T
        lora_output = lora_x @ self.lora_A.T
        # Up projection: result @ B.T
        lora_output = lora_output @ self.lora_B.T

        # Combine with scaling
        result = result + self.scaling * lora_output

        return result

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into original layer for efficient inference.

        After merging, the layer behaves like a regular nn.Linear
        with no additional computation overhead.
        """
        with torch.no_grad():
            # W = W_0 + scaling * B @ A
            self.original_layer.weight.add_(
                self.scaling * (self.lora_B @ self.lora_A)
            )

    def unmerge_weights(self) -> None:
        """
        Unmerge LoRA weights (reverse of merge_weights).

        Useful if you need to continue training after merging.
        """
        with torch.no_grad():
            self.original_layer.weight.sub_(
                self.scaling * (self.lora_B @ self.lora_A)
            )

    def get_delta_weight(self) -> torch.Tensor:
        """
        Get the weight update ΔW = scaling * B @ A.

        Useful for analysis and visualization.
        """
        with torch.no_grad():
            return self.scaling * (self.lora_B @ self.lora_A)

    @property
    def trainable_params(self) -> int:
        """Number of trainable parameters (just LoRA)."""
        return self.lora_A.numel() + self.lora_B.numel()

    @property
    def total_params(self) -> int:
        """Total parameters including frozen base."""
        return self.original_layer.weight.numel() + self.trainable_params


def add_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Add LoRA adapters to specified modules in a model.

    Args:
        model: The model to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout for LoRA path
        target_modules: List of module names to adapt (e.g., ['q_proj', 'v_proj'])
                       If None, adapts all Linear layers

    Returns:
        Modified model with LoRA layers

    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("bert-base")
        >>> model = add_lora_to_model(model, rank=8, target_modules=['query', 'value'])
    """
    if target_modules is None:
        target_modules = []  # Will adapt all Linear layers

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be adapted
            if target_modules and not any(t in name for t in target_modules):
                continue

            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)

            # Replace with LoRA layer
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)

    return model


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count trainable and total parameters in a model.

    Args:
        model: The model to analyze

    Returns:
        Tuple of (trainable_params, total_params, trainable_ratio)

    Example:
        >>> trainable, total, ratio = count_trainable_parameters(model)
        >>> print(f"Trainable: {trainable:,} ({ratio:.2%})")
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = trainable / total if total > 0 else 0

    return trainable, total, ratio


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA weights from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        State dict containing only LoRA parameters

    Example:
        >>> lora_weights = get_lora_state_dict(model)
        >>> torch.save(lora_weights, 'lora_adapter.pt')
    """
    lora_state_dict = {}

    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data.clone()

    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
) -> None:
    """
    Load LoRA weights into a model.

    Args:
        model: Model with LoRA layers
        lora_state_dict: State dict with LoRA parameters

    Example:
        >>> lora_weights = torch.load('lora_adapter.pt')
        >>> load_lora_state_dict(model, lora_weights)
    """
    model_state = model.state_dict()

    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")

    model.load_state_dict(model_state)


def analyze_lora_updates(model: nn.Module) -> Dict[str, Dict]:
    """
    Analyze LoRA weight updates for debugging and visualization.

    Args:
        model: Model with LoRA layers

    Returns:
        Dictionary with analysis for each LoRA layer

    Example:
        >>> analysis = analyze_lora_updates(model)
        >>> for name, stats in analysis.items():
        ...     print(f"{name}: norm={stats['delta_norm']:.4f}")
    """
    analysis = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            delta_w = module.get_delta_weight()

            analysis[name] = {
                'rank': module.rank,
                'alpha': module.alpha,
                'scaling': module.scaling,
                'delta_norm': delta_w.norm().item(),
                'delta_mean': delta_w.mean().item(),
                'delta_std': delta_w.std().item(),
                'A_norm': module.lora_A.norm().item(),
                'B_norm': module.lora_B.norm().item(),
                'trainable_params': module.trainable_params,
            }

    return analysis


if __name__ == "__main__":
    # Example usage
    print("LoRA Utilities Demo")
    print("=" * 50)

    # Create a simple linear layer
    linear = nn.Linear(512, 512, bias=False)
    print(f"Original layer parameters: {linear.weight.numel():,}")

    # Add LoRA
    lora = LoRALayer(linear, rank=16, alpha=32)
    print(f"LoRA trainable parameters: {lora.trainable_params:,}")
    print(f"Compression: {lora.trainable_params / lora.total_params:.4%}")

    # Test forward pass
    x = torch.randn(8, 512)
    output = lora(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Analyze
    print(f"\nΔW norm: {lora.get_delta_weight().norm().item():.6f}")

    print("\nDemo complete!")
