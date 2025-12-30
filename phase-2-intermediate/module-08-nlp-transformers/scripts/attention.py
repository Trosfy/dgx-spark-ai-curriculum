"""
Attention Mechanism Implementations

This module provides production-ready implementations of various attention mechanisms
used in Transformer models. All implementations are designed to work efficiently on
DGX Spark's unified memory architecture.

Example usage:
    >>> from attention import MultiHeadAttention, scaled_dot_product_attention
    >>>
    >>> # Scaled dot-product attention
    >>> Q = torch.randn(2, 8, 64)  # (batch, seq, dim)
    >>> K = torch.randn(2, 8, 64)
    >>> V = torch.randn(2, 8, 64)
    >>> output, weights = scaled_dot_product_attention(Q, K, V)
    >>>
    >>> # Multi-head attention
    >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
    >>> output = mha(Q, K, V)

Author: DGX Spark AI Curriculum
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention.

    This is the core attention mechanism from "Attention Is All You Need".

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional boolean mask where True = keep, False = mask out
              Shape: (..., seq_len_q, seq_len_k)
        dropout_p: Dropout probability applied to attention weights
        training: Whether in training mode (affects dropout)

    Returns:
        output: Attention-weighted values, shape (..., seq_len_q, d_v)
        attention_weights: Attention weights, shape (..., seq_len_q, seq_len_k)

    Example:
        >>> Q = torch.randn(2, 10, 64)
        >>> K = torch.randn(2, 10, 64)
        >>> V = torch.randn(2, 10, 64)
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
        >>> output.shape
        torch.Size([2, 10, 64])
    """
    d_k = query.size(-1)

    # Compute attention scores: Q @ K^T
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale by sqrt(d_k) to prevent large values
    scores = scores / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        # Use -inf for masked positions (becomes 0 after softmax)
        scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Handle NaN from all-masked rows
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

    # Apply dropout
    if dropout_p > 0.0 and training:
        attention_weights = F.dropout(attention_weights, p=dropout_p)

    # Apply attention to values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Runs multiple attention heads in parallel and concatenates results.
    This allows the model to attend to different representation subspaces.

    Args:
        d_model: Total model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections

    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(2, 10, 512)
        >>> output = mha(x, x, x)  # Self-attention
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)
            nn.init.zeros_(self.W_k.bias)
            nn.init.zeros_(self.W_v.bias)
            nn.init.zeros_(self.W_o.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            query: Query tensor (batch, seq_len_q, d_model)
            key: Key tensor (batch, seq_len_k, d_model)
            value: Value tensor (batch, seq_len_k, d_model)
            mask: Optional attention mask (batch, 1, seq_len_q, seq_len_k)
                  or (batch, num_heads, seq_len_q, seq_len_k)
            need_weights: Whether to return attention weights

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: Optional (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)    # (batch, seq_k, d_model)
        V = self.W_v(value)  # (batch, seq_k, d_model)

        # Reshape for multi-head: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Expand mask for broadcasting if needed
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k)
            scores = scores.masked_fill(~mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )

        # Final projection
        output = self.W_o(context)

        if need_weights:
            return output, attention_weights
        return output, None


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) attention mask.

    Used for autoregressive models like GPT where each position
    can only attend to earlier positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        mask: Boolean tensor of shape (seq_len, seq_len)
              True = can attend, False = cannot attend

    Example:
        >>> mask = create_causal_mask(4)
        >>> mask
        tensor([[ True, False, False, False],
                [ True,  True, False, False],
                [ True,  True,  True, False],
                [ True,  True,  True,  True]])
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths (batch,)
        max_len: Maximum sequence length

    Returns:
        mask: Boolean tensor of shape (batch, max_len)
              True = valid token, False = padding

    Example:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = create_padding_mask(lengths, 6)
        >>> mask
        tensor([[ True,  True,  True, False, False, False],
                [ True,  True,  True,  True,  True, False],
                [ True,  True, False, False, False, False]])
    """
    batch_size = lengths.size(0)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return mask


if __name__ == "__main__":
    # Run tests
    print("Testing attention implementations...")

    # Test scaled dot-product attention
    Q = torch.randn(2, 10, 64)
    K = torch.randn(2, 10, 64)
    V = torch.randn(2, 10, 64)

    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (2, 10, 64), f"Expected (2, 10, 64), got {output.shape}"
    assert weights.shape == (2, 10, 10), f"Expected (2, 10, 10), got {weights.shape}"
    print("  scaled_dot_product_attention: PASSED")

    # Test multi-head attention
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    output, _ = mha(x, x, x)
    assert output.shape == (2, 10, 512), f"Expected (2, 10, 512), got {output.shape}"
    print("  MultiHeadAttention: PASSED")

    # Test causal mask
    mask = create_causal_mask(5)
    assert mask.shape == (5, 5)
    assert mask[0, 1] == False  # Position 0 cannot attend to position 1
    assert mask[2, 1] == True   # Position 2 can attend to position 1
    print("  create_causal_mask: PASSED")

    # Test padding mask
    lengths = torch.tensor([3, 5, 2])
    mask = create_padding_mask(lengths, 6)
    assert mask.shape == (3, 6)
    assert mask[0, 2] == True   # Position 2 is valid for seq 0 (length 3)
    assert mask[0, 3] == False  # Position 3 is padding for seq 0
    print("  create_padding_mask: PASSED")

    print("\nAll tests passed!")
