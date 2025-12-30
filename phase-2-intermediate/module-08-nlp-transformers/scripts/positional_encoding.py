"""
Positional Encoding Implementations

This module provides various positional encoding strategies used in
Transformer models, from the original sinusoidal encoding to modern
approaches like RoPE and ALiBi.

Example usage:
    >>> from positional_encoding import SinusoidalPositionalEncoding, RoPE
    >>>
    >>> # Sinusoidal
    >>> pe = SinusoidalPositionalEncoding(d_model=512)
    >>> x = torch.randn(2, 100, 512)
    >>> x = pe(x)  # Adds position info
    >>>
    >>> # RoPE
    >>> rope = RoPE(dim=64)
    >>> q = torch.randn(2, 8, 100, 64)  # (batch, heads, seq, dim)
    >>> k = torch.randn(2, 8, 100, 64)
    >>> q_rot, k_rot = rope(q, k, seq_len=100)

Author: DGX Spark AI Curriculum
License: MIT
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".

    Uses sin and cos functions at different frequencies to encode position.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length to pre-compute
        dropout: Dropout probability

    Example:
        >>> pe = SinusoidalPositionalEncoding(512, max_len=1000)
        >>> x = torch.randn(2, 100, 512)
        >>> x = pe(x)  # Shape: (2, 100, 512)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output with position info added (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embeddings.

    Each position has a learnable embedding vector.
    Used in GPT-2, BERT, and other models.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability

    Example:
        >>> pe = LearnedPositionalEmbedding(512, max_len=512)
        >>> x = torch.randn(2, 100, 512)
        >>> x = pe(x)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output with position embeddings added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.embedding(positions)
        x = x + pos_emb
        return self.dropout(x)


class RoPE(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Used in LLaMA, Mistral, and many modern LLMs.
    Applies rotation to Q and K based on position, making the
    dot product naturally encode relative position.

    Args:
        dim: Dimension of the embeddings (must be even)
        max_len: Maximum sequence length
        base: Base for frequency calculation

    Example:
        >>> rope = RoPE(dim=64, max_len=2048)
        >>> q = torch.randn(2, 8, 100, 64)  # (batch, heads, seq, dim)
        >>> k = torch.randn(2, 8, 100, 64)
        >>> q_rot, k_rot = rope(q, k, seq_len=100)
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 2048,
        base: int = 10000
    ):
        super().__init__()

        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute cos and sin
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor (batch, heads, seq_len, dim)
            k: Key tensor (batch, heads, seq_len, dim)
            seq_len: Sequence length (inferred if not provided)

        Returns:
            Rotated q and k tensors
        """
        if seq_len is None:
            seq_len = q.size(2)

        if seq_len > self.max_len:
            self._build_cache(seq_len)
            self.max_len = seq_len

        cos = self.cos_cached[:seq_len].to(q.dtype)
        sin = self.sin_cached[:seq_len].to(q.dtype)

        # Reshape for broadcasting: (1, 1, seq, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Adds a linear bias based on relative position to attention scores.
    Used in BLOOM, MPT, and other models. Known for excellent
    length extrapolation properties.

    Args:
        num_heads: Number of attention heads

    Example:
        >>> alibi = ALiBi(num_heads=8)
        >>> bias = alibi(seq_len=100)  # (num_heads, 100, 100)
        >>> attention_scores = attention_scores + bias.unsqueeze(0)
    """

    def __init__(self, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', torch.tensor(slopes).float())

    def _get_slopes(self, num_heads: int) -> list:
        """Compute ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-8 / n)
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra = get_slopes_power_of_2(2 * closest_power_of_2)[0::2]
            return slopes + extra[:num_heads - closest_power_of_2]

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute ALiBi bias matrix.

        Args:
            seq_len: Sequence length

        Returns:
            Bias tensor (num_heads, seq_len, seq_len)
        """
        positions = torch.arange(seq_len, device=self.slopes.device)
        relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        bias = -torch.abs(relative_pos).float()
        bias = bias.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
        return bias


class NTKAwareRoPE(RoPE):
    """
    NTK-aware Rotary Position Embeddings.

    Extends RoPE for better length extrapolation by scaling
    the base frequency.

    Args:
        dim: Dimension of the embeddings
        max_len: Maximum sequence length during training
        target_len: Target sequence length for extrapolation
        base: Original base frequency

    Example:
        >>> rope = NTKAwareRoPE(dim=64, max_len=2048, target_len=8192)
        >>> q_rot, k_rot = rope(q, k)
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 2048,
        target_len: int = 8192,
        base: int = 10000
    ):
        # Scale the base for NTK interpolation
        scale = target_len / max_len
        new_base = base * (scale ** (dim / (dim - 2)))

        super().__init__(dim=dim, max_len=target_len, base=int(new_base))


def visualize_positional_encoding(
    encoding_type: str = 'sinusoidal',
    d_model: int = 64,
    max_len: int = 100
) -> torch.Tensor:
    """
    Generate positional encoding for visualization.

    Args:
        encoding_type: 'sinusoidal' or 'learned'
        d_model: Model dimension
        max_len: Sequence length

    Returns:
        Positional encoding tensor (max_len, d_model)
    """
    if encoding_type == 'sinusoidal':
        pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
        x = torch.zeros(1, max_len, d_model)
        return pe(x)[0]  # (max_len, d_model)
    elif encoding_type == 'learned':
        pe = LearnedPositionalEmbedding(d_model, max_len, dropout=0.0)
        x = torch.zeros(1, max_len, d_model)
        return pe(x)[0]
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


if __name__ == "__main__":
    print("Testing positional encoding implementations...")

    # Test Sinusoidal
    pe = SinusoidalPositionalEncoding(512, max_len=1000)
    x = torch.randn(2, 100, 512)
    out = pe(x)
    assert out.shape == (2, 100, 512)
    print("  SinusoidalPositionalEncoding: PASSED")

    # Test Learned
    pe = LearnedPositionalEmbedding(512, max_len=512)
    out = pe(x)
    assert out.shape == (2, 100, 512)
    print("  LearnedPositionalEmbedding: PASSED")

    # Test RoPE
    rope = RoPE(dim=64, max_len=2048)
    q = torch.randn(2, 8, 100, 64)
    k = torch.randn(2, 8, 100, 64)
    q_rot, k_rot = rope(q, k, seq_len=100)
    assert q_rot.shape == (2, 8, 100, 64)
    assert k_rot.shape == (2, 8, 100, 64)
    print("  RoPE: PASSED")

    # Test RoPE extrapolation
    q_long = torch.randn(2, 8, 4096, 64)
    k_long = torch.randn(2, 8, 4096, 64)
    q_rot, k_rot = rope(q_long, k_long, seq_len=4096)
    assert q_rot.shape == (2, 8, 4096, 64)
    print("  RoPE (extrapolation): PASSED")

    # Test ALiBi
    alibi = ALiBi(num_heads=8)
    bias = alibi(seq_len=100)
    assert bias.shape == (8, 100, 100)
    print("  ALiBi: PASSED")

    # Test NTK-aware RoPE
    ntk_rope = NTKAwareRoPE(dim=64, max_len=2048, target_len=8192)
    q_rot, k_rot = ntk_rope(q, k, seq_len=100)
    assert q_rot.shape == (2, 8, 100, 64)
    print("  NTKAwareRoPE: PASSED")

    print("\nAll tests passed!")
