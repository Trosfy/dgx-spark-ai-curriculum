"""
Transformer Building Blocks

This module provides modular Transformer components that can be combined
to build encoder-only (BERT), decoder-only (GPT), or encoder-decoder (T5)
architectures.

Example usage:
    >>> from transformer import TransformerEncoder, TransformerDecoder
    >>>
    >>> # Build BERT-style encoder
    >>> encoder = TransformerEncoder(
    ...     num_layers=12,
    ...     d_model=768,
    ...     num_heads=12,
    ...     d_ff=3072
    ... )
    >>>
    >>> x = torch.randn(2, 128, 768)
    >>> output = encoder(x)

Author: DGX Spark AI Curriculum
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
try:
    # When imported as part of package
    from .attention import MultiHeadAttention, create_causal_mask
except ImportError:
    # When run directly
    from attention import MultiHeadAttention, create_causal_mask


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = activation(xW1 + b1)W2 + b2

    Args:
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'swiglu')

    Example:
        >>> ffn = FeedForward(512, 2048)
        >>> x = torch.randn(2, 10, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.activation = activation

        if activation == 'swiglu':
            # SwiGLU uses 2/3 the dimension due to the gate
            d_ff = int(d_ff * 2 / 3)
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)
            self.w3 = nn.Linear(d_model, d_ff)  # Gate
        else:
            self.w1 = nn.Linear(d_model, d_ff)
            self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.activation == 'swiglu':
            # SwiGLU: gate * up
            gate = F.silu(self.w1(x))
            up = self.w3(x)
            x = gate * up
        elif self.activation == 'gelu':
            x = F.gelu(self.w1(x))
        else:  # relu
            x = F.relu(self.w1(x))

        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Architecture (Pre-LN):
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> + ->
        -> LayerNorm -> FeedForward -> Dropout -> + -> output

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: FFN activation
        pre_norm: Use Pre-LN (True) or Post-LN (False)

    Example:
        >>> layer = TransformerEncoderLayer(512, 8)
        >>> x = torch.randn(2, 10, 512)
        >>> output = layer(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.pre_norm:
            # Pre-LN: Norm -> Sublayer -> Residual
            residual = x
            x = self.norm1(x)
            x, _ = self.self_attn(x, x, x, mask)
            x = residual + self.dropout1(x)

            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = residual + self.dropout2(x)
        else:
            # Post-LN: Sublayer -> Residual -> Norm
            residual = x
            x, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(residual + self.dropout1(x))

            residual = x
            x = self.ffn(x)
            x = self.norm2(residual + self.dropout2(x))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Includes self-attention with causal mask and cross-attention to encoder.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: FFN activation
        pre_norm: Use Pre-LN (True) or Post-LN (False)

    Example:
        >>> layer = TransformerDecoderLayer(512, 8)
        >>> x = torch.randn(2, 10, 512)
        >>> encoder_output = torch.randn(2, 20, 512)
        >>> output = layer(x, encoder_output)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.pre_norm = pre_norm

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Encoder output (batch, src_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Output tensor (batch, tgt_len, d_model)
        """
        if self.pre_norm:
            # Self-attention
            residual = x
            x = self.norm1(x)
            x, _ = self.self_attn(x, x, x, self_attn_mask)
            x = residual + self.dropout1(x)

            # Cross-attention
            residual = x
            x = self.norm2(x)
            x, _ = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
            x = residual + self.dropout2(x)

            # Feed-forward
            residual = x
            x = self.norm3(x)
            x = self.ffn(x)
            x = residual + self.dropout3(x)
        else:
            # Post-LN variant
            residual = x
            x, _ = self.self_attn(x, x, x, self_attn_mask)
            x = self.norm1(residual + self.dropout1(x))

            residual = x
            x, _ = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
            x = self.norm2(residual + self.dropout2(x))

            residual = x
            x = self.ffn(x)
            x = self.norm3(residual + self.dropout3(x))

        return x


class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder (stack of encoder layers).

    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: FFN activation
        pre_norm: Use Pre-LN architecture

    Example:
        >>> encoder = TransformerEncoder(
        ...     num_layers=6,
        ...     d_model=512,
        ...     num_heads=8
        ... )
        >>> x = torch.randn(2, 100, 512)
        >>> output = encoder(x)
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])

        # Final layer norm for Pre-LN architecture
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """
    Full Transformer Decoder (stack of decoder layers).

    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: FFN activation
        pre_norm: Use Pre-LN architecture

    Example:
        >>> decoder = TransformerDecoder(
        ...     num_layers=6,
        ...     d_model=512,
        ...     num_heads=8
        ... )
        >>> x = torch.randn(2, 50, 512)
        >>> encoder_output = torch.randn(2, 100, 512)
        >>> output = decoder(x, encoder_output)
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Encoder output (batch, src_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Output tensor (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        return self.final_norm(x)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_usage(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    dtype_bytes: int = 2  # bfloat16
) -> dict:
    """
    Estimate memory usage for a Transformer model.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dtype_bytes: Bytes per element (2 for bfloat16, 4 for float32)

    Returns:
        Dictionary with memory breakdown in GB
    """
    # Parameters per layer
    attention_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ff  # Up and down projections
    norm_params = 4 * d_model  # 2 layer norms

    total_params = num_layers * (attention_params + ffn_params + norm_params)
    param_memory = total_params * dtype_bytes

    # Activations per layer (for backward pass)
    input_act = batch_size * seq_len * d_model * dtype_bytes
    attn_scores = batch_size * num_heads * seq_len * seq_len * dtype_bytes
    ffn_hidden = batch_size * seq_len * d_ff * dtype_bytes

    activation_memory = num_layers * (input_act + attn_scores + ffn_hidden)

    # Optimizer states (Adam: 2x for moment estimates)
    optimizer_memory = 2 * param_memory

    return {
        'parameters': total_params,
        'param_memory_gb': param_memory / 1e9,
        'activation_memory_gb': activation_memory / 1e9,
        'optimizer_memory_gb': optimizer_memory / 1e9,
        'total_memory_gb': (param_memory + activation_memory + optimizer_memory) / 1e9
    }


if __name__ == "__main__":
    print("Testing Transformer components...")

    # Test FeedForward
    ffn = FeedForward(512, 2048, activation='gelu')
    x = torch.randn(2, 10, 512)
    out = ffn(x)
    assert out.shape == (2, 10, 512)
    print("  FeedForward: PASSED")

    # Test SwiGLU
    ffn_swiglu = FeedForward(512, 2048, activation='swiglu')
    out = ffn_swiglu(x)
    assert out.shape == (2, 10, 512)
    print("  FeedForward (SwiGLU): PASSED")

    # Test Encoder Layer
    enc_layer = TransformerEncoderLayer(512, 8)
    out = enc_layer(x)
    assert out.shape == (2, 10, 512)
    print("  TransformerEncoderLayer: PASSED")

    # Test Full Encoder
    encoder = TransformerEncoder(6, 512, 8)
    out = encoder(x)
    assert out.shape == (2, 10, 512)
    print("  TransformerEncoder: PASSED")

    # Test Decoder Layer
    dec_layer = TransformerDecoderLayer(512, 8)
    encoder_out = torch.randn(2, 20, 512)
    out = dec_layer(x, encoder_out)
    assert out.shape == (2, 10, 512)
    print("  TransformerDecoderLayer: PASSED")

    # Test Full Decoder
    decoder = TransformerDecoder(6, 512, 8)
    out = decoder(x, encoder_out)
    assert out.shape == (2, 10, 512)
    print("  TransformerDecoder: PASSED")

    # Memory estimation
    mem = estimate_memory_usage(
        batch_size=32,
        seq_len=512,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072
    )
    print(f"\n  BERT-base memory estimate: {mem['total_memory_gb']:.2f} GB")

    print("\nAll tests passed!")
