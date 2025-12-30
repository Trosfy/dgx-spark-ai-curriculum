"""
NLP & Transformers utility modules.

This package provides implementations of attention mechanisms,
transformer blocks, positional encodings, and text generation utilities
optimized for DGX Spark's unified memory architecture.

Example usage:
    >>> from scripts import MultiHeadAttention, TransformerEncoder
    >>> from scripts import SinusoidalPositionalEncoding, RoPE
    >>> from scripts import top_p_sampling, beam_search
    >>> from scripts import SimpleBPE, compare_tokenizers

Author: DGX Spark AI Curriculum
License: MIT
"""

from .attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    create_causal_mask,
    create_padding_mask
)
from .transformer import (
    FeedForward,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    count_parameters,
    estimate_memory_usage
)
from .positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RoPE,
    ALiBi,
    NTKAwareRoPE,
    visualize_positional_encoding
)
from .generation import (
    greedy_decode,
    sample_with_temperature,
    top_k_sampling,
    top_p_sampling,
    top_p_with_repetition_penalty,
    beam_search,
    contrastive_search,
    generate_multiple
)
from .tokenizer_utils import (
    SimpleBPE,
    compare_tokenizers,
    analyze_tokenization,
    estimate_token_cost,
    chunk_text_by_tokens,
    get_special_tokens_info
)

__all__ = [
    # Attention
    'scaled_dot_product_attention',
    'MultiHeadAttention',
    'create_causal_mask',
    'create_padding_mask',
    # Transformer
    'FeedForward',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'count_parameters',
    'estimate_memory_usage',
    # Positional Encoding
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEmbedding',
    'RoPE',
    'ALiBi',
    'NTKAwareRoPE',
    'visualize_positional_encoding',
    # Generation
    'greedy_decode',
    'sample_with_temperature',
    'top_k_sampling',
    'top_p_sampling',
    'top_p_with_repetition_penalty',
    'beam_search',
    'contrastive_search',
    'generate_multiple',
    # Tokenizer
    'SimpleBPE',
    'compare_tokenizers',
    'analyze_tokenization',
    'estimate_token_cost',
    'chunk_text_by_tokens',
    'get_special_tokens_info',
]
