# Understanding the Transformer Architecture

## Introduction

The Transformer architecture, introduced in the seminal 2017 paper "Attention Is All You Need" by Vaswani et al., revolutionized natural language processing and became the foundation for modern large language models. This document provides a comprehensive explanation of the architecture and its components.

## The Attention Mechanism

### Self-Attention Basics
At the heart of the Transformer is the self-attention mechanism, which allows each position in a sequence to attend to all other positions. This enables the model to capture long-range dependencies that were difficult for previous architectures like RNNs and LSTMs.

The attention mechanism computes three vectors for each input token:
- **Query (Q)**: What the token is looking for
- **Key (K)**: What information the token contains
- **Value (V)**: The actual information to be passed forward

### Computing Attention
The attention score between positions is computed as:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- Q, K, V are the query, key, and value matrices
- d_k is the dimension of the key vectors
- The softmax ensures attention weights sum to 1

### Multi-Head Attention
Instead of performing a single attention function, Transformers use multi-head attention. This allows the model to jointly attend to information from different representation subspaces at different positions.

For example, with 8 attention heads:
- Each head learns different aspects of relationships
- One head might focus on syntactic relationships
- Another might focus on semantic relationships
- Results are concatenated and linearly transformed

## Positional Encoding

Since Transformers process all positions in parallel (unlike RNNs which process sequentially), they need explicit positional information. This is achieved through positional encoding.

### Sinusoidal Encoding
The original Transformer uses sinusoidal functions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Learned Positional Embeddings
Many modern models like BERT and GPT use learned positional embeddings, where positions are represented as trainable vectors.

### Rotary Position Embedding (RoPE)
RoPE, used in models like LLaMA, encodes positions by rotating the query and key vectors, offering better extrapolation to longer sequences.

## The Encoder

The encoder consists of a stack of identical layers, each containing:

1. **Multi-Head Self-Attention**: Allows each position to attend to all positions in the input
2. **Feed-Forward Network**: A simple two-layer network applied to each position
3. **Layer Normalization**: Stabilizes training by normalizing activations
4. **Residual Connections**: Allow gradients to flow directly through the network

### Encoder Stack
Typically, the encoder consists of 6-12 identical layers stacked on top of each other. Each layer refines the representations from the previous layer.

## The Decoder

The decoder is similar to the encoder but includes an additional attention layer:

1. **Masked Self-Attention**: Prevents positions from attending to future positions (crucial for autoregressive generation)
2. **Cross-Attention**: Attends to the encoder output
3. **Feed-Forward Network**: Same as encoder
4. **Layer Normalization and Residual Connections**

### Causal Masking
In the masked self-attention layer, a mask prevents each position from attending to subsequent positions. This ensures the prediction for position i can only depend on known outputs at positions less than i.

## Feed-Forward Networks

Each Transformer layer contains a position-wise feed-forward network:

```python
FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
```

Modern variants often use:
- GELU or SiLU activation functions
- Gated Linear Units (GLU)
- SwiGLU (used in LLaMA)

The intermediate dimension is typically 4x the model dimension.

## Layer Normalization

Layer normalization is applied to stabilize training:

```python
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + epsilon) + beta
```

### Pre-Norm vs Post-Norm
- **Post-Norm** (original): LayerNorm after attention/FFN
- **Pre-Norm** (modern): LayerNorm before attention/FFN
  - More stable training
  - Used in GPT-2, LLaMA, and most modern LLMs

## Variants and Improvements

### Encoder-Only Models
- **BERT**: Bidirectional encoder for understanding tasks
- **RoBERTa**: Optimized BERT training
- **DeBERTa**: Disentangled attention

### Decoder-Only Models
- **GPT series**: Autoregressive language modeling
- **LLaMA**: Meta's open-source models
- **Mistral**: Efficient decoder with sliding window attention

### Encoder-Decoder Models
- **T5**: Text-to-text framework
- **BART**: Denoising autoencoder
- **mBART**: Multilingual BART

## Training Considerations

### Scaling Laws
Transformer performance scales predictably with:
- Model size (parameters)
- Dataset size (tokens)
- Compute budget (FLOPs)

The Chinchilla scaling laws suggest optimal allocation of compute between model size and data size.

### Optimization
Standard training uses:
- Adam or AdamW optimizer
- Learning rate warmup
- Cosine learning rate decay
- Gradient clipping

## Conclusion

The Transformer architecture's ability to capture long-range dependencies through attention, combined with its parallelizable design, has made it the dominant architecture in NLP and increasingly in computer vision and other domains.
