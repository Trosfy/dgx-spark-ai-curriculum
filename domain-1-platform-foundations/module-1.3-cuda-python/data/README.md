# Data Files for Module 1.3: CUDA Python & GPU Programming

This directory contains data files used in the CUDA Python labs.

## Overview

This module primarily uses **synthetic data** generated programmatically within each notebook. This approach ensures:
- Consistent benchmarking across different systems
- No external data dependencies
- Reproducible results with random seeds

## Data Generated Per Lab

| Lab | Data Type | Typical Size | Description |
|-----|-----------|--------------|-------------|
| 1.3.1 | Float arrays | 10M-50M elements | Random arrays for reduction benchmarks |
| 1.3.2 | Matrices | 1024×1024 to 4096×4096 | Random matrices for matmul benchmarks |
| 1.3.3 | Embedding tables | 50K-128K vocab × 768-4096 dim | Simulated LLM embedding tables |
| 1.3.4 | Tabular data | 1M rows × 100 features | Random data for preprocessing |
| 1.3.5 | Image data | 64-128 batch × 224×224×3 | Synthetic CIFAR-like images |

## Data Generation Examples

```python
import numpy as np

# Large array for reduction benchmarks
data = np.random.randn(10_000_000).astype(np.float32)

# Matrices for multiplication
A = np.random.randn(2048, 2048).astype(np.float32)
B = np.random.randn(2048, 2048).astype(np.float32)

# Embedding table
vocab_size = 50000
embed_dim = 768
embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32)
```

## Memory Considerations

On DGX Spark with 128GB unified memory:
- 10M float32 array: ~40 MB
- 2048×2048 float32 matrix: ~16 MB
- 50K×768 embedding table: ~146 MB

All lab data fits comfortably in memory with room for GPU operations.
