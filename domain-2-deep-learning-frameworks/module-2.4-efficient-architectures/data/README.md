# Data Directory - Module 2.4: Efficient Architectures

## Overview

This module primarily uses pre-built models and dynamically generated/downloaded data for benchmarking. No large static datasets are pre-included.

## Datasets Used in Labs

| Dataset | Source | Size | Use Case | Lab |
|---------|--------|------|----------|-----|
| WikiText-2 | HuggingFace | ~2MB | Perplexity evaluation | 2.4.5 |
| WikiText-103 | HuggingFace | ~516MB | Extended perplexity testing | 2.4.5 |
| PG19 | HuggingFace | ~11GB | Long document benchmarks | 2.4.1 |
| Alpaca | HuggingFace | ~50MB | Instruction fine-tuning | 2.4.6 |
| Custom prompts | Generated | Small | Expert analysis | 2.4.3, 2.4.4 |

## Loading Datasets

### WikiText for Perplexity
```python
from datasets import load_dataset

# Lightweight version for quick testing
wikitext2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Full version for comprehensive evaluation
wikitext103 = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
```

### Long Document Benchmark (PG19)
```python
from datasets import load_dataset

# PG19 contains full books - perfect for testing long context
pg19 = load_dataset("pg19", split="test[:10]")  # First 10 books

# Extract a single long document
long_text = pg19[0]["text"]  # Usually 50K+ tokens
print(f"Document length: {len(long_text)} characters")
```

### Expert Analysis Prompts
```python
# Diverse prompts to analyze MoE expert activation patterns
expert_test_prompts = [
    # Code-related (expect specific experts)
    "def fibonacci(n):\n    '''Calculate the nth Fibonacci number'''\n    if n <= 1:",

    # Math-related
    "The integral of x^2 dx equals",

    # Creative writing
    "In the depths of the enchanted forest, a mysterious light appeared",

    # Scientific explanation
    "Photosynthesis is the process by which plants convert",

    # Logical reasoning
    "If all mammals are warm-blooded and whales are mammals, then",

    # Translation-like
    "Translate to French: The cat sat on the mat",

    # Summarization
    "TL;DR: The paper presents a novel approach to neural network training",
]
```

### Instruction Dataset for Fine-tuning (Lab 2.4.6)
```python
from datasets import load_dataset

# Stanford Alpaca - 52K instruction-following examples
alpaca = load_dataset("tatsu-lab/alpaca", split="train[:1000]")  # Subset for demo

# Format for training
def format_instruction(example):
    if example["input"]:
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
```

## Memory Considerations for DGX Spark

### Model Memory Requirements (FP16/BF16)
| Model | Parameters | Memory | Notes |
|-------|------------|--------|-------|
| Mamba-130M | 130M | ~260MB | Good for learning |
| Mamba-1.4B | 1.4B | ~2.8GB | Fast experiments |
| Mamba-2.8B | 2.8B | ~5.6GB | Production quality |
| DeepSeekMoE-16B | 16B total | ~32GB | 2.5B active |
| Mixtral 8x7B | 45B total | ~90GB | 12.9B active |
| Jamba-52B | 52B total | ~104GB | Hybrid architecture |

### Context Length Memory (Approximate)
| Context | Transformer KV Cache | Mamba State |
|---------|---------------------|-------------|
| 4K tokens | ~512MB | ~8MB |
| 16K tokens | ~2GB | ~8MB |
| 64K tokens | ~8GB | ~8MB |
| 100K tokens | ~12GB+ | ~8MB |

**Key Insight**: Mamba's memory stays constant regardless of context length!

## Caching

All downloaded datasets are cached in:
```bash
~/.cache/huggingface/datasets/
```

To clear cache if needed:
```bash
rm -rf ~/.cache/huggingface/datasets/wikitext*
rm -rf ~/.cache/huggingface/datasets/pg19*
```

## Synthetic Data Generation

For benchmarking, we generate synthetic sequences:

```python
import torch

def generate_benchmark_data(tokenizer, lengths=[1024, 4096, 16384, 32768]):
    """Generate sequences of various lengths for benchmarking."""
    data = {}
    for length in lengths:
        # Generate random token IDs within vocabulary
        input_ids = torch.randint(
            100, tokenizer.vocab_size - 100,
            (1, length),
            device="cuda"
        )
        data[length] = input_ids
    return data
```
