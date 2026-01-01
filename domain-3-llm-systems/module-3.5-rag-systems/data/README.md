# Data Files for Module 3.5: RAG Systems & Vector Databases

This directory contains sample documents and datasets for RAG experiments.

## Recommended Datasets

| Dataset | Source | Size | Use Case |
|---------|--------|------|----------|
| SQuAD | HuggingFace | ~100K Q&A | Reading comprehension |
| Natural Questions | Google | ~300K Q&A | Open-domain QA |
| MS MARCO | Microsoft | ~1M passages | Passage retrieval |
| Multilingual Wikipedia | HuggingFace | Varies | Multilingual RAG |

## Sample Documents for Labs

The `sample_documents/` folder contains pre-created documents for all labs:

```
sample_documents/
├── dgx_spark_technical_guide.md      # DGX Spark hardware and software guide
├── transformer_architecture_explained.md  # Transformer architecture deep dive
├── lora_finetuning_guide.md          # LoRA and QLoRA fine-tuning
├── quantization_methods.md           # Quantization methods comparison
├── rag_architecture_patterns.md      # RAG design patterns and best practices
└── vector_database_comparison.md     # Vector DB comparison (ChromaDB, FAISS, Qdrant)
```

### Document Contents Summary

| Document | Topics Covered | Approx. Size |
|----------|---------------|--------------|
| dgx_spark_technical_guide.md | Hardware specs, memory, model capacity, software stack | ~5KB |
| transformer_architecture_explained.md | Attention, positional encoding, encoder/decoder | ~6KB |
| lora_finetuning_guide.md | LoRA, QLoRA, target modules, advanced techniques | ~6KB |
| quantization_methods.md | GPTQ, AWQ, GGUF, NVFP4, FP8, bitsandbytes | ~7KB |
| rag_architecture_patterns.md | Chunking, embeddings, vector DBs, evaluation | ~7KB |
| vector_database_comparison.md | ChromaDB, FAISS, Qdrant comparison | ~6KB |

## Loading Sample Data

### Create Sample Documents

```python
# Generate sample technical documents for testing
sample_docs = [
    {
        "content": """
# PyTorch Tensor Operations

PyTorch tensors are multi-dimensional arrays similar to NumPy arrays
but with GPU acceleration support.

## Creating Tensors
```python
import torch
x = torch.tensor([1, 2, 3])
y = torch.zeros(3, 3)
z = torch.randn(3, 3)
```

## GPU Operations
Move tensors to GPU with `.cuda()` or `.to('cuda')`.
        """,
        "metadata": {"source": "pytorch_tutorial.md", "type": "tutorial"}
    },
    # Add more sample documents...
]
```

### Using MS MARCO for Benchmarking

```python
from datasets import load_dataset

# Load MS MARCO for retrieval benchmarks
msmarco = load_dataset("ms_marco", "v2.1", split="train[:1000]")

# Each sample has: query, passages (positive and negative)
print(msmarco[0])
```

### Using SQuAD for Q&A

```python
from datasets import load_dataset

squad = load_dataset("squad", split="validation[:100]")

# Create RAG evaluation set
eval_set = []
for item in squad:
    eval_set.append({
        "question": item["question"],
        "context": item["context"],
        "answer": item["answers"]["text"][0]
    })
```

## Embedding Model Recommendations

| Model | Dimensions | Quality | Speed |
|-------|-----------|---------|-------|
| bge-large-en-v1.5 | 1024 | Best | Medium |
| bge-base-en-v1.5 | 768 | Great | Fast |
| qwen3-embedding:8b | 768 | Great | Fast (Ollama) |
| e5-large-v2 | 1024 | Great | Medium |

## Vector Database Storage

Estimated storage requirements:

| Documents | Embeddings (1024d) | ChromaDB | FAISS |
|-----------|-------------------|----------|-------|
| 10K | ~40MB | ~50MB | ~45MB |
| 100K | ~400MB | ~500MB | ~450MB |
| 1M | ~4GB | ~5GB | ~4.5GB |

All fit easily on DGX Spark's storage.

## Creating Evaluation Sets

For RAGAS evaluation, create ground-truth datasets:

```python
# evaluation_set.json format
[
    {
        "question": "What is the capital of France?",
        "ground_truth": "Paris is the capital of France.",
        "relevant_docs": ["france_geography.pdf:chunk_42"]
    },
    ...
]
```

Save evaluation sets in this directory for reproducible benchmarks.
