# Data Directory - Module 9: Hugging Face Ecosystem

## Overview

This module primarily uses datasets from the Hugging Face Hub, so most data is loaded programmatically rather than stored locally. This directory contains supplementary data files and documentation.

---

## Datasets Used in This Module

### From Hugging Face Hub

| Dataset | Task | Size | Used In |
|---------|------|------|---------|
| `imdb` | Sentiment Classification | 50K reviews | Notebooks 03, 04 |
| `squad` | Question Answering | 100K+ examples | Notebook 02 |
| `conll2003` | Named Entity Recognition | 22K sentences | Notebook 02 |
| `xsum` | Summarization | 227K articles | Notebook 02 |
| `ag_news` | Text Classification | 120K articles | Notebook 04 |
| `wikitext` | Language Modeling | ~100M tokens | Notebook 05 |

### Loading Datasets

```python
from datasets import load_dataset

# Basic loading
dataset = load_dataset("imdb")

# Load specific split
train = load_dataset("imdb", split="train")

# Load with streaming (for large datasets)
dataset = load_dataset("imdb", streaming=True)

# Load subset for quick experimentation
dataset = load_dataset("imdb", split="train[:1000]")
```

---

## Local Files

### sample_texts.json (if created during exercises)

Sample text data for pipeline demonstrations:
```json
{
    "texts": [
        "This movie was absolutely fantastic!",
        "I've never been so disappointed.",
        "The acting was mediocre but the plot was interesting."
    ]
}
```

### model_comparison.csv (created in Notebook 05)

Results from LoRA vs full fine-tuning comparison:
```csv
method,trainable_params,memory_gb,training_time_min,final_accuracy
full_finetuning,125000000,45.2,120,0.923
lora_r8,524288,12.1,35,0.918
lora_r16,1048576,12.4,38,0.921
lora_r32,2097152,12.8,42,0.922
```

---

## Caching

Hugging Face datasets are cached locally to avoid re-downloading:

**Default cache location:** `~/.cache/huggingface/datasets/`

### Managing Cache

```python
from datasets import load_dataset

# Check cache directory
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
print(f"Cache size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(cache_dir) for f in filenames) / 1e9:.2f} GB")

# Load with custom cache directory
dataset = load_dataset("imdb", cache_dir="/workspace/data/hf_cache")

# Clear specific dataset from cache
from datasets import disable_caching
disable_caching()  # Disable for current session
```

### DGX Spark Docker Volume

When using the NGC container, mount your cache:
```bash
docker run --gpus all -it --rm \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    ...
```

---

## Memory Considerations on DGX Spark

With 128GB unified memory, you can:

1. **Load entire datasets into memory**: No need for streaming for most datasets
2. **Use larger batch sizes**: Process more samples per batch during map operations
3. **Parallel processing**: Use more workers in `num_proc` parameter

```python
# Optimized for DGX Spark
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,  # Large batches
    num_proc=8,       # Parallel processing
    remove_columns=dataset.column_names  # Save memory
)
```

---

## Data Formats

### Hugging Face Dataset Format

Datasets are stored in Apache Arrow format for:
- Memory-mapped loading (fast, low memory)
- Zero-copy data access
- Column-oriented storage

### Converting Formats

```python
# To pandas
df = dataset.to_pandas()

# To PyTorch tensors
dataset.set_format("torch")

# To NumPy arrays
dataset.set_format("numpy")

# Back to Arrow
dataset.reset_format()
```

---

## Creating Custom Datasets

### From Local Files

```python
from datasets import Dataset, DatasetDict

# From dictionary
data = {"text": ["Hello", "World"], "label": [0, 1]}
dataset = Dataset.from_dict(data)

# From pandas
import pandas as pd
df = pd.read_csv("my_data.csv")
dataset = Dataset.from_pandas(df)

# From JSON
dataset = load_dataset("json", data_files="my_data.json")

# From CSV
dataset = load_dataset("csv", data_files="my_data.csv")
```

### Upload to Hub

```python
from huggingface_hub import HfApi

# Push dataset
dataset.push_to_hub("username/my-dataset")
```

---

## Troubleshooting

### Common Issues

1. **Download fails**: Check internet connection, try `download_mode="force_redownload"`
2. **Out of memory**: Use `streaming=True` or smaller splits
3. **Slow loading**: Ensure data is cached, use `num_proc` for parallel loading
4. **Format errors**: Check `dataset.features` for expected types

### Verify Dataset Integrity

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")

# Check structure
print(dataset)
print(dataset.features)
print(dataset[0])

# Verify no missing values
print(dataset.filter(lambda x: x["text"] is None))
```
