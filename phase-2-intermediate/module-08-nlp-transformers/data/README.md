# Data Directory

This directory is for storing datasets used in Module 8: NLP & Transformers.

## Datasets Used in This Module

### Task 8.4: Tokenization Lab
- **Training corpus**: Generated inline in the notebook
- **No external data required**

### Task 8.5: BERT Fine-tuning
- **Dataset**: IMDB Movie Reviews
- **Source**: Hugging Face `datasets` library
- **Download**: Automatic via `load_dataset("imdb")`
- **Size**: ~130MB compressed
- **Format**: 25K training + 25K test reviews with binary sentiment labels

### Task 8.6: GPT Text Generation
- **No training data required** - uses pre-trained GPT-2

## Downloading Datasets

All datasets are automatically downloaded by the notebooks. To pre-download:

```python
from datasets import load_dataset

# IMDB for sentiment classification
imdb = load_dataset("imdb")
print(f"IMDB: {len(imdb['train'])} train, {len(imdb['test'])} test")

# Optional: Emotion dataset for multi-class classification
emotion = load_dataset("emotion")
print(f"Emotion: {len(emotion['train'])} train")
```

## Cache Location

Hugging Face datasets are cached at:
- Linux/Mac: `~/.cache/huggingface/datasets/`
- Windows: `C:\Users\<username>\.cache\huggingface\datasets\`

## DGX Spark Setup

For optimal performance on DGX Spark, use the NGC container with all required flags:

```bash
docker run --gpus all -it --rm \
    --ipc=host \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/workspace:/workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Important flags:**
- `--gpus all`: Enable GPU access for CUDA operations
- `--ipc=host`: **Required** for DataLoader multiprocessing (num_workers > 0)
- `-v .../huggingface`: Share model/dataset cache between runs

## DGX Spark Advantages

With 128GB unified memory, you can:
- Load entire datasets into memory
- Use larger batch sizes (64+ instead of 16-32)
- Skip memory-saving tricks like lazy loading
- Fine-tune larger models without gradient checkpointing

## Custom Data

If you want to use custom data for fine-tuning:

1. **CSV Format**:
```csv
text,label
"This is a positive review",1
"This is a negative review",0
```

2. **JSON Format**:
```json
[
  {"text": "This is a positive review", "label": 1},
  {"text": "This is a negative review", "label": 0}
]
```

3. **Loading Custom Data**:
```python
from datasets import load_dataset

# From CSV
dataset = load_dataset("csv", data_files="data/custom.csv")

# From JSON
dataset = load_dataset("json", data_files="data/custom.json")
```

## Pre-trained Models

The following pre-trained models are used:
- `bert-base-uncased` (~440MB)
- `gpt2` (~500MB)

These are cached at:
- Linux/Mac: `~/.cache/huggingface/hub/`
- Windows: `C:\Users\<username>\.cache\huggingface\hub\`

To share cache in Docker:
```bash
docker run -v $HOME/.cache/huggingface:/root/.cache/huggingface ...
```
