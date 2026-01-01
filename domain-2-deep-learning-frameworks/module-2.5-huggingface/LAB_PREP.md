# Module 2.5: Hugging Face Ecosystem - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 2.5.1 Hub Exploration | 5 min | 2 hr | ~2 hr |
| 2.5.2 Pipeline Showcase | 10 min | 2 hr | ~2 hr |
| 2.5.3 Dataset Processing | 10 min | 2 hr | ~2 hr |
| 2.5.4 Trainer Fine-tuning | 15 min | 2 hr | ~2.5 hr |
| 2.5.5 LoRA Introduction | 10 min | 2 hr | ~2 hr |
| 2.5.6 Model Upload | 10 min | 2 hr | ~2 hr |

**Total**: ~10-12 hours

---

## Required Downloads

### Models (Auto-download)

```python
# Common models used in labs
from transformers import AutoModel

# For pipelines (Lab 2.5.2) - ~500 MB each
# gpt2, bert-base-uncased, facebook/bart-large-cnn, etc.

# For fine-tuning (Lab 2.5.4) - ~440 MB
AutoModel.from_pretrained("bert-base-uncased")

# For LoRA (Lab 2.5.5) - varies by model
```

### Datasets (Auto-download)

```python
from datasets import load_dataset

# IMDb (Labs 2.5.3, 2.5.4) - ~85 MB
load_dataset("imdb")

# Other common datasets
load_dataset("squad")       # Question answering
load_dataset("glue", "sst2")  # Sentiment
```

### Packages

```bash
pip install transformers datasets peft accelerate evaluate scikit-learn
```

---

## Environment Setup

### 1. Start NGC Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 2. Install Packages

```bash
pip install transformers datasets peft accelerate evaluate scikit-learn --quiet
```

### 3. Verify Setup

```python
import transformers
import datasets
import peft

print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"peft: {peft.__version__}")

# Quick test
from transformers import pipeline
classifier = pipeline("sentiment-analysis", device=0)
print(classifier("HuggingFace makes AI accessible!"))
```

### 4. HuggingFace Login (for Lab 2.5.6)

```python
from huggingface_hub import login

# Get token from https://huggingface.co/settings/tokens
login()  # Interactive prompt
# Or: login(token="your_token_here")
```

---

## Pre-Lab Checklist

### Lab 2.5.1: Hub Exploration
- [ ] Internet connection for browsing HF Hub
- [ ] transformers library installed

### Lab 2.5.2: Pipeline Showcase
- [ ] GPU accessible (`device=0`)
- [ ] ~5 GB free for various pipeline models

### Lab 2.5.3: Dataset Processing
- [ ] datasets library installed
- [ ] IMDb dataset accessible

### Lab 2.5.4: Trainer Fine-tuning
- [ ] BERT accessible
- [ ] IMDb downloaded
- [ ] evaluate library for metrics

### Lab 2.5.5: LoRA Introduction
- [ ] peft library installed
- [ ] ~16 GB GPU memory for comparison

### Lab 2.5.6: Model Upload
- [ ] HuggingFace account created
- [ ] API token generated and logged in
- [ ] Model from Lab 2.5.4 or 2.5.5

---

## Resource Requirements

| Lab | GPU Memory | Notes |
|-----|------------|-------|
| 2.5.1 | ~2 GB | Model loading tests |
| 2.5.2 | ~4 GB | Multiple pipelines |
| 2.5.3 | ~2 GB | CPU-focused |
| 2.5.4 | ~8 GB | BERT fine-tuning |
| 2.5.5 | ~16 GB | Full vs LoRA comparison |
| 2.5.6 | ~4 GB | Model upload |

---

## Quick Start Commands

```bash
# Inside NGC container
cd /workspace/domain-2-deep-learning-frameworks/module-2.5-huggingface

# Install dependencies
pip install transformers datasets peft accelerate evaluate scikit-learn --quiet

# Pre-download common models and datasets
python -c "
from transformers import AutoModel, AutoTokenizer, pipeline
from datasets import load_dataset

print('Downloading BERT...')
AutoModel.from_pretrained('bert-base-uncased')
AutoTokenizer.from_pretrained('bert-base-uncased')

print('Downloading GPT-2...')
pipeline('text-generation', model='gpt2')

print('Downloading IMDb...')
load_dataset('imdb')

print('All downloads complete!')
"
```

---

## Expected File Structure

```
/workspace/
├── domain-2-deep-learning-frameworks/
│   └── module-2.5-huggingface/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── STUDY_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       ├── LAB_PREP.md
│       ├── TROUBLESHOOTING.md
│       ├── labs/
│       │   ├── lab-2.5.1-hub-exploration.ipynb
│       │   ├── lab-2.5.2-pipeline-showcase.ipynb
│       │   ├── lab-2.5.3-dataset-processing.ipynb
│       │   ├── lab-2.5.4-trainer-finetuning.ipynb
│       │   ├── lab-2.5.5-lora-introduction.ipynb
│       │   └── lab-2.5.6-model-upload.ipynb
│       ├── scripts/
│       ├── configs/
│       └── solutions/
```

---

## DGX Spark Advantages

| Feature | Benefit |
|---------|---------|
| 128GB Memory | Load larger models (Llama-8B, Mistral-7B) for fine-tuning |
| BF16 Native | Faster training with `bf16=True` in TrainingArguments |
| Fast Memory | Larger batch sizes for faster training |
| Multi-model | Load base + adapter without swapping |
