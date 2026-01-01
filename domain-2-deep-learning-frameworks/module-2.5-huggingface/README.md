# Module 2.5: Hugging Face Ecosystem

**Domain:** 2 - Deep Learning Frameworks
**Duration:** Week 14 (10-12 hours)
**Prerequisites:** Module 2.4 (Efficient Architectures)

---

## Overview

Hugging Face has become the de facto platform for sharing and using pre-trained models. This module covers the complete ecosystem: Hub, Transformers, Datasets, and the Trainer API. You'll also get your first hands-on experience with parameter-efficient fine-tuning.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Navigate and utilize the Hugging Face Hub effectively
- ✅ Use Transformers library for various NLP tasks
- ✅ Load and preprocess datasets with the Datasets library
- ✅ Apply the Trainer API for efficient model training

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.5.1 | Load and use pre-trained models from Hugging Face Hub | Apply |
| 2.5.2 | Preprocess datasets using datasets library transformations | Apply |
| 2.5.3 | Configure and use the Trainer API for fine-tuning | Apply |
| 2.5.4 | Use PEFT library for parameter-efficient fine-tuning | Apply |

---

## Topics

### 2.5.1 Hugging Face Hub
- Model cards and documentation
- Model discovery and selection
- Uploading models and datasets

### 2.5.2 Transformers Library
- Auto classes (AutoModel, AutoTokenizer)
- Pipeline API for quick inference
- Model-specific classes

### 2.5.3 Datasets Library
- Loading datasets
- Map and filter operations
- Streaming for large datasets

### 2.5.4 Training with HF
- Trainer and TrainingArguments
- Custom training loops with Accelerate
- Evaluation metrics

### 2.5.5 PEFT Library
- Parameter-efficient fine-tuning intro
- LoRA configuration
- Merging adapters

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 2.5.1 | Hub Exploration | 2h | Document 10 models, test 3 locally |
| 2.5.2 | Pipeline Showcase | 2h | Demo 5 pipelines (text-gen, sentiment, NER, QA, summarization) |
| 2.5.3 | Dataset Processing | 2h | Process large dataset with map(), create splits |
| 2.5.4 | Trainer Fine-tuning | 2h | Text classification with custom metrics |
| 2.5.5 | LoRA Introduction | 2h | Compare LoRA vs full fine-tuning memory |
| 2.5.6 | Model Upload | 2h | Fine-tune, create model card, upload to Hub |

---

## DGX Spark Environment Setup

Start your development environment with the NGC PyTorch container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**DGX Spark Key Specs:**
- NVIDIA Blackwell GB10 Superchip
- 128GB unified LPDDR5X memory
- 6,144 CUDA cores, 192 Tensor Cores (5th gen)
- Native BF16 support for optimal training performance

---

## Guidance

### Loading Models with DGX Spark Advantage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# DGX Spark's 128GB unified memory enables loading large models!
# BF16 inference supports up to 50-55B parameters
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",  # Fits easily in BF16
    torch_dtype=torch.bfloat16,  # Native Blackwell GB10 support
    device_map="auto"  # Automatic device placement
)

# For larger models (70B+), use quantization:
# - FP8 inference: up to 90-100B parameters
# - NVFP4 inference: up to ~200B parameters (Blackwell exclusive)
# See Module 3.2 for quantization techniques
```

### Datasets Library Patterns

```python
from datasets import load_dataset

# Load and process
dataset = load_dataset("imdb")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized = dataset.map(tokenize, batched=True, num_proc=4)
```

### Trainer API

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,  # Use on DGX Spark
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### LoRA with PEFT

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 0.1% of total
```

---

## Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Run your first HuggingFace pipeline in 5 minutes |
| [PREREQUISITES.md](./PREREQUISITES.md) | Skills self-check and required knowledge |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and module roadmap |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Commands, patterns, and code snippets |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and lab preparation |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors and solutions |

---

## Common Issues

| Issue | Cause | Quick Fix |
|-------|-------|-----------|
| `OSError: model-name is not a valid model identifier` | Typo in model name | Check exact name on [huggingface.co/models](https://huggingface.co/models) |
| `401 Client Error: Unauthorized` | Model requires auth | Run `huggingface-cli login` first |
| `RuntimeError: CUDA out of memory` | Model too large | Use `torch_dtype=torch.bfloat16` and/or smaller batch size |
| Tokenizer/model mismatch | Different checkpoints | Load both from same model ID |
| LoRA not training | Wrong target_modules | Check model architecture with `print(model)` |

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions.

---

## Milestone Checklist

- [ ] 10 models documented from HF Hub
- [ ] 5 pipeline demonstrations complete
- [ ] Large dataset processing pipeline working
- [ ] Trainer fine-tuning with custom metrics
- [ ] LoRA vs full fine-tuning comparison
- [ ] Model uploaded to HF Hub with card

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save reusable pipeline code to `scripts/`
3. ➡️ Proceed to [Module 2.6: Diffusion Models](../module-2.6-diffusion-models/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 2.4: Efficient Architectures](../module-2.4-efficient-architectures/) | **Module 2.5: Hugging Face Ecosystem** | [Module 2.6: Diffusion Models](../module-2.6-diffusion-models/) |

---

## Related Modules

This module provides foundations for advanced topics:

| Module | Connection |
|--------|------------|
| **Module 3.1: LLM Fine-Tuning** | Advanced LoRA techniques (DoRA, NEFTune, SimPO, ORPO) |
| **Module 3.2: Quantization & Optimization** | NVFP4/FP8 for 100B+ models, QLoRA |
| **Module 3.3: Deployment & Inference** | SGLang, vLLM for production serving |

---

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
