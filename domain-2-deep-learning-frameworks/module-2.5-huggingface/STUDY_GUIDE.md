# Module 2.5: Hugging Face Ecosystem - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Navigate the HuggingFace Hub** to find and evaluate models
2. **Use Pipeline API** for quick inference across multiple tasks
3. **Process datasets** efficiently with the datasets library
4. **Fine-tune models** using the Trainer API
5. **Apply PEFT/LoRA** for parameter-efficient fine-tuning
6. **Upload models** to the Hub with proper documentation

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-2.5.1-hub-exploration.ipynb | Hub navigation | ~2 hr | 10 models documented |
| 2 | lab-2.5.2-pipeline-showcase.ipynb | Pipeline API | ~2 hr | 5 pipeline demos |
| 3 | lab-2.5.3-dataset-processing.ipynb | Datasets library | ~2 hr | Large dataset pipeline |
| 4 | lab-2.5.4-trainer-finetuning.ipynb | Trainer API | ~2 hr | Fine-tuned classifier |
| 5 | lab-2.5.5-lora-introduction.ipynb | PEFT/LoRA | ~2 hr | Memory comparison |
| 6 | lab-2.5.6-model-upload.ipynb | Hub upload | ~2 hr | Model card + upload |

**Total time**: ~10-12 hours

---

## Core Concepts

### Hugging Face Hub
**What**: Central repository for models, datasets, and spaces
**Why it matters**: Access 500K+ pre-trained models; share your work with the community
**First appears in**: Lab 2.5.1

### Pipeline API
**What**: High-level API for inference with one line of code
**Why it matters**: Quick prototyping; handles tokenization, inference, and post-processing automatically
**First appears in**: Lab 2.5.2

### Datasets Library
**What**: Efficient dataset loading and processing with Arrow/memory-mapping
**Why it matters**: Handle datasets larger than memory; fast preprocessing with map()
**First appears in**: Lab 2.5.3

### Trainer API
**What**: High-level training loop with logging, checkpointing, evaluation
**Why it matters**: Production-ready training without writing boilerplate
**First appears in**: Lab 2.5.4

### PEFT (Parameter-Efficient Fine-Tuning)
**What**: Techniques like LoRA that fine-tune small adapters instead of full model
**Why it matters**: 100x fewer trainable parameters; enables fine-tuning large models on smaller GPUs
**First appears in**: Lab 2.5.5

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 2.4              ►   Module 2.5              ►  Module 2.6
Efficient Arch              HuggingFace                 Diffusion
(Mamba, MoE)                (ecosystem)                 (images)
```

**Builds on**:
- **PyTorch fundamentals** from Module 2.1 (HF is built on PyTorch)
- **Transformer understanding** from Module 2.3 (know what's under the hood)
- **Model loading patterns** from all previous modules

**Prepares for**:
- **Module 2.6** uses HF Diffusers library for image generation
- **Module 3.1** extends LoRA for LLM fine-tuning
- **All of Domain 3** uses HuggingFace heavily

---

## Recommended Approach

### Standard Path (10-12 hours)
1. **Start with Lab 2.5.1** - Understand how to find and evaluate models
2. **Work through Lab 2.5.2** - Pipelines are the fastest way to prototype
3. **Complete Lab 2.5.3** - Data processing is crucial for real projects
4. **Master Lab 2.5.4** - Trainer is the standard for fine-tuning
5. **Apply Lab 2.5.5** - LoRA will be essential in Domain 3
6. **Finish Lab 2.5.6** - Learn to share your work properly

### Quick Path (6-8 hours, if familiar with HF)
1. Skim Lab 2.5.1 - Focus on model selection criteria
2. Focus on Lab 2.5.4 - Trainer is the key skill
3. Complete Lab 2.5.5 - LoRA is critical for Domain 3
4. Skip Lab 2.5.6 unless you plan to share models

### Deep-Dive Path (15+ hours)
1. Explore 20+ models in Lab 2.5.1
2. Build a multi-task system using pipelines
3. Process a custom dataset from scratch
4. Implement custom Trainer callbacks
5. Compare LoRA, QLoRA, AdaLoRA techniques

---

## Key APIs

### AutoModel Pattern
```python
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Works for any model on the Hub
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained("model-name")

# Task-specific variants
AutoModelForCausalLM        # Text generation
AutoModelForSequenceClassification  # Classification
AutoModelForQuestionAnswering  # QA
```

### Datasets Pattern
```python
from datasets import load_dataset

dataset = load_dataset("dataset-name")
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.1)
```

### Trainer Pattern
```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    bf16=True,  # Use on DGX Spark
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## DGX Spark Advantages

| Task | Consumer GPU | DGX Spark |
|------|--------------|-----------|
| Load Llama-3-8B | 16GB, tight | Easy, room for batching |
| Fine-tune 7B model | Usually requires QLoRA | LoRA at full precision |
| Process large datasets | Memory limited | 128GB for fast processing |
| Multi-model inference | One at a time | Several models concurrently |

---

## Before You Start

- See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute pipeline demo

---

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "Which model to choose?" | Check model card, downloads, likes, benchmarks |
| "Tokenizer mismatch" | Always load tokenizer from same checkpoint as model |
| "Trainer not logging" | Check `logging_steps` in TrainingArguments |
| "LoRA not training" | Verify `target_modules` match model architecture |
| "Dataset too slow" | Use `batched=True` with `num_proc` in map() |

---

## Success Metrics

You've mastered this module when you can:

- [ ] Find and evaluate models on the Hub for any task
- [ ] Use pipelines for quick inference
- [ ] Process datasets efficiently with map()
- [ ] Fine-tune a model with Trainer
- [ ] Apply LoRA to reduce training memory
- [ ] Upload a model with proper documentation
