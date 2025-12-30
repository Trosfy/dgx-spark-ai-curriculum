# Module 2.4: Hugging Face Ecosystem

**Domain:** 2 - Deep Learning Frameworks  
**Duration:** Weeks 13-14 (10-12 hours)  
**Prerequisites:** Module 8 (Transformers)

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
| 2.4.1 | Load and use pre-trained models from Hugging Face Hub | Apply |
| 2.4.2 | Preprocess datasets using datasets library transformations | Apply |
| 2.4.3 | Configure and use the Trainer API for fine-tuning | Apply |
| 2.4.4 | Evaluate models using the evaluate library | Analyze |

---

## Topics

### 2.4.1 Hugging Face Hub
- Model cards and documentation
- Model discovery and selection
- Uploading models and datasets

### 2.4.2 Transformers Library
- Auto classes (AutoModel, AutoTokenizer)
- Pipeline API for quick inference
- Model-specific classes

### 2.4.3 Datasets Library
- Loading datasets
- Map and filter operations
- Streaming for large datasets

### 2.4.4 Training with HF
- Trainer and TrainingArguments
- Custom training loops with Accelerate
- Evaluation metrics

### 2.4.5 PEFT Library
- Parameter-efficient fine-tuning intro
- LoRA configuration
- Merging adapters

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 2.4.1 | Hub Exploration | 2h | Document 10 models, test 3 locally |
| 2.4.2 | Pipeline Showcase | 2h | Demo 5 pipelines (text-gen, sentiment, NER, QA, summarization) |
| 2.4.3 | Dataset Processing | 2h | Process large dataset with map(), create splits |
| 2.4.4 | Trainer Fine-tuning | 2h | Text classification with custom metrics |
| 2.4.5 | LoRA Introduction | 2h | Compare LoRA vs full fine-tuning memory |
| 2.4.6 | Model Upload | 2h | Fine-tune, create model card, upload to Hub |

---

## Guidance

### Loading Models with DGX Spark Advantage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# With 128GB unified memory, you can load large models!
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Automatic device placement
)
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

## Milestone Checklist

- [ ] 10 models documented from HF Hub
- [ ] 5 pipeline demonstrations complete
- [ ] Large dataset processing pipeline working
- [ ] Trainer fine-tuning with custom metrics
- [ ] LoRA vs full fine-tuning comparison
- [ ] Model uploaded to HF Hub with card

---

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
