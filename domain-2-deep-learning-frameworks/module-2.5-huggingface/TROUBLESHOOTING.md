# Module 2.5: Hugging Face Ecosystem - Troubleshooting Guide

## Quick Diagnostic

1. Check transformers version: `pip show transformers`
2. Check HF cache: `ls ~/.cache/huggingface/hub/`
3. Verify GPU: `torch.cuda.is_available()`
4. Check HF login: `huggingface-cli whoami`

---

## Model Loading Errors

### Error: `OSError: model-name is not a local folder and is not a valid model identifier`

**Cause**: Model name typo or model doesn't exist.

**Solution**:
```python
# Check exact name on HuggingFace Hub
# Go to huggingface.co/models and search

# Common mistakes:
# "bert-base" → "bert-base-uncased"
# "gpt2-large" → "gpt2" (then check for "-large" variant)
# "llama-7b" → "Qwen/Qwen3-8B" (check organization)
```

---

### Error: `401 Client Error: Unauthorized`

**Cause**: Model requires authentication.

**Solution**:
```python
from huggingface_hub import login

# Login with your token
login()

# Or pass token directly
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-8B",
    token="your_token_here"  # Or use_auth_token=True after login
)
```

---

### Error: `ValueError: text input must be of type str`

**Cause**: Pipeline received wrong input type.

**Solution**:
```python
# Wrong: passing list where string expected
pipeline("sentiment-analysis")(["text1", "text2"])  # Some pipelines

# Check pipeline documentation
# Most accept both, but verify:
result = classifier("Single string")
results = classifier(["List", "of", "strings"])
```

---

## Dataset Errors

### Error: `FileNotFoundError` when loading dataset

**Cause**: Dataset doesn't exist or network issue.

**Solution**:
```python
from datasets import load_dataset

# Check dataset name on huggingface.co/datasets
# Some datasets have subsets:
load_dataset("glue", "sst2")  # Not just "glue"
load_dataset("wikitext", "wikitext-103-v1")

# For local files:
load_dataset("csv", data_files="path/to/file.csv")
load_dataset("json", data_files="path/to/file.json")
```

---

### Error: `pyarrow.lib.ArrowNotImplementedError`

**Cause**: Dataset has unsupported data type.

**Solution**:
```python
# Remove problematic columns before processing
dataset = dataset.remove_columns(["problematic_column"])

# Or convert to supported type in map function
def fix_types(example):
    example["column"] = str(example["column"])
    return example

dataset = dataset.map(fix_types)
```

---

### Error: Dataset processing too slow

**Solution**:
```python
# Use batched processing
dataset = dataset.map(
    tokenize_function,
    batched=True,           # Process in batches
    batch_size=1000,        # Batch size
    num_proc=4              # Parallel processes
)

# For very large datasets, use streaming
dataset = load_dataset("dataset-name", streaming=True)
```

---

## Trainer Errors

### Error: `IndexError: Invalid key: X is out of bounds`

**Cause**: `num_labels` doesn't match dataset.

**Solution**:
```python
# Check unique labels in your dataset
labels = set(dataset["train"]["label"])
print(f"Unique labels: {labels}")
num_labels = len(labels)

# Then create model with correct num_labels
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)
```

---

### Error: `RuntimeError: expected scalar type Long but found Float`

**Cause**: Labels are float instead of int.

**Solution**:
```python
# Convert labels to integers
def fix_labels(example):
    example["label"] = int(example["label"])
    return example

dataset = dataset.map(fix_labels)
```

---

### Error: Trainer doesn't log anything

**Solution**:
```python
training_args = TrainingArguments(
    logging_steps=10,           # Log every 10 steps
    logging_first_step=True,    # Log first step
    logging_dir="./logs",       # Log directory
    report_to="tensorboard",    # Or "wandb", "none"
)

# View logs
# tensorboard --logdir ./logs
```

---

### Error: `OutOfMemoryError` during training

**Solutions**:
```python
# Solution 1: Reduce batch size
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Reduce from 8
    gradient_accumulation_steps=2,   # Simulate larger batch
)

# Solution 2: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use BF16 (if not already)
training_args = TrainingArguments(bf16=True)

# Solution 4: Use LoRA instead of full fine-tuning
```

---

## PEFT/LoRA Errors

### Error: `ValueError: Target modules not found`

**Cause**: `target_modules` don't match model architecture.

**Solution**:
```python
# Find correct module names
print(model)  # See full architecture

# Or programmatically
for name, module in model.named_modules():
    if "proj" in name or "linear" in name:
        print(name)

# Common patterns:
# Llama: ["q_proj", "v_proj", "k_proj", "o_proj"]
# BERT: ["query", "value"]
# GPT-2: ["c_attn", "c_proj"]
```

---

### Error: LoRA not training (loss not decreasing)

**Solutions**:
```python
# Check trainable parameters
model.print_trainable_parameters()
# Should show small percentage (0.1-1%)

# Verify LoRA is applied
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Training: {name}")

# Try higher rank or more target modules
config = LoraConfig(
    r=32,  # Increase from 16
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Add more
)
```

---

### Error: Can't load PEFT model

**Solution**:
```python
from peft import PeftModel, PeftConfig

# Load config first to get base model name
config = PeftConfig.from_pretrained("./lora_adapter")
print(f"Base model: {config.base_model_name_or_path}")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16
)

# Apply adapter
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

---

## Hub Upload Errors

### Error: `403 Forbidden` when pushing

**Causes**:
1. Not logged in
2. Repository doesn't exist
3. No write permission

**Solutions**:
```python
# Solution 1: Login
from huggingface_hub import login
login()

# Solution 2: Create repo first
from huggingface_hub import create_repo
create_repo("your-username/model-name")

# Solution 3: Check permissions
# Go to huggingface.co/settings/tokens
# Ensure token has "write" permission
```

---

### Error: Model too large to push

**Solution**:
```python
# Use Git LFS for large files (automatic for HF)
# Or push in shards
model.push_to_hub(
    "your-username/model-name",
    max_shard_size="5GB"  # Split into smaller files
)
```

---

## Pipeline Errors

### Error: Pipeline returns unexpected format

**Solution**:
```python
# Check pipeline documentation for output format
result = pipeline(text)
print(type(result))  # Often list of dicts
print(result)        # See structure

# Common outputs:
# sentiment: [{"label": "POSITIVE", "score": 0.99}]
# ner: [{"entity": "PER", "word": "John", "score": 0.98}]
# qa: {"answer": "...", "score": 0.95, "start": 0, "end": 10}
```

---

## Reset Procedures

### Clear HuggingFace Cache

```bash
# Remove all cached models (use with caution!)
rm -rf ~/.cache/huggingface/hub/

# Remove specific model
rm -rf ~/.cache/huggingface/hub/models--bert-base-uncased
```

### Memory Reset

```python
import gc
import torch

del model
del trainer
gc.collect()
torch.cuda.empty_cache()
```

---

## Still Stuck?

1. **Check HuggingFace model card** - Has usage examples
2. **Check HuggingFace forums** - Community solutions
3. **Print everything** - `print(model)`, `print(dataset)`, `print(tokenizer)`
4. **Check solution notebooks** - in `solutions/` folder
