# Module 2.5: Hugging Face Ecosystem - Quick Reference

## Essential Commands

### Loading Models

```python
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained(
    "model-name",
    torch_dtype=torch.bfloat16,  # DGX Spark optimized
    device_map="auto"
)

# Task-specific loading
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

### Pipeline API

```python
from transformers import pipeline

# Common pipelines
generator = pipeline("text-generation", model="gpt2", device=0)
classifier = pipeline("sentiment-analysis", device=0)
qa = pipeline("question-answering", device=0)
ner = pipeline("ner", aggregation_strategy="simple", device=0)
summarizer = pipeline("summarization", device=0)
translator = pipeline("translation_en_to_fr", device=0)
```

---

## Key Patterns

### Pattern: Dataset Loading and Processing

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")  # or local: load_dataset("csv", data_files="file.csv")

# Tokenize with map
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,       # Process in batches (faster)
    num_proc=4,         # Parallel processing
    remove_columns=["text"]  # Remove original text column
)

# Split if needed
dataset = dataset.train_test_split(test_size=0.1)
```

### Pattern: Trainer Fine-tuning

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=True,  # DGX Spark: use BF16
    dataloader_num_workers=4,
    logging_steps=100,
    report_to="none",  # or "wandb", "tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./final_model")
```

### Pattern: Custom Metrics

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }
```

### Pattern: LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(
    r=16,                       # Rank
    lora_alpha=32,              # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM  # or SEQ_CLS, SEQ_2_SEQ_LM
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: X | all params: Y | trainable%: ~0.1%

# Training is the same as before
trainer = Trainer(model=model, args=args, ...)
trainer.train()

# Save adapter only
model.save_pretrained("./lora_adapter")

# Load adapter later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base-model")
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

### Pattern: Model Upload to Hub

```python
from huggingface_hub import HfApi, login

# Login (get token from huggingface.co/settings/tokens)
login()

# Push model
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")

# Or use Trainer
trainer.push_to_hub()
```

### Pattern: Model Card

```python
from huggingface_hub import ModelCard

card = ModelCard.load("your-username/model-name")
card.data.license = "apache-2.0"
card.data.tags = ["text-classification", "bert"]
card.text = """
# Model Name

## Description
Fine-tuned BERT for sentiment analysis.

## Training Data
IMDb movie reviews dataset.

## Usage
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="your-username/model-name")
```

## Performance
- Accuracy: 92%
- F1: 0.91
"""
card.push_to_hub("your-username/model-name")
```

---

## Key Values

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| Learning rate (fine-tuning) | 2e-5 to 5e-5 | BERT recommendation |
| Batch size | 8-32 | Increase on DGX Spark |
| LoRA rank (r) | 8-64 | 16 is common starting point |
| LoRA alpha | 2× rank | Typical: alpha = 32 for r = 16 |
| Warmup steps | 500-1000 | 10% of total steps |
| Weight decay | 0.01 | Standard for AdamW |

---

## Common Target Modules for LoRA

| Model Family | Target Modules |
|--------------|----------------|
| Llama/Mistral | `["q_proj", "v_proj", "k_proj", "o_proj"]` |
| BERT | `["query", "value"]` |
| GPT-2 | `["c_attn"]` |
| T5 | `["q", "v"]` |
| Falcon | `["query_key_value"]` |

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Tokenizer/model mismatch | Load both from same checkpoint |
| Forgetting `device_map="auto"` | Model stays on CPU |
| Missing `bf16=True` in TrainingArguments | Slower training |
| Wrong `num_labels` | Check dataset classes |
| Not setting `pad_token` | `tokenizer.pad_token = tokenizer.eos_token` |
| LoRA target_modules wrong | Check model architecture with `print(model)` |

---

## Quick Links

- [Hugging Face Hub](https://huggingface.co/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [HF Course](https://huggingface.co/learn/nlp-course)
