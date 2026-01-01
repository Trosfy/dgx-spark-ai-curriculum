# Module 2.5: Hugging Face Ecosystem - Prerequisites

## Required Knowledge

Before starting this module, ensure you can do the following:

### From Module 2.1 (PyTorch Fundamentals)

| Skill | Self-Check | If No |
|-------|------------|-------|
| Create and manipulate PyTorch tensors | Can you create a random tensor and move it to GPU? | Review Module 2.1 Lab 1 |
| Understand `torch.no_grad()` context | Do you know why it's used for inference? | Review Module 2.1 Lab 3 |
| Use DataLoader for batching | Can you create a DataLoader from a dataset? | Review Module 2.1 Lab 4 |
| Understand model.eval() vs model.train() | Do you know when to use each? | Review Module 2.1 Lab 5 |

<details>
<summary>Quick Self-Test: PyTorch Basics</summary>

```python
import torch

# Q1: Create a tensor on GPU with bfloat16
# Your answer:
tensor = torch.randn(3, 4, dtype=torch.bfloat16, device="cuda")

# Q2: What does this context manager do?
with torch.no_grad():
    output = model(input)
# Answer: Disables gradient computation for inference (saves memory)

# Q3: Before inference, you should call:
# Answer: model.eval()
```
</details>

---

### From Module 2.3 (NLP & Transformers)

| Skill | Self-Check | If No |
|-------|------------|-------|
| Understand tokenization | Do you know what input_ids and attention_mask represent? | Review Module 2.3 Lab 1 |
| Know transformer architecture basics | Can you explain what attention does at a high level? | Review Module 2.3 Lab 2 |
| Understand classification head concept | Do you know how a model outputs class probabilities? | Review Module 2.3 Lab 3 |

<details>
<summary>Quick Self-Test: NLP Concepts</summary>

```python
# Q1: What does a tokenizer return?
# Answer: input_ids (token indices), attention_mask (which tokens to attend to)

# Q2: Why do transformers need attention masks?
# Answer: To handle variable-length sequences (ignore padding tokens)

# Q3: What's the purpose of a classification head?
# Answer: Converts hidden states to class logits/probabilities
```
</details>

---

### From Module 2.4 (Efficient Architectures)

| Skill | Self-Check | If No |
|-------|------------|-------|
| Understand model parameter counting | Can you estimate memory from parameter count? | Review Module 2.4 |
| Know about model efficiency concepts | Do you understand why smaller models matter? | Review Module 2.4 |

<details>
<summary>Quick Self-Test: Efficiency Concepts</summary>

```python
# Q1: How much memory does a 7B parameter model need in FP16?
# Answer: ~14 GB (7B * 2 bytes)

# Q2: How much memory in INT4?
# Answer: ~3.5 GB (7B * 0.5 bytes)

# Q3: Why is distillation useful?
# Answer: Creates smaller, faster models with similar performance
```
</details>

---

## Technical Requirements

### Environment

- [ ] DGX Spark with NGC PyTorch container (25.11-py3)
- [ ] GPU accessible (`torch.cuda.is_available()` returns `True`)
- [ ] Internet connection (for Hub access)

### Python Packages

The NGC container includes most requirements. Verify with:

```python
import transformers
import datasets
import peft

print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"peft: {peft.__version__}")
```

If missing, install with:
```bash
pip install transformers datasets peft accelerate evaluate scikit-learn --quiet
```

---

## Time Commitment

| Your Background | Expected Time |
|-----------------|---------------|
| New to HuggingFace | 12-15 hours |
| Some HuggingFace experience | 8-10 hours |
| Experienced with HuggingFace | 6-8 hours (focus on LoRA) |

---

## Ready to Start?

If you answered "Yes" to most self-check questions:
- Proceed to [QUICKSTART.md](./QUICKSTART.md) for a 5-minute intro
- Then start with [Lab 2.5.1](./labs/lab-2.5.1-hub-exploration.ipynb)

If you answered "No" to several questions:
- Review the referenced modules first
- Focus on the hands-on labs, not just reading

---

## Module Dependencies

```
Module 2.1 (PyTorch)
       │
       ▼
Module 2.3 (NLP/Transformers)
       │
       ▼
Module 2.4 (Efficient Architectures)
       │
       ▼
Module 2.5 (HuggingFace) ◄── YOU ARE HERE
       │
       ▼
Module 3.1 (LLM Fine-Tuning) - Advanced LoRA: DoRA, NEFTune, SimPO, ORPO
```
