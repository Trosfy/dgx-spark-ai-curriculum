# Module 3.1: LLM Fine-Tuning - Prerequisites Check

## 🎯 Purpose
This module assumes specific prior knowledge. Use this self-check to ensure you're ready, or identify gaps to fill first.

## ⏱️ Estimated Time
- **If all prerequisites met**: Jump straight to [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~2-4 hours of review
- **If multiple gaps**: Complete prerequisite modules first

---

## Required Skills

### 1. PyTorch: Model Loading and Inference

**Can you do this?**
```python
# Without looking anything up, write code to:
# 1. Load a pre-trained model from HuggingFace
# 2. Run inference on a single input
# 3. Check if CUDA is available and use it
```

<details>
<summary>✅ Check your answer</summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # or model.to(device)
)

# Run inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**Key points**:
- Use `device_map="auto"` for automatic GPU placement
- Use `torch.bfloat16` for DGX Spark (native Blackwell support)
- Always move inputs to the same device as the model

</details>

**Not ready?** Review: [Module 2.1: PyTorch Mastery](../../domain-2-deep-learning-frameworks/module-2.1-pytorch/)

---

### 2. Transformers: Understanding Architecture Components

**Can you answer this?**
> What are the main components of a transformer decoder, and what do q_proj, k_proj, v_proj, and o_proj refer to?

<details>
<summary>✅ Check your answer</summary>

**Transformer Decoder Components:**
1. **Self-Attention**: Allows each token to attend to all previous tokens
2. **Feed-Forward Network (MLP)**: Processes each position independently
3. **Layer Normalization**: Stabilizes training
4. **Residual Connections**: Enable gradient flow through deep networks

**Projection Matrices:**
- `q_proj` (Query projection): Transforms input to query vectors
- `k_proj` (Key projection): Transforms input to key vectors
- `v_proj` (Value projection): Transforms input to value vectors
- `o_proj` (Output projection): Transforms attention output back to model dimension

These are the layers typically targeted by LoRA because they're involved in attention computation.

</details>

**Not ready?** Review: [Module 2.3: NLP & Transformers](../../domain-2-deep-learning-frameworks/module-2.3-nlp-transformers/)

---

### 3. HuggingFace: Dataset Loading and Tokenization

**Can you do this?**
```python
# Without looking anything up, write code to:
# 1. Load a dataset from HuggingFace Hub
# 2. Tokenize it with padding and truncation
# 3. Create a data collator for training
```

<details>
<summary>✅ Check your answer</summary>

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize function
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

# Apply tokenization
tokenized = dataset.map(tokenize, batched=True)

# Create data collator
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)
```

**Key points**:
- Set `pad_token = eos_token` for models that don't have a pad token
- Use `mlm=False` for causal language models (GPT-style)
- `max_length` should match your training context window

</details>

**Not ready?** Review: [Module 2.5: HuggingFace Ecosystem](../../domain-2-deep-learning-frameworks/module-2.5-huggingface/)

---

### 4. Training Basics: Loss, Optimizer, Training Loop

**Can you answer this?**
> What is cross-entropy loss, and why do we use it for language model training?

<details>
<summary>✅ Check your answer</summary>

**Cross-Entropy Loss:**
Measures the difference between predicted token probabilities and actual next tokens.

For language modeling:
- Model predicts probability distribution over vocabulary for next token
- Cross-entropy measures how "surprised" the model is by the actual next token
- Lower = better predictions

**Why for LLMs:**
- Language modeling is classification: predict the next token from vocabulary
- Cross-entropy is the standard loss for multi-class classification
- It naturally handles the probability distribution over tokens

**Formula (simplified):**
```
Loss = -log(probability of correct token)
```

If model predicts 90% probability for correct token: Loss = -log(0.9) = 0.11 (low)
If model predicts 10% probability for correct token: Loss = -log(0.1) = 2.30 (high)

</details>

**Not ready?** Review: [Module 1.5: Neural Network Fundamentals](../../domain-1-platform-foundations/module-1.5-neural-networks/)

---

### 5. Memory Management: GPU Memory Concepts

**Do you know these terms?**
| Term | Your Definition |
|------|-----------------|
| VRAM / GPU Memory | [Write yours here] |
| Gradient Checkpointing | [Write yours here] |
| Mixed Precision (FP16/BF16) | [Write yours here] |
| Memory Fragmentation | [Write yours here] |

<details>
<summary>✅ Check definitions</summary>

| Term | Definition |
|------|------------|
| VRAM / GPU Memory | Memory on the GPU for storing model weights, activations, and gradients. DGX Spark has 128GB unified memory. |
| Gradient Checkpointing | Trade compute for memory: don't store all activations, recompute during backward pass. Reduces memory ~50% but slows training. |
| Mixed Precision (FP16/BF16) | Use 16-bit instead of 32-bit floats. Halves memory, often faster. BF16 preferred on Blackwell for better range. |
| Memory Fragmentation | Memory becomes divided into small unusable chunks. Solution: `torch.cuda.empty_cache()` or restart. |

</details>

**Not ready?** Review: [Module 1.3: CUDA Python Programming](../../domain-1-platform-foundations/module-1.3-cuda-python/)

---

### 6. PEFT Basics: Parameter-Efficient Fine-Tuning Concepts

**Can you answer this?**
> What is the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT)?

<details>
<summary>✅ Check your answer</summary>

**Full Fine-Tuning:**
- Updates ALL model parameters
- Requires storing gradients for all weights
- Memory: ~4x model size (weights + gradients + optimizer states)
- For 70B model: Would need ~280GB+ just for training

**Parameter-Efficient Fine-Tuning (PEFT):**
- Adds small trainable modules (adapters, LoRA, etc.)
- Freezes original weights
- Only stores gradients for new parameters
- Memory: Model size + tiny adapter overhead
- For 70B model: ~45-55GB with QLoRA

**Why PEFT:**
1. Fits larger models in memory
2. Prevents catastrophic forgetting
3. Can store multiple task-specific adapters
4. Often matches full fine-tuning quality

</details>

**Not ready?** Review: [Module 2.5: HuggingFace Ecosystem - PEFT Section](../../domain-2-deep-learning-frameworks/module-2.5-huggingface/)

---

## Optional But Helpful

These aren't required but will accelerate your learning:

### Linear Algebra: Matrix Decomposition
**Why it helps**: LoRA is based on low-rank matrix decomposition. Understanding this makes the "rank" parameter intuitive.
**Quick primer**: Any matrix W can be approximated as W ≈ BA where B and A are smaller matrices. The rank controls the approximation quality.

### Reinforcement Learning Basics
**Why it helps**: Preference optimization (DPO, RLHF) has roots in RL concepts.
**Quick primer**: You don't need full RL knowledge—just understand that we're optimizing for "preferred" outputs using comparison data.

---

## Ready?

- [ ] I can load and run inference with HuggingFace models
- [ ] I understand transformer architecture components
- [ ] I can tokenize datasets and create data collators
- [ ] I understand training loss and optimization basics
- [ ] I know how to manage GPU memory
- [ ] I understand the difference between full and parameter-efficient fine-tuning

**All boxes checked?** → Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** → No shame! Review the linked materials first. This module builds heavily on previous knowledge.
