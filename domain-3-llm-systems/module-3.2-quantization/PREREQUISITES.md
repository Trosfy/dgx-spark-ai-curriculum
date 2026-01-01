# Module 3.2: Quantization & Optimization - Prerequisites Check

## 🎯 Purpose
This module assumes familiarity with model loading and fine-tuning. Use this self-check to ensure you're ready.

## ⏱️ Estimated Time
- **If all prerequisites met**: Jump straight to [QUICKSTART.md](./QUICKSTART.md)
- **If 1-2 gaps**: ~2-3 hours of review
- **If multiple gaps**: Complete Module 3.1 first

---

## Required Skills

### 1. Model Loading: HuggingFace Transformers

**Can you do this?**
```python
# Without looking anything up, load a model with:
# 1. A specific dtype (bfloat16)
# 2. Automatic device mapping
# 3. Check memory usage
```

<details>
<summary>✅ Check your answer</summary>

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Check memory
memory_gb = torch.cuda.memory_allocated() / 1e9
print(f"Memory used: {memory_gb:.2f} GB")
```

**Key points**:
- `torch.bfloat16` is preferred on DGX Spark (Blackwell native)
- `device_map="auto"` handles GPU placement automatically
- Use `memory_allocated()` to check actual usage

</details>

**Not ready?** Review: [Module 3.1: LLM Fine-Tuning](../module-3.1-llm-finetuning/)

---

### 2. BitsAndBytes: 4-bit Quantization

**Can you do this?**
```python
# Without looking anything up, create a BitsAndBytesConfig for:
# - 4-bit quantization
# - NF4 quantization type
# - bfloat16 compute dtype
```

<details>
<summary>✅ Check your answer</summary>

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Optional: extra memory savings
)

model = AutoModelForCausalLM.from_pretrained(
    "model-name",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Key points**:
- `nf4` is optimized for normally-distributed weights
- `bfloat16` compute dtype works best on Blackwell
- Double quantization adds ~5% more compression

</details>

**Not ready?** Review: [Module 3.1 Lab 3.1.5](../module-3.1-llm-finetuning/labs/lab-3.1.5-70b-qlora-finetuning.ipynb)

---

### 3. Data Types: Understanding Precision

**Do you know these terms?**
| Term | Your Definition |
|------|-----------------|
| FP32 | [Write yours here] |
| FP16 | [Write yours here] |
| BF16 | [Write yours here] |
| INT8 | [Write yours here] |

<details>
<summary>✅ Check definitions</summary>

| Term | Definition |
|------|------------|
| FP32 | 32-bit floating point. Full precision, 4 bytes per weight. Standard but memory-heavy. |
| FP16 | 16-bit floating point. Half precision, smaller range than FP32. Can have overflow issues with very large values. |
| BF16 | 16-bit "brain float". Same range as FP32, less precision. Preferred on Blackwell/DGX Spark. |
| INT8 | 8-bit integer. 256 possible values. Requires scaling to represent float-like values. |

</details>

**Not ready?** Review: [ELI5.md](./ELI5.md) in this module

---

### 4. Memory Management: GPU Memory

**Can you answer this?**
> How much memory does a 70B parameter model need in FP16 vs INT4?

<details>
<summary>✅ Check your answer</summary>

**FP16 (2 bytes per parameter):**
- 70B × 2 bytes = 140 GB
- Plus overhead = ~150 GB (won't fit in 128GB)

**INT4 (0.5 bytes per parameter):**
- 70B × 0.5 bytes = 35 GB
- Plus overhead = ~45 GB (fits easily in 128GB)

**Memory formula:**
```
Memory (GB) = Parameters × Bytes per parameter + Overhead

FP32: 4 bytes/param
FP16/BF16: 2 bytes/param
INT8: 1 byte/param
INT4: 0.5 bytes/param
```

</details>

**Not ready?** Review: [Module 1.3: CUDA Python](../../domain-1-platform-foundations/module-1.3-cuda-python/)

---

### 5. Inference: Running a Model

**Can you do this?**
```python
# Without looking anything up, generate text from a loaded model:
# - Tokenize input
# - Generate with max_new_tokens
# - Decode output
```

<details>
<summary>✅ Check your answer</summary>

```python
# Assuming model and tokenizer are loaded
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize
inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

# Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

**Key points**:
- Always move inputs to same device as model
- `max_new_tokens` controls output length
- `torch.no_grad()` saves memory during inference

</details>

**Not ready?** Review: [Module 2.5: HuggingFace Ecosystem](../../domain-2-deep-learning-frameworks/module-2.5-huggingface/)

---

### 6. Hardware: DGX Spark Specifics

**Can you answer this?**
> What makes DGX Spark special for quantization compared to other GPUs?

<details>
<summary>✅ Check your answer</summary>

**DGX Spark (Blackwell GB10 Superchip) Advantages:**

1. **NVFP4 Support**: Native FP4 tensor cores (only on Blackwell)
   - 3.5× memory reduction vs FP16
   - Hardware-accelerated inference

2. **128GB Unified Memory**: Huge memory pool
   - Run 200B models with NVFP4
   - No need to shard across GPUs

3. **FP8 Native Support**: E4M3/E5M2 tensor cores
   - 2× compute efficiency vs FP16
   - Works for training and inference

4. **Compute Capability 10.0**: Latest NVIDIA architecture
   - All quantization methods supported
   - Best performance for all formats

</details>

---

## Optional But Helpful

These aren't required but will accelerate your learning:

### NumPy Broadcasting and Scaling
**Why it helps**: Understanding how scale factors work in quantization.
**Quick primer**: When quantizing, we scale values to fit in a smaller range, then scale back during computation.

### Basic Statistics (Mean, Variance)
**Why it helps**: Calibration data statistics determine optimal scaling.
**Quick primer**: NF4 quantization assumes weights are normally distributed (bell curve).

---

## Ready?

- [ ] I can load models with different dtypes and check memory
- [ ] I understand BitsAndBytes 4-bit configuration
- [ ] I know the memory requirements for different precisions
- [ ] I can run inference and generate text
- [ ] I understand DGX Spark's unique capabilities

**All boxes checked?** → Start with [QUICKSTART.md](./QUICKSTART.md)!

**Some gaps?** → Review Module 3.1 first—quantization builds on fine-tuning knowledge.
