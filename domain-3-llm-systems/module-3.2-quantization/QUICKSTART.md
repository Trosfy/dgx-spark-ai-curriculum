# Module 3.2: Quantization & Optimization - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Load a quantized model and see the memory savings compared to full precision.

## ✅ Before You Start
- [ ] DGX Spark NGC container running
- [ ] At least 8GB GPU memory free

## 🚀 Let's Go!

### Step 1: Start the Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Step 2: Check Baseline Memory
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model in full precision
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

fp16_memory = torch.cuda.memory_allocated() / 1e9
print(f"FP16 Memory: {fp16_memory:.2f} GB")

# Clean up
del model_fp16
torch.cuda.empty_cache()
```

### Step 3: Load with 4-bit Quantization
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

int4_memory = torch.cuda.memory_allocated() / 1e9
print(f"4-bit Memory: {int4_memory:.2f} GB")
```

### Step 4: Compare the Savings
```python
savings = (1 - int4_memory / fp16_memory) * 100
print(f"\n📊 Memory Comparison:")
print(f"FP16:  {fp16_memory:.2f} GB")
print(f"4-bit: {int4_memory:.2f} GB")
print(f"Savings: {savings:.1f}%")
```

**Expected output**:
```
📊 Memory Comparison:
FP16:  2.12 GB
4-bit: 0.68 GB
Savings: 67.9%
```

### Step 5: Verify It Still Works
```python
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
outputs = model_4bit.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Expected output**:
```
The capital of France is Paris.
```

## 🎉 You Did It!

You just quantized a model and achieved **~68% memory savings** while maintaining functionality! The same technique scales dramatically:

| Model | FP16 | 4-bit | Savings |
|-------|------|-------|---------|
| 8B | 16 GB | 4 GB | 75% |
| 70B | 140 GB | 35 GB | 75% |
| 200B | OOM | ~100 GB | Fits! |

On DGX Spark, NVFP4 (Blackwell exclusive) achieves even better results with hardware acceleration.

## ▶️ Next Steps
1. **Explore data types**: See [Lab 3.2.1](./labs/lab-3.2.1-data-type-exploration.ipynb)
2. **Try NVFP4**: See [Lab 3.2.2](./labs/lab-3.2.2-nvfp4-quantization.ipynb) (DGX Spark showcase!)
3. **Full setup**: Start with [LAB_PREP.md](./LAB_PREP.md)
