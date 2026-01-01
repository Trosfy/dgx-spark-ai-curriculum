# Module 3.2: Quantization & Optimization - Troubleshooting Guide

## 🔍 Quick Diagnostic

**Before diving into specific errors, try these:**
1. Verify GPU: `python -c "import torch; print(torch.cuda.get_device_capability())"`
2. Check memory: `nvidia-smi` or `torch.cuda.memory_summary()`
3. Clear cache: `torch.cuda.empty_cache(); gc.collect()`
4. Verify compute capability for NVFP4: Need CC ≥ 10.0 (Blackwell)

---

## 🚨 Error Categories

### NVFP4 Errors

#### Error: `NVFP4 not available` or `FP4 not supported`
**Symptoms**:
```
RuntimeError: FP4 quantization requires Blackwell architecture (SM100+)
```

**Cause**: GPU doesn't have native FP4 tensor cores.

**Solution**:
```python
# First, check your GPU
import torch
cc = torch.cuda.get_device_capability()
print(f"Compute Capability: {cc[0]}.{cc[1]}")

# Blackwell = 10.x (supports NVFP4 natively)
# Hopper = 9.0 (FP8 only)
# Ampere = 8.x (INT8/FP16 only)

if cc[0] < 10:
    print("NVFP4 not supported. Use these alternatives:")
    print("- bitsandbytes 4-bit (NF4)")
    print("- GPTQ")
    print("- AWQ")
```

---

#### Error: `modelopt import error`
**Symptoms**:
```
ModuleNotFoundError: No module named 'modelopt'
```

**Solution**:
```bash
# Install TensorRT Model Optimizer
pip install nvidia-modelopt

# Or from NGC container, it may be at a different path
pip install nvidia-modelopt --extra-index-url https://pypi.nvidia.com
```

---

### FP8 Errors

#### Error: `NaN loss during FP8 training`
**Symptoms**:
```
Loss: nan after X steps
```

**Causes**:
1. Learning rate too high for FP8 range
2. Gradient overflow
3. Missing loss scaling

**Solutions**:
```python
# Solution 1: Reduce learning rate
learning_rate = 1e-5  # Lower than FP16 training

# Solution 2: Enable gradient scaling
from transformer_engine.pytorch import fp8_autocast

with fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()):
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss

# Solution 3: Use E5M2 for training (larger range)
from transformer_engine.common import recipe
fp8_recipe = recipe.DelayedScaling(
    fp8_format=recipe.Format.E5M2,  # More range for training
    amax_compute_algo="max"
)
```

---

### GPTQ Errors

#### Error: `GPTQ quantization is very slow`
**Cause**: Using CPU instead of GPU, or too many calibration samples.

**Solution**:
```python
from transformers import GPTQConfig

# Reduce calibration samples
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=True,
    # Reduce these for faster quantization:
    num_samples=128,  # Default is 128, can go lower for testing
)

# Ensure GPU is used
import torch
assert torch.cuda.is_available(), "GPU required for fast GPTQ"
```

---

#### Error: `GPTQ CUDA error` or `illegal memory access`
**Cause**: GPU memory fragmentation or version mismatch.

**Solution**:
```python
# Clear memory completely
import torch, gc
torch.cuda.empty_cache()
gc.collect()

# Reduce batch size during quantization
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    batch_size=1,  # Reduce from default
)

# If still failing, restart kernel and try again
```

---

### AWQ Errors

#### Error: `AWQ OOM during quantization`
**Cause**: AWQ needs more memory during the quantization process.

**Solution**:
```python
# Use smaller calibration dataset
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_path)

# Smaller calibration
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    },
    calib_data=calibration_data[:64],  # Reduce samples
)
```

---

#### Error: `AWQ version mismatch`
**Symptoms**:
```
KeyError: 'awq' or ValueError: Unknown quantization type
```

**Solution**:
```bash
# Update to latest AWQ
pip install autoawq>=0.2.0 --upgrade

# Ensure compatible transformers version
pip install transformers>=4.40.0
```

---

### GGUF Errors

#### Error: `llama.cpp convert script fails`
**Symptoms**:
```
KeyError: 'model.layers.0.self_attn.q_proj.weight'
```

**Cause**: llama.cpp version doesn't match model architecture.

**Solution**:
```bash
# Update llama.cpp to latest
cd llama.cpp
git pull origin master
make clean
make -j$(nproc) GGML_CUDA=1

# Check supported architectures
python convert_hf_to_gguf.py --help
```

---

#### Error: `Ollama won't load GGUF model`
**Symptoms**:
```
Error: model not found or invalid format
```

**Solutions**:
```bash
# 1. Create proper Modelfile
cat > Modelfile << 'EOF'
FROM ./model-q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
TEMPLATE """{{ .Prompt }}"""
EOF

# 2. Import with Modelfile
ollama create mymodel -f Modelfile

# 3. Verify it loaded
ollama list
```

---

#### Error: `GGUF quantization produces bad quality`
**Cause**: Wrong quantization level for model size.

**Solution**:
```bash
# For 7B models, Q4_K_M is usually best balance
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# For larger models (70B+), can use more aggressive
./llama-quantize model-f16.gguf model-q3_k_m.gguf Q3_K_M

# For best quality, use Q5 or Q6
./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
```

---

### TensorRT-LLM Errors

#### Error: `TensorRT build fails`
**Symptoms**:
```
[TensorRT] ERROR: ...
```

**Solutions**:
```python
# 1. Verify TensorRT version compatibility
import tensorrt
print(tensorrt.__version__)

# 2. Use matching container
# Check NGC for latest TensorRT-LLM container

# 3. Reduce complexity for initial build
python -m tensorrt_llm.commands.build \
    --model_dir ./model \
    --output_dir ./trt_engine \
    --dtype bfloat16 \
    --max_batch_size 1  # Start small
```

---

#### Error: `TensorRT build slow (>2 hours)`
**Cause**: Large model with many optimizations.

**Solution**:
```bash
# Expected times:
# 8B model: 15-30 minutes
# 70B model: 45-90 minutes

# Speed up by reducing optimization level:
python -m tensorrt_llm.commands.build \
    --model_dir ./model \
    --output_dir ./trt_engine \
    --builder_opt_level 2  # Lower = faster build, slower inference
```

---

### Quality Issues

#### Problem: High perplexity after quantization
**Symptoms**: Perplexity increases by >1.0 after quantization.

**Solutions**:
```python
# 1. Use more calibration data
calibration_samples = 256  # Increase from 128

# 2. Try larger group size (less aggressive quantization)
gptq_config = GPTQConfig(
    bits=4,
    group_size=64,  # Smaller = better quality, more memory
)

# 3. Use AWQ instead of GPTQ (often better quality)
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained(...)

# 4. For critical applications, use 8-bit instead of 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Instead of load_in_4bit
)
```

---

#### Problem: Model outputs gibberish after quantization
**Cause**: Severe quantization damage, wrong config, or tokenizer mismatch.

**Solutions**:
```python
# 1. Verify tokenizer matches model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, ...)

# 2. Check quantization applied correctly
print(model)  # Look for quantized layer types

# 3. Test with simple prompt
inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))

# 4. If still broken, try less aggressive quantization
```

---

## 🔄 Reset Procedures

### Full Environment Reset
```bash
# 1. Exit container
exit

# 2. Clear quantized model cache
rm -rf ~/.cache/huggingface/hub/*-gptq
rm -rf ~/.cache/huggingface/hub/*-awq

# 3. Clear buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 4. Restart container
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Memory-Only Reset
```python
import torch
import gc

# Delete all models
for name in list(globals()):
    if isinstance(globals()[name], torch.nn.Module):
        del globals()[name]

# Force cleanup
gc.collect()
torch.cuda.empty_cache()

# Verify
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Do I need a Blackwell GPU for this module?**

**A**: Not entirely. Most quantization methods (GPTQ, AWQ, GGUF, bitsandbytes) work on any modern GPU. However, NVFP4 is **Blackwell-exclusive** (compute capability 10.0+). DGX Spark's unique advantage is native NVFP4 and FP8 tensor core support.

| Method | Ampere (8.x) | Hopper (9.x) | Blackwell (10.x) |
|--------|--------------|--------------|------------------|
| bitsandbytes 4-bit | ✅ | ✅ | ✅ |
| GPTQ | ✅ | ✅ | ✅ |
| AWQ | ✅ | ✅ | ✅ |
| GGUF | ✅ | ✅ | ✅ |
| FP8 | ⚠️ Emulated | ✅ Native | ✅ Native |
| NVFP4 | ❌ | ❌ | ✅ Native |

---

**Q: Which quantization method should I use?**

**A**:

| If you need... | Use... | Why |
|----------------|--------|-----|
| Quick 4-bit for fine-tuning | bitsandbytes | Integrated with HuggingFace |
| Best inference quality | AWQ | Activation-aware, minimal quality loss |
| Fastest quantization | GPTQ | Well-optimized process |
| Ollama/llama.cpp compatibility | GGUF | Universal format |
| Maximum DGX Spark performance | NVFP4 | Hardware-accelerated |
| Training in reduced precision | FP8 | Native Blackwell support |

**Default recommendation**: Start with bitsandbytes for development, AWQ for production inference, GGUF for Ollama deployment.

---

**Q: How much quality do I lose with 4-bit quantization?**

**A**: Less than you'd expect! With proper calibration:

| Quantization | Typical Perplexity Increase | MMLU Accuracy Loss |
|--------------|----------------------------|-------------------|
| FP16 → INT8 | ~0.05 | <0.5% |
| FP16 → INT4 (GPTQ) | ~0.2-0.3 | ~1-2% |
| FP16 → INT4 (AWQ) | ~0.1-0.2 | ~0.5-1% |
| FP16 → NVFP4 | ~0.05 | <0.1% |

NVFP4's dual-level scaling achieves near-FP16 quality at 4-bit size.

---

### Concepts

**Q: What's the difference between NF4 and INT4?**

**A**:

**INT4** (regular 4-bit integer):
- 16 evenly spaced values: -8, -7, -6, ... 0, ... 6, 7
- Works poorly for neural networks (weights aren't uniform)

**NF4** (NormalFloat4):
- 16 values optimized for normally-distributed data
- More values near 0 (where most weights are)
- Much better quality for the same 4 bits

```
INT4:  |-8    -4    0    4    8|  (uniform spacing)
NF4:   ||||||  |    |  |||||||   (clustered near zero)
```

This is why bitsandbytes uses NF4 by default.

---

**Q: What is calibration data and why does it matter?**

**A**: Calibration data is a small sample (128-512 examples) used during quantization to:

1. Determine activation ranges for each layer
2. Set optimal scale factors
3. Identify which weights are "important"

**Bad calibration** = poor quantization:
- If you calibrate on English but run on code → wrong scale factors
- If you use too few samples → high variance, unstable quality

**Best practice**: Use data similar to your actual use case.

---

**Q: What's the difference between PTQ and QAT?**

**A**:

**PTQ (Post-Training Quantization)**:
- Quantize after training is complete
- Fast: minutes to hours
- Slight quality loss
- Methods: GPTQ, AWQ, NVFP4

**QAT (Quantization-Aware Training)**:
- Train with quantization simulation
- Slow: full training run
- Better quality
- Methods: FP8 training, some advanced approaches

For most LLM use cases, **PTQ is sufficient** and much faster.

---

**Q: Why does GGUF have so many quantization levels (Q2, Q3, Q4, Q5, Q6, Q8)?**

**A**: GGUF is designed for flexibility across different hardware:

| Level | Use Case |
|-------|----------|
| Q2_K | Extreme memory constraints (not recommended for quality) |
| Q3_K | Very tight memory, acceptable for simple tasks |
| Q4_K_M | **Best balance** for most users |
| Q5_K_M | Quality priority, more memory |
| Q6_K | Near-original quality |
| Q8_0 | Maximum quality in GGUF format |

The "_K" variants use "k-quant" (smarter quantization) and "_M" is medium granularity.

---

### Common Issues

**Q: Why is my GPTQ quantization taking hours?**

**A**: Common causes:
1. **Using CPU instead of GPU** - Ensure CUDA is available
2. **Too many calibration samples** - 128 is usually enough
3. **Large model without streaming** - Use gradient checkpointing

```python
# Check GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True

# Reduce samples for faster testing
gptq_config = GPTQConfig(bits=4, dataset="c4", num_samples=64)
```

---

**Q: My quantized model gives wrong/gibberish outputs. Why?**

**A**: Check these in order:
1. **Tokenizer mismatch** - Load tokenizer from same path as model
2. **Incomplete quantization** - Process may have failed partway
3. **Too aggressive quantization** - Try Q5 instead of Q4, or AWQ instead of GPTQ
4. **Chat template issues** - Quantized model may need same template as original

---

**Q: How do I check if quantization worked correctly?**

**A**:
```python
# 1. Check model size
import os
original_size = os.path.getsize("original_model/model.safetensors")
quantized_size = os.path.getsize("quantized_model/model.safetensors")
print(f"Compression: {original_size/quantized_size:.1f}x")

# 2. Check memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 3. Run quick quality test
from lm_eval import evaluator
results = evaluator.simple_evaluate(model, tasks=["wikitext"])
print(f"Perplexity: {results['results']['wikitext']['word_perplexity']}")

# 4. Sanity check generation
output = model.generate(tokenizer("Hello, world!", return_tensors="pt").to("cuda"))
print(tokenizer.decode(output[0]))
```

---

### Beyond the Basics

**Q: Can I quantize any model?**

**A**: Most transformer-based models yes, but:
- **Well-supported**: Llama, Mistral, Qwen, Falcon, GPT-NeoX
- **Partial support**: Some architectures need custom handling
- **Check first**: Look for existing quantized versions on HuggingFace

---

**Q: Should I quantize for fine-tuning or just inference?**

**A**:
- **For fine-tuning**: Use bitsandbytes (QLoRA). The base model is quantized, adapters are FP16.
- **For inference**: Use GPTQ, AWQ, or NVFP4 for best performance.
- **Don't mix**: A QLoRA model and a GPTQ model are different. Choose based on your use case.

---

**Q: What's the best quantization for production deployment?**

**A**: Depends on your deployment target:

| Target | Recommended | Why |
|--------|-------------|-----|
| DGX Spark (TensorRT) | NVFP4 | Hardware-accelerated, best perf |
| vLLM/SGLang | AWQ or GPTQ | Good integration, fast serving |
| Ollama | GGUF Q4_K_M | Universal, easy deployment |
| Cloud API | AWQ | Good balance of quality and speed |

---

**Q: How do I combine quantization with speculative decoding?**

**A**: Yes, and it's powerful! In Module 3.3 you'll learn:
- Draft model: Small, FP16 (fast)
- Target model: Large, quantized (memory-efficient)
- Result: 2-3x speedup on already-quantized models

---

## 📞 Still Stuck?

1. **Check compute capability** - Many errors are GPU architecture mismatches
2. **Verify package versions** - Quantization libraries update frequently
3. **Try smaller model first** - Confirm process works before scaling up
4. **Check the lab notebook comments** - Often contain specific hints
5. **Search with library name** - e.g., "auto-gptq CUDA error"
