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

## 📞 Still Stuck?

1. **Check compute capability** - Many errors are GPU architecture mismatches
2. **Verify package versions** - Quantization libraries update frequently
3. **Try smaller model first** - Confirm process works before scaling up
4. **Check the lab notebook comments** - Often contain specific hints
5. **Search with library name** - e.g., "auto-gptq CUDA error"
