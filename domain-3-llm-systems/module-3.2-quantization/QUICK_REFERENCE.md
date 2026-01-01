# Module 3.2: Quantization & Optimization - Quick Reference

## 🚀 Essential Commands

### NGC Container Setup
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Verify Blackwell (Required for NVFP4)
```python
import torch
cc = torch.cuda.get_device_capability()
print(f"Compute Capability: {cc[0]}.{cc[1]}")
if cc[0] >= 10:
    print("✅ Blackwell detected! NVFP4 tensor cores available.")
else:
    print("⚠️ Pre-Blackwell GPU. NVFP4 will run in emulation.")
```

### Clear Memory Before Large Models
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## 📊 Key Values to Remember

### Data Type Comparison
| Format | Bits | Memory (70B) | Range | Best For |
|--------|------|--------------|-------|----------|
| FP32 | 32 | 280 GB | ±3.4×10³⁸ | Training (legacy) |
| FP16 | 16 | 140 GB | ±65,504 | Inference |
| BF16 | 16 | 140 GB | ±3.4×10³⁸ | Training/Inference (DGX Spark default) |
| FP8 E4M3 | 8 | 70 GB | ±448 | Inference |
| FP8 E5M2 | 8 | 70 GB | ±57,344 | Training |
| INT8 | 8 | 70 GB | -128 to 127 | Weight quantization |
| INT4 | 4 | 35 GB | -8 to 7 | Weight quantization |
| NVFP4 | 4 | 35 GB | Float range | Blackwell inference |

### Quantization Methods
| Method | Type | Hardware | Best For | DGX Spark |
|--------|------|----------|----------|-----------|
| NVFP4 | PTQ | Blackwell only | Production inference | ⭐ Native |
| FP8 | PTQ/QAT | Blackwell+ | Training + inference | ✅ Native |
| GPTQ | PTQ | Any GPU | Fast quantization | ✅ Works |
| AWQ | PTQ | Any GPU | Best quality | ✅ Works |
| GGUF | PTQ | CPU/GPU | Ollama/llama.cpp | ✅ Works |
| bitsandbytes | PTQ | Any GPU | Quick 4/8-bit | ✅ Works |

### DGX Spark Performance

Verify these benchmarks in your Ollama Web UI for consistent test conditions:

| Model | Precision | Memory | Prefill (tok/s) | Decode (tok/s) |
|-------|-----------|--------|-----------------|----------------|
| Llama 3.1 8B | NVFP4 | ~4 GB | ~10,000 | ~39 |
| Llama 3.1 8B | FP16 | ~16 GB | ~3,000 | ~20 |
| Llama 3.1 70B | NVFP4 | ~35 GB | ~2,500 | ~15 |
| GPT-OSS 20B | MXFP4 | ~10 GB | ~4,500 | ~59 |

## 🔧 Common Patterns

### Pattern: bitsandbytes 4-bit (Quick)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Pattern: NVFP4 with TensorRT Model Optimizer
```python
from modelopt.torch.quantization import quantize

# Load calibration data
calib_data = load_calibration_dataset()

# Apply NVFP4 quantization
model = quantize(
    model,
    quant_cfg="nvfp4",
    calibration_dataloader=calib_data
)

# Expected: 3.5x memory reduction, <1% accuracy loss
```

### Pattern: FP8 Training
```python
from transformer_engine.pytorch import fp8_autocast

# Enable FP8 computation
with fp8_autocast():
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

# 2x compute throughput on Tensor Cores
```

### Pattern: GPTQ Quantization
```python
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=gptq_config,
    device_map="auto"
)
```

### Pattern: AWQ Quantization
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Quantize
model.quantize(
    tokenizer,
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }
)

# Save
model.save_quantized("llama-8b-awq")
```

### Pattern: GGUF Conversion for Ollama
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert HuggingFace model to GGUF
python convert_hf_to_gguf.py ../merged_model --outfile model-f16.gguf

# Quantize to Q4_K_M
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Import to Ollama
ollama create mymodel -f Modelfile
```

### Pattern: Quality Benchmark
```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model=model,
    tasks=["wikitext"],
    batch_size=8
)

perplexity = results['results']['wikitext']['word_perplexity']
print(f"Perplexity: {perplexity:.2f}")

# Target: <0.5 increase from FP16 baseline
```

## 📝 GGUF Quantization Levels

| Format | Bits | Size (7B) | Quality | Use Case |
|--------|------|-----------|---------|----------|
| Q2_K | 2 | ~2.7 GB | Poor | Extreme compression |
| Q3_K_S | 3 | ~3.0 GB | Low | Memory-critical |
| Q4_0 | 4 | ~3.5 GB | Medium | Basic 4-bit |
| Q4_K_M | 4 | ~4.0 GB | Good | **Recommended default** |
| Q5_K_M | 5 | ~4.5 GB | Better | Quality priority |
| Q6_K | 6 | ~5.5 GB | High | Near-original |
| Q8_0 | 8 | ~7.0 GB | Very High | Maximum quality |

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| NVFP4 "not available" error | Verify Blackwell GPU: compute capability ≥ 10.0 |
| FP8 produces NaN | Reduce learning rate, check gradient scaling |
| GPTQ slow quantization | Reduce calibration samples, ensure GPU is used |
| AWQ OOM during quantization | Use smaller calibration dataset |
| GGUF conversion fails | Check llama.cpp version matches model architecture |
| Poor quality after quantization | Use more calibration data, try larger group size |
| bitsandbytes CUDA error | Reinstall: `pip install bitsandbytes --no-cache-dir` |

## 🔗 Quick Links
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [NVIDIA FP8 Format](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [TensorRT Model Optimizer](https://developer.nvidia.com/tensorrt)
- [llama.cpp GGUF](https://github.com/ggerganov/llama.cpp)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
