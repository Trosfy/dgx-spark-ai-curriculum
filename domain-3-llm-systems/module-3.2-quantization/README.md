# Module 3.2: Model Quantization & Optimization

**Domain:** 3 - LLM Systems
**Duration:** Weeks 19-20 (12-15 hours)
**Prerequisites:** Module 3.1 (LLM Fine-tuning)
**Priority:** P0 Critical (NVFP4, FP8 Expansion)

---

## Overview

Quantization is the key to running large models efficiently. This module covers all major quantization techniques, with special focus on Blackwell's exclusive NVFP4 and native FP8 capabilities—a unique advantage of the DGX Spark platform.

Deep dive into NVFP4 (3.5× memory reduction), FP8 training, and TensorRT-LLM engine building. This is where DGX Spark's Blackwell architecture truly shines!

---

## Environment Setup

### NGC Container (Required for DGX Spark)

Launch the PyTorch NGC container with proper flags for DGX Spark:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Important flags:**
- `--gpus all`: Required for GPU access
- `--ipc=host`: Required for DataLoader with multiple workers
- Volume mounts: Persist your work and HuggingFace cache

### Verifying Blackwell Hardware

To confirm you're running on Blackwell (required for FP4):

```python
import torch
cc = torch.cuda.get_device_capability()
print(f"Compute Capability: {cc[0]}.{cc[1]}")

if cc[0] >= 10:
    print("✅ Blackwell detected! FP4 tensor cores available.")
else:
    print("⚠️  Non-Blackwell GPU. FP4 will run in emulation mode.")
```

### Pre-clearing Memory for Large Models

Before loading models >10GB, clear the buffer cache:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

This ensures maximum available unified memory on DGX Spark's 128GB system.

### Library Requirements by Notebook

| Notebook | Required Libraries |
|----------|-------------------|
| 01-quantization-overview | transformers, torch, bitsandbytes |
| 02-gptq-quantization | auto-gptq, transformers |
| 03-awq-quantization | autoawq, transformers |
| 04-gguf-conversion | llama.cpp (built from source), sentencepiece, gguf |
| 05-fp4-deep-dive | nvidia-modelopt, transformers |
| 06-quality-benchmark-suite | lm-eval, transformers, bitsandbytes |

**Note:** Most libraries are pre-installed in the NGC PyTorch container.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Apply various quantization techniques (GPTQ, AWQ, GGUF, FP4/FP8)
- ✅ Optimize models for inference using TensorRT
- ✅ Evaluate quantization impact on model quality
- ✅ Select optimal quantization strategy for deployment

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.2.1 | Explain NVFP4 micro-block scaling and FP8 E4M3 format | Understand |
| 3.2.2 | Quantize models using NVFP4 with TensorRT Model Optimizer | Apply |
| 3.2.3 | Apply GPTQ, AWQ, and GGUF quantization | Apply |
| 3.2.4 | Measure and compare quality degradation | Evaluate |

---

## Topics

### 3.2.1 Quantization Fundamentals

- **Data Types Overview**
  - FP32 → FP16 → BF16 → INT8 → INT4 → FP8 → FP4
  - Precision vs memory tradeoffs

- **Quantization Approaches**
  - Post-training quantization (PTQ): Fast, no retraining
  - Quantization-aware training (QAT): Better quality, more effort
  - Calibration datasets and their importance

### 3.2.2 Quantization Methods

| Method | Bits | Best For | DGX Spark Support |
|--------|------|----------|-------------------|
| GPTQ | 4-bit | GPU inference | ✅ |
| AWQ | 4-bit | Activation-aware | ✅ |
| GGUF | 2-8 bit | llama.cpp | ✅ |
| FP8 | 8-bit | Training + inference | ✅ Native |
| **NVFP4** | 4-bit | Blackwell exclusive | ⭐ **Only on DGX Spark** |

### 3.2.3 Blackwell-Specific Quantization [P0 Expansion]

- **NVFP4 (NVIDIA FP4)**
  - Dual-level scaling for accuracy
  - Micro-block scaling within groups
  - 3.5× memory reduction vs FP16
  - <0.1% accuracy loss on MMLU
  - ~10,000+ tok/s prefill on 8B models

- **FP8 (E4M3/E5M2)**
  - E4M3: 4-bit exponent, 3-bit mantissa (inference)
  - E5M2: 5-bit exponent, 2-bit mantissa (training, larger range)
  - Native Blackwell Tensor Core support
  - 2× compute efficiency vs FP16

- **MXFP4 (Open Compute Project)**
  - Open standard for FP4
  - Compatible with multiple vendors
  - Similar quality to NVFP4

### 3.2.4 TensorRT-LLM Integration

- Building optimized TensorRT engines
- Weight-only vs full quantization
- INT8 KV cache for memory savings
- Optimal configurations for DGX Spark

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 3.2.1 | Data Type Exploration | 1.5h | Visualize FP32→FP16→FP8→FP4 precision loss |
| 3.2.2 | NVFP4 Quantization ⭐ | 3h | NVFP4 on 70B model (Blackwell showcase!) |
| 3.2.3 | FP8 Training and Inference | 2h | Train with FP8, compare with FP16 |
| 3.2.4 | GPTQ Quantization | 2h | Quantize 7B with group sizes 32/64/128 |
| 3.2.5 | AWQ Quantization | 1.5h | Compare AWQ vs GPTQ |
| 3.2.6 | GGUF Conversion | 2h | Convert to GGUF, test with llama.cpp |
| 3.2.7 | Quality Benchmark Suite | 2h | Perplexity + MMLU across all variants |
| 3.2.8 | TensorRT-LLM Engine | 2h | Build TRT engine with NVFP4, benchmark |

---

## Guidance

### NVFP4 Quantization (Blackwell Exclusive!)

```python
from modelopt.torch.quantization import quantize

# Calibration data
calib_data = load_calibration_dataset()

# Apply NVFP4 quantization
model = quantize(
    model,
    quant_cfg="nvfp4",
    calibration_dataloader=calib_data
)

# Expected: 3.5x memory reduction, <1% accuracy loss
```

### Performance Expectations on DGX Spark

| Model | Precision | Prefill (tok/s) | Decode (tok/s) |
|-------|-----------|-----------------|----------------|
| Llama 3.1 8B | NVFP4 | ~10,000 | ~39 |
| Llama 3.1 8B | FP16 | ~3,000 | ~20 |
| GPT-OSS 20B | MXFP4 | ~4,500 | ~59 |

### Quality Metrics

```python
# Perplexity (lower is better)
# Target: <0.5 increase from FP16 baseline

from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model=model,
    tasks=["wikitext"],
    batch_size=8
)
print(f"Perplexity: {results['results']['wikitext']['word_perplexity']}")
```

### FP8 Training

```python
import torch

# Enable FP8 training with transformer engine
from transformer_engine.pytorch import fp8_autocast

with fp8_autocast():
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

# FP8 benefits:
# - 2x compute throughput on Tensor Cores
# - Reduced memory for activations
# - Native Blackwell support
```

### DGX Spark Quantization Capacity

| Quantization | 8B Model | 70B Model | 200B Model |
|--------------|----------|-----------|------------|
| FP16 | ✅ 16GB | ✅ 140GB* | ❌ OOM |
| FP8 | ✅ 8GB | ✅ 70GB | ❌ OOM |
| NVFP4 | ✅ 4GB | ✅ 35GB | ✅ ~100GB |
| GPTQ/AWQ | ✅ 4GB | ✅ 35GB | ✅ ~100GB |

*With gradient checkpointing or inference-only

---

## Milestone Checklist

- [ ] Data type precision visualization complete
- [ ] **NVFP4 quantization of 70B model** ⭐ (DGX Spark showcase!)
- [ ] FP8 training and inference demonstrated
- [ ] GPTQ with multiple group sizes
- [ ] AWQ vs GPTQ comparison
- [ ] GGUF conversion and llama.cpp testing
- [ ] Quality benchmark suite with all variants
- [ ] TensorRT-LLM engine built and benchmarked

---

## Common Issues

| Issue | Solution |
|-------|----------|
| NVFP4 not available | Verify Blackwell GPU (compute capability ≥ 10.0) |
| FP8 NaN loss | Reduce learning rate, check gradient scaling |
| GPTQ slow quantization | Reduce calibration samples or use GPU |
| GGUF conversion fails | Check llama.cpp version compatibility |
| TensorRT build fails | Verify TensorRT-LLM version matches container |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save your quantized models
3. ➡️ Proceed to [Module 3.3: Deployment & Inference](../module-3.3-deployment/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 3.1: LLM Fine-Tuning](../module-3.1-llm-finetuning/) | **Module 3.2: Quantization** | [Module 3.3: Deployment](../module-3.3-deployment/) |

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | See memory savings in 5 minutes |
| [ELI5.md](./ELI5.md) | Jargon-free explanations of FP4, FP8, GPTQ, AWQ |
| [PREREQUISITES.md](./PREREQUISITES.md) | Self-check before starting |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning objectives and lab roadmap |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Data types, commands, and code patterns |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and downloads |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | NVFP4, GPTQ, AWQ, GGUF errors and FAQ |

---

## Resources

- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [NVIDIA FP8 Format](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [TensorRT Model Optimizer](https://developer.nvidia.com/tensorrt)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
