# Module 11: Model Quantization & Optimization

**Phase:** 3 - Advanced  
**Duration:** Weeks 18-19 (10-12 hours)  
**Prerequisites:** Module 10 (LLM Fine-tuning)

---

## Overview

Quantization is the key to running large models efficiently. This module covers all major quantization techniques, with special focus on Blackwell's exclusive FP4 capabilities—a unique advantage of the DGX Spark platform.

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
| 11.1 | Explain different quantization methods and their tradeoffs | Understand |
| 11.2 | Quantize models using GPTQ, AWQ, and GGUF | Apply |
| 11.3 | Apply Blackwell-exclusive FP4 quantization | Apply |
| 11.4 | Measure and compare quality degradation | Evaluate |

---

## Topics

### 11.1 Quantization Fundamentals
- Data types: FP32 → FP16 → BF16 → INT8 → INT4 → FP8 → FP4
- Post-training quantization vs quantization-aware training
- Calibration datasets

### 11.2 Quantization Methods

| Method | Bits | Best For | DGX Spark Support |
|--------|------|----------|-------------------|
| GPTQ | 4-bit | GPU inference | ✅ |
| AWQ | 4-bit | Activation-aware | ✅ |
| GGUF | 2-8 bit | llama.cpp | ✅ |
| FP8 | 8-bit | Training + inference | ✅ |
| **NVFP4** | 4-bit | Blackwell exclusive | ⭐ **Only on DGX Spark** |

### 11.3 Blackwell-Specific (DGX Spark Exclusive)
- NVFP4 format with dual-level scaling
- MXFP4 (Open Compute Project)
- 3.5× memory reduction vs FP16
- <1% accuracy loss

---

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 11.1 | Quantization Overview | 2h | Compare FP16/INT8/INT4: size, speed, perplexity |
| 11.2 | GPTQ Quantization | 2h | Quantize 7B with group sizes 32/64/128 |
| 11.3 | AWQ Quantization | 1.5h | Compare AWQ vs GPTQ |
| 11.4 | GGUF Conversion | 2h | Convert to GGUF, test with llama.cpp |
| 11.5 | FP4 Deep Dive ⭐ | 3h | NVFP4 quantization (Blackwell exclusive!) |
| 11.6 | Quality Benchmark Suite | 2h | Perplexity + MMLU across all variants |

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

---

## Milestone Checklist

- [ ] Quantization comparison table (size, speed, perplexity)
- [ ] GPTQ with multiple group sizes
- [ ] AWQ vs GPTQ comparison
- [ ] GGUF conversion and llama.cpp testing
- [ ] **NVFP4 quantization** ⭐ (DGX Spark exclusive)
- [ ] Quality benchmark suite with all variants

---

## Resources

- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [NVIDIA TensorRT Model Optimizer](https://developer.nvidia.com/tensorrt)
