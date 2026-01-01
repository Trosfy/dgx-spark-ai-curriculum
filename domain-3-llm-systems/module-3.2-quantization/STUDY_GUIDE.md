# Module 3.2: Quantization & Optimization - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Explain** quantization fundamentals and data type tradeoffs
2. **Apply** NVFP4 quantization using TensorRT Model Optimizer (Blackwell exclusive)
3. **Quantize** models using GPTQ, AWQ, and GGUF methods
4. **Evaluate** quantization impact on model quality using perplexity and benchmarks

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-3.2.1-data-type-exploration.ipynb | FP32→FP16→FP8→FP4 comparison | ~1.5 hr | Visualize precision loss across formats |
| 2 | lab-3.2.2-nvfp4-quantization.ipynb | NVFP4 on 70B model ⭐ | ~3 hr | DGX Spark showcase: 3.5× memory savings |
| 3 | lab-3.2.3-fp8-training-inference.ipynb | FP8 E4M3/E5M2 formats | ~2 hr | Native Blackwell FP8 tensor cores |
| 4 | lab-3.2.4-gptq-quantization.ipynb | GPTQ with group sizes | ~2 hr | Quantize with 32/64/128 group sizes |
| 5 | lab-3.2.5-awq-quantization.ipynb | Activation-aware quantization | ~1.5 hr | Compare AWQ vs GPTQ quality |
| 6 | lab-3.2.6-gguf-conversion.ipynb | GGUF for llama.cpp/Ollama | ~2 hr | Convert models for universal deployment |
| 7 | lab-3.2.7-quality-benchmark-suite.ipynb | Perplexity + MMLU testing | ~2 hr | Comprehensive quality comparison |
| 8 | lab-3.2.8-tensorrt-llm-engine.ipynb | TensorRT-LLM optimization | ~2 hr | Build TRT engine with NVFP4 |

**Total time**: ~16 hours

## 🔑 Core Concepts

This module introduces these fundamental ideas:

### Quantization
**What**: Reducing the precision of model weights to decrease memory and increase speed
**Why it matters**: Enables running 70B-200B models on DGX Spark's 128GB unified memory
**First appears in**: Lab 3.2.1

### NVFP4 (NVIDIA FP4)
**What**: Blackwell-exclusive 4-bit floating point with micro-block scaling
**Why it matters**: 3.5× memory reduction with <0.1% accuracy loss—only available on DGX Spark
**First appears in**: Lab 3.2.2

### FP8 (E4M3/E5M2)
**What**: 8-bit floating point formats optimized for inference (E4M3) and training (E5M2)
**Why it matters**: 2× compute efficiency with native Blackwell tensor core support
**First appears in**: Lab 3.2.3

### Calibration Data
**What**: Representative samples used to determine optimal scaling factors for quantization
**Why it matters**: Quality of calibration data directly affects quantization quality
**First appears in**: Lab 3.2.4

## 🔗 How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 3.1          ──►     Module 3.2          ──►    Module 3.3
Fine-tuning                 Quantization              Deployment
(create models)             (compress models)         (serve models)
```

**Builds on**:
- **Model loading** from Module 3.1 (BitsAndBytesConfig)
- **Memory management** from Module 3.1 (GPU memory optimization)
- **Model architecture** from Module 2.3 (understanding layers to quantize)

**Prepares for**:
- Module 3.3 will deploy quantized models with TensorRT-LLM
- Module 3.5 will use quantized models for RAG systems
- Module 4.4 will containerize quantized models for production

## 📖 Recommended Approach

### Standard Path (12-16 hours):
1. **Day 1: Fundamentals (Labs 1-3)**
   - Start with Lab 3.2.1 to understand data types
   - Lab 3.2.2 is the module highlight: NVFP4 on 70B
   - Lab 3.2.3 covers FP8 for training

2. **Day 2: Quantization Methods (Labs 4-6)**
   - Labs 4-5 cover GPTQ and AWQ for GPU inference
   - Lab 6 covers GGUF for Ollama deployment

3. **Day 3: Quality & TensorRT (Labs 7-8)**
   - Lab 7 establishes quality baselines
   - Lab 8 builds production TensorRT engines

### Quick Path (8-10 hours, if experienced):
1. Skim Lab 3.2.1, focus on FP4/FP8 sections
2. Do Lab 3.2.2 (NVFP4) - this is essential for DGX Spark
3. Choose ONE of GPTQ/AWQ/GGUF based on your deployment target
4. Complete Lab 3.2.7 for quality verification
5. Lab 3.2.8 if targeting TensorRT deployment

### Deep-Dive Path (20+ hours):
1. Complete all labs in sequence
2. Quantize multiple model sizes (8B, 30B, 70B) with each method
3. Create your own quality benchmark suite
4. Optimize TensorRT engines for specific batch sizes

## 📋 Before You Start
→ See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
