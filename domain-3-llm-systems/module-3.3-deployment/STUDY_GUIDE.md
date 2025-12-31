# Module 3.3: Model Deployment & Inference Engines - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Compare** inference engines and select optimal for specific use cases
2. **Implement** speculative decoding with Medusa and EAGLE for 2-3x speedup
3. **Configure** continuous batching and PagedAttention for high throughput
4. **Deploy** production-ready REST APIs with streaming and monitoring

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-3.3.1-engine-benchmark.ipynb | Compare all engines | ~3 hr | Comprehensive benchmark report |
| 2 | lab-3.3.2-sglang-deployment.ipynb | SGLang RadixAttention | ~2 hr | 29-45% faster with prefix caching |
| 3 | lab-3.3.3-vllm-continuous-batching.ipynb | vLLM PagedAttention | ~2 hr | High throughput serving |
| 4 | lab-3.3.4-speculative-decoding.ipynb | Medusa implementation | ~2 hr | 2-3x decode speedup |
| 5 | lab-3.3.5-eagle-implementation.ipynb | EAGLE comparison | ~2 hr | Compare with Medusa |
| 6 | lab-3.3.6-tensorrt-llm-optimization.ipynb | TensorRT-LLM engine | ~2 hr | Maximum prefill performance |
| 7 | lab-3.3.7-production-api.ipynb | FastAPI with monitoring | ~2 hr | Production-ready server |

**Total time**: ~15 hours

## 🔑 Core Concepts

This module introduces these fundamental ideas:

### Continuous Batching
**What**: Dynamically grouping requests as they arrive to maximize GPU utilization
**Why it matters**: 10-100x throughput improvement over naive serving
**First appears in**: Lab 3.3.3

### PagedAttention
**What**: Memory-efficient KV cache management using paging (like OS virtual memory)
**Why it matters**: Eliminates memory fragmentation, enables more concurrent requests
**First appears in**: Lab 3.3.3

### Speculative Decoding
**What**: Draft-and-verify approach using small model predictions verified by target model
**Why it matters**: 2-3x speedup without quality loss
**First appears in**: Lab 3.3.4

### RadixAttention
**What**: SGLang's prefix caching that reuses computation for shared prompts
**Why it matters**: 29-45% faster for chat apps with system prompts
**First appears in**: Lab 3.3.2

## 🔗 How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 3.2          ──►     Module 3.3          ──►    Module 3.4
Quantization                Deployment                Test-Time Compute
(compress models)           (serve models)            (reasoning)
```

**Builds on**:
- **Quantized models** from Module 3.2 (GPTQ, AWQ, NVFP4)
- **Memory management** from Module 3.1-3.2
- **Model architecture** from Module 2.3 (attention for KV cache)

**Prepares for**:
- Module 3.4 will use deployed models for reasoning experiments
- Module 3.5 will deploy RAG systems using these engines
- Module 4.3 will add MLOps monitoring to deployments

## 📖 Recommended Approach

### Standard Path (12-15 hours):
1. **Day 1: Fundamentals (Labs 1-3)**
   - Lab 3.3.1 establishes baselines for all engines
   - Lab 3.3.2 covers SGLang (fastest for chat)
   - Lab 3.3.3 covers vLLM (most popular)

2. **Day 2: Speedups (Labs 4-6)**
   - Labs 4-5 cover speculative decoding (major speedup!)
   - Lab 6 covers TensorRT-LLM (maximum performance)

3. **Day 3: Production (Lab 7)**
   - Build production API with streaming and monitoring

### Quick Path (8-10 hours, if experienced):
1. Skim Lab 3.3.1, focus on benchmark results
2. Do Lab 3.3.2 (SGLang) OR Lab 3.3.3 (vLLM) based on your use case
3. Lab 3.3.4 (Medusa speculative decoding)
4. Lab 3.3.7 (production API)

### Deep-Dive Path (20+ hours):
1. Complete all labs in sequence
2. Benchmark each engine with multiple model sizes
3. Implement custom speculative decoding
4. Build full production deployment with monitoring

## 📋 Before You Start
→ See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
