# Module 3.1: LLM Fine-Tuning - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Implement** LoRA, QLoRA, and DoRA fine-tuning from scratch
2. **Fine-tune** 70B+ parameter models on DGX Spark using QLoRA
3. **Apply** preference optimization methods (DPO, SimPO, ORPO, KTO)
4. **Evaluate** fine-tuned models and deploy via Ollama

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-3.1.1-lora-theory.ipynb | LoRA mathematics & implementation | ~2 hr | Understand W = W₀ + BA decomposition |
| 2 | lab-3.1.2-dora-comparison.ipynb | DoRA weight decomposition | ~2 hr | Measure +3.7 point improvement |
| 3 | lab-3.1.3-neftune-magic.ipynb | Noisy embedding training | ~1 hr | Implement 5-line quality boost |
| 4 | lab-3.1.4-8b-lora-finetuning.ipynb | Full 8B fine-tuning pipeline | ~3 hr | Complete LoRA training workflow |
| 5 | lab-3.1.5-70b-qlora-finetuning.ipynb | 70B model with QLoRA ⭐ | ~4 hr | DGX Spark showcase task |
| 6 | lab-3.1.6-dataset-preparation.ipynb | Instruction & preference data | ~2 hr | Create production-ready datasets |
| 7 | lab-3.1.7-dpo-training.ipynb | Direct Preference Optimization | ~2 hr | Train without reward model |
| 8 | lab-3.1.8-simpo-vs-orpo.ipynb | Modern preference methods | ~2 hr | Compare SimPO, ORPO memory/quality |
| 9 | lab-3.1.9-kto-binary-feedback.ipynb | Binary feedback training | ~2 hr | Train with thumbs up/down data |
| 10 | lab-3.1.10-ollama-integration.ipynb | Deploy fine-tuned model | ~2 hr | GGUF conversion and Ollama testing |

**Total time**: ~22 hours

## 🔑 Core Concepts

This module introduces these fundamental ideas:

### LoRA (Low-Rank Adaptation)
**What**: Adds small trainable matrices alongside frozen model weights: W' = W + BA
**Why it matters**: Reduces trainable parameters to ~0.1%, enabling fine-tuning of massive models on limited hardware
**First appears in**: Lab 3.1.1

### QLoRA (Quantized LoRA)
**What**: Combines 4-bit quantization of base model with LoRA adapters
**Why it matters**: Fine-tune 70B models in ~45GB on DGX Spark's 128GB unified memory
**First appears in**: Lab 3.1.5

### Preference Optimization
**What**: Training models to prefer "good" responses over "bad" ones using comparison data
**Why it matters**: Aligns models with human preferences without complex reward modeling
**First appears in**: Lab 3.1.7

### NEFTune
**What**: Adding noise to embeddings during training for improved instruction following
**Why it matters**: Simple 5-line change that can double benchmark performance
**First appears in**: Lab 3.1.3

## 🔗 How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 2.5/2.6      ──►     Module 3.1          ──►    Module 3.2
PEFT concepts,              Fine-tuning               Quantization for
Diffusion models            LoRA, DPO                 deployment
```

**Builds on**:
- **PEFT library** from Module 2.5 (HuggingFace Ecosystem)
- **Transformer architecture** from Module 2.3 (NLP & Transformers)
- **GPU memory management** from Module 1.3 (CUDA Python)

**Prepares for**:
- Module 3.2 will use fine-tuned models for quantization experiments
- Module 3.3 will deploy fine-tuned models with vLLM/SGLang
- Module 3.5 will fine-tune models for RAG-specific tasks

## 📖 Recommended Approach

### Standard Path (18-22 hours):
1. **Week 1: LoRA Fundamentals (Labs 1-4)**
   - Start with Lab 3.1.1 to understand LoRA math
   - Complete Labs 2-3 for DoRA and NEFTune enhancements
   - Lab 4 ties everything together with full 8B training

2. **Week 2: QLoRA & Datasets (Labs 5-6)**
   - Lab 5 is the showcase: 70B fine-tuning on DGX Spark
   - Lab 6 prepares datasets for preference optimization

3. **Week 3: Preference Optimization & Deployment (Labs 7-10)**
   - Labs 7-9 cover all major preference methods
   - Lab 10 deploys your model to Ollama

### Quick Path (10-12 hours, if experienced):
1. Skim Lab 3.1.1, focus on rank selection guidance
2. Skip to Lab 3.1.4 for hands-on LoRA training
3. Do Lab 3.1.5 (70B QLoRA) - this is essential
4. Choose ONE preference method (recommend SimPO via Lab 3.1.8)
5. Complete Lab 3.1.10 for deployment

### Deep-Dive Path (25+ hours):
1. Complete all labs in sequence
2. After each preference method lab, run extended benchmarks
3. Fine-tune on your own custom dataset
4. Compare all preference methods on the same task

## 📋 Before You Start
→ See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success
