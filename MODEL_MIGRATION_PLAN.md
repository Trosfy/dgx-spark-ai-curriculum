# 2025 Model Migration Plan - DGX Spark AI Curriculum

## Overview

This document outlines the comprehensive plan to update all model references across the DGX Spark AI Curriculum to the 2025 Tier 1 model stack.

---

## 📋 Model Migration Matrix

### Primary Model Replacements

| Category | OLD Model | NEW Model | Capabilities | Notes |
|----------|-----------|-----------|--------------|-------|
| **General LLM** | llama3.1:8b | nemotron-3-nano | Think ✅ Tools ✅ Vision ❌ | 1M context, 4x faster |
| **General LLM (Alt)** | llama3.1:70b | qwen3:32b | Think ✅ Tools ✅ Vision ❌ | Best BFCL (68.2) |
| **Reasoning** | deepseek-r1:8b/70b | qwq:32b | Think ✅ Tools ⚠️ Vision ❌ | 79.5% AIME, Apache 2.0 |
| **Reasoning+Vision** | - | magistral-small-1.2 | Think ✅ Tools ✅ Vision ✅ | Multimodal reasoning |
| **Code** | qwen2.5-coder:7b | devstral-small-2 | Think ✅ Tools ✅ Vision ✅ | 68% SWE-Bench |
| **Embeddings** | nomic-embed-text | qwen3-embedding:8b | - | #1 MTEB, 100+ langs |
| **Vision (General)** | llava:7b/13b | qwen3-vl:8b | Think ✅ Tools ✅ | 32-lang OCR, GUI agents |
| **Vision (OCR)** | - | deepseek-ocr | Think ❌ Tools ❌ | Document extraction |
| **Image Gen** | FLUX.1-schnell/dev | flux.2-dev | Inpaint ✅ Outpaint ✅ | Non-commercial free |
| **Image Gen (SDXL)** | stable-diffusion-xl | flux.2-dev | - | FLUX.2 supersedes SDXL |

### Capability Reference

| Model | Thinking | Tool Calling | Vision | Context | License |
|-------|----------|--------------|--------|---------|---------|
| nemotron-3-nano | ✅ ON/OFF | ✅ Strong | ❌ | 1M | Open |
| qwen3:8b | ✅ /think | ✅ BFCL 60.2 | ❌ | 131K | Apache 2.0 |
| qwen3:32b | ✅ /think | ✅ BFCL 68.2 | ❌ | 131K | Apache 2.0 |
| qwq:32b | ✅ Always | ⚠️ BFCL 66.4 | ❌ | 131K | Apache 2.0 |
| deepseek-r1:8b | ✅ `<think>` | ❌ Unstable | ❌ | 64K | MIT |
| magistral-small-1.2 | ✅ Traceable | ✅ Enhanced | ✅ | 128K | Apache 2.0 |
| devstral-small-2 | ✅ | ✅ Function | ✅ | 256K | Apache 2.0 |
| qwen3-vl:8b | ✅ Thinking | ✅ GUI agents | ✅ | 256K-1M | Apache 2.0 |
| deepseek-ocr | ❌ | ❌ | ✅ OCR only | - | Open |
| qwen3-embedding:8b | - | - | - | 32K | Apache 2.0 |

---

## 📁 Files to Update

### Phase 1: Core Documentation (High Priority)

#### 1.1 docs/SETUP.md
- [ ] Update "Pull Recommended Models" section
- [ ] Add Nemotron 3 Nano as primary general LLM
- [ ] Add Devstral Small 2 as primary code model
- [ ] Add Magistral Small 1.2 for reasoning+vision
- [ ] Update model notes with capability matrix
- [ ] Add DeepSeek-OCR for document processing

#### 1.2 CURRICULUM.md
- [ ] Update performance benchmarks table
- [ ] Update model recommendations per domain
- [ ] Add capability notes (tool calling, vision, etc.)

### Phase 2: Domain 1 - Platform Foundations

#### 2.1 module-1.1-dgx-spark-platform/
- [ ] LAB_PREP.md - Update ollama pull commands
- [ ] QUICK_REFERENCE.md - Update model performance tables
- [ ] labs/*.ipynb - Update model references in code cells

#### 2.2 module-1.2-ngc-containers/
- [ ] LAB_PREP.md - Update container + model setup
- [ ] QUICK_REFERENCE.md - Update model lists

### Phase 3: Domain 2 - ML Fundamentals

#### 3.1 module-2.1-pytorch-fundamentals/
- [ ] No model changes needed (PyTorch basics)

#### 3.2 module-2.2-transformer-architectures/
- [ ] Update model architecture references if any

### Phase 4: Domain 3 - LLM Systems (Most Changes)

#### 4.1 module-3.1-llm-finetuning/
- [ ] LAB_PREP.md - Update base model references
- [ ] QUICK_REFERENCE.md - Update model recommendations
- [ ] labs/*.ipynb - Update all notebook model references
- [ ] Note: Keep some legacy models for fine-tuning comparison

#### 4.2 module-3.2-quantization/
- [ ] Update quantization target models
- [ ] Update memory estimates for new models
- [ ] Update performance benchmarks

#### 4.3 module-3.3-deployment/
- [ ] Update vLLM/TensorRT-LLM model examples
- [ ] Update inference optimization references

#### 4.4 module-3.4-test-time-compute/ ✅ (Partially done)
- [ ] LAB_PREP.md - Add Magistral Small comparison
- [ ] QUICK_REFERENCE.md - Add Magistral to reasoning comparison
- [ ] Add note: DeepSeek-R1 lacks tool calling
- [ ] labs/*.ipynb - Update notebook examples

#### 4.5 module-3.5-rag-systems/ ✅ (Partially done)
- [ ] LAB_PREP.md - Add DeepSeek-OCR for document processing
- [ ] QUICK_REFERENCE.md - Update embedding model references
- [ ] labs/*.ipynb - Update to qwen3-embedding:8b

#### 4.6 module-3.6-ai-agents/ ✅ (Partially done)
- [ ] LAB_PREP.md - Add Nemotron 3 Nano for agents
- [ ] QUICK_REFERENCE.md - Add capability matrix
- [ ] Add warning: DeepSeek-R1 NOT suitable for agents (no tool calling)
- [ ] labs/*.ipynb - Update agent examples

### Phase 5: Domain 4 - Production AI

#### 5.1 module-4.1-multimodal/
- [ ] LAB_PREP.md - Update VLM references
- [ ] QUICK_REFERENCE.md - Add Qwen3-VL, Magistral 1.2, DeepSeek-OCR comparison
- [ ] Update FLUX.1 → FLUX.2 dev references
- [ ] labs/*.ipynb - Update all vision/image notebooks

#### 5.2 module-4.2-ai-safety/
- [ ] Update guard model references
- [ ] Update safety evaluation models

#### 5.3 module-4.3-mlops/
- [ ] Update deployment model examples

#### 5.4 module-4.4-edge-deployment/
- [ ] Update edge model recommendations
- [ ] Consider Nemotron 3 Nano for efficiency

#### 5.5 module-4.5-demo-building/
- [ ] Update demo model recommendations
- [ ] Update Gradio/Streamlit examples

### Phase 6: Jupyter Notebooks (.ipynb files)

All notebooks need code cell updates for:
- [ ] Model name strings
- [ ] ollama.chat() model parameters
- [ ] HuggingFace model IDs
- [ ] Memory estimates in comments
- [ ] Performance expectations in markdown

---

## 🔄 Migration Patterns

### Pattern 1: Ollama Model References
```python
# OLD
response = ollama.chat(model="llama3.1:8b", ...)

# NEW (General)
response = ollama.chat(model="nemotron-3-nano", ...)  # or qwen3:8b

# NEW (Reasoning)
response = ollama.chat(model="qwq:32b", ...)

# NEW (Code)
response = ollama.chat(model="devstral-small-2", ...)
```

### Pattern 2: HuggingFace Model IDs
```python
# OLD
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# NEW
model = AutoModel.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
# or
model = AutoModel.from_pretrained("Qwen/Qwen3-8B-Instruct")
```

### Pattern 3: Embedding Models
```python
# OLD
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# NEW
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
```

### Pattern 4: Vision Models
```python
# OLD
model = "llava:13b"

# NEW (General VLM)
model = "qwen3-vl:8b"

# NEW (Document OCR)
model = "deepseek-ocr"  # via HuggingFace/vLLM
```

### Pattern 5: Image Generation
```python
# OLD
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

# NEW
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.2-dev")
```

### Pattern 6: Agent Tool Calling (CRITICAL)
```python
# WARNING: DeepSeek-R1 does NOT support tool calling!

# OLD (BROKEN for agents)
agent_llm = Ollama(model="deepseek-r1:8b")  # ❌ No tool support

# NEW (Correct for agents)
agent_llm = Ollama(model="nemotron-3-nano")  # ✅ Strong tool calling
# or
agent_llm = Ollama(model="qwen3:32b")  # ✅ Best BFCL score
```

---

## ⚠️ Critical Notes

### Models WITHOUT Tool Calling
These models should NOT be used for agentic tasks:
- `deepseek-r1:*` - No native tool calling, workarounds unstable
- `deepseek-ocr` - OCR only, no tool calling

### Models WITH Vision
Use these for multimodal tasks:
- `qwen3-vl:8b` - Full VLM with tool calling
- `magistral-small-1.2` - Reasoning + vision
- `devstral-small-2` - Code + vision (image inputs)
- `deepseek-ocr` - Document OCR only

### Reasoning Model Comparison
| Model | AIME 2024 | Tool Calling | Best For |
|-------|-----------|--------------|----------|
| QwQ-32B | 79.5% | ⚠️ Via Qwen-Agent | Pure reasoning |
| DeepSeek-R1 | 79.8% | ❌ No | Reasoning (non-agentic) |
| Magistral Small 1.2 | 86.1% | ✅ Yes | Reasoning + vision + tools |

---

## 📊 Estimated Impact

| Category | Files | Notebooks | Estimated Changes |
|----------|-------|-----------|-------------------|
| Core Docs | 5 | 0 | ~50 lines each |
| Domain 1 | 8 | 4 | ~30 lines each |
| Domain 2 | 4 | 6 | ~10 lines each |
| Domain 3 | 18 | 24 | ~100 lines each |
| Domain 4 | 15 | 18 | ~80 lines each |
| **Total** | **50+** | **52+** | **~3000 lines** |

---

## ✅ Execution Checklist

### Pre-Migration
- [ ] Backup current branch
- [ ] Create migration tracking issue
- [ ] Verify all new models available in Ollama

### Migration
- [ ] Phase 1: Core Documentation
- [ ] Phase 2: Domain 1
- [ ] Phase 3: Domain 2
- [ ] Phase 4: Domain 3
- [ ] Phase 5: Domain 4
- [ ] Phase 6: Notebooks

### Post-Migration
- [ ] Run verification scripts
- [ ] Test sample notebooks
- [ ] Update any broken links
- [ ] Commit with detailed message
- [ ] Push to branch

---

## 🚀 Ready to Execute?

This plan covers:
- **50+ markdown/documentation files**
- **52+ Jupyter notebooks**
- **~3000 lines of changes**
- **10 model category migrations**

Approval needed before proceeding with updates.
