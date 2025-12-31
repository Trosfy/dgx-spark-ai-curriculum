# Domain 2: Deep Learning Frameworks - Overview

## Domain Learning Objectives

By completing this domain, you will:

1. **Master PyTorch fundamentals** - Build, train, debug, and profile neural networks professionally
2. **Apply deep learning to vision and language** - Implement CNNs, transformers, and attention mechanisms
3. **Understand modern architectures** - Work with Mamba (State Space Models) and Mixture of Experts
4. **Use the Hugging Face ecosystem** - Navigate Hub, use pipelines, fine-tune with Trainer and PEFT
5. **Generate images with diffusion models** - Run Stable Diffusion, Flux, train custom LoRAs

---

## Domain Roadmap

```
Week 8-9         Week 10-11       Week 12          Week 13
    │                │               │               │
    ▼                ▼               ▼               ▼
┌────────┐      ┌────────┐      ┌────────┐      ┌────────┐
│  2.1   │─────►│  2.2   │─────►│  2.3   │─────►│  2.4   │
│PyTorch │      │  CV    │      │  NLP   │      │Efficient│
└────────┘      └────────┘      └────────┘      └────────┘
Foundation      Images          Transformers    Mamba/MoE
                                                    │
                                                    ▼
                              Week 15          Week 14
                                  │               │
                                  ▼               ▼
                              ┌────────┐      ┌────────┐
                              │  2.6   │◄─────│  2.5   │
                              │Diffusion│      │  HF    │
                              └────────┘      └────────┘
                              Generation      Ecosystem
```

---

## Module Summaries

### Module 2.1: Deep Learning with PyTorch
**Focus**: Professional PyTorch skills - custom modules, data pipelines, AMP, profiling
**Key deliverable**: ResNet-18 implementation with mixed-precision training
**Time**: ~12-15 hours (Weeks 8-9)
**Priority**: Core foundation

### Module 2.2: Computer Vision
**Focus**: CNN architectures, transfer learning, YOLO, Vision Transformers, SAM
**Key deliverable**: Fine-tuned model with >90% accuracy + object detection demo
**Time**: ~14-16 hours (Weeks 10-11)
**Priority**: P2 Expanded

### Module 2.3: NLP & Transformers
**Focus**: Attention from scratch, transformer architecture, tokenizers, BERT/GPT fine-tuning
**Key deliverable**: Complete transformer implementation + text generation system
**Time**: ~12-15 hours (Week 12)
**Priority**: P2 Expanded

### Module 2.4: Efficient Architectures
**Focus**: Mamba (State Space Models), Mixture of Experts, hybrid architectures
**Key deliverable**: Architecture comparison report + Mamba fine-tuning
**Time**: ~10-12 hours (Week 13)
**Priority**: P1 High

### Module 2.5: Hugging Face Ecosystem
**Focus**: Hub, Transformers library, Datasets, Trainer API, PEFT/LoRA
**Key deliverable**: Fine-tuned model uploaded to HF Hub with model card
**Time**: ~10-12 hours (Week 14)
**Priority**: Core

### Module 2.6: Diffusion Models
**Focus**: DDPM theory, Stable Diffusion, Flux, ControlNet, LoRA training
**Key deliverable**: Custom LoRA-trained diffusion model + generation pipeline
**Time**: ~10-12 hours (Week 15)
**Priority**: P1 High

---

## Prerequisites from Previous Domains

| This Domain Needs | From Domain 1 | Specifically |
|-------------------|---------------|--------------|
| Neural network fundamentals | Module 1.5 | Backprop, gradients, loss functions |
| Python proficiency | Module 1.2 | OOP, NumPy, debugging |
| CUDA awareness | Module 1.3 | GPU memory, kernel basics |
| Math foundations | Module 1.4 | Linear algebra, calculus |
| MicroGrad experience | Module 1.7 | Building from scratch mindset |

**Domain 1 Capstone required**: Complete Module 1.7 (MicroGrad+) before starting Domain 2.

---

## What This Domain Prepares You For

| Future Module | Will Use | From This Domain |
|---------------|----------|------------------|
| Module 3.1 (LLM Fine-Tuning) | LoRA, PEFT, Trainer API | Module 2.5 |
| Module 3.2 (Quantization) | Model loading, inference | Modules 2.4, 2.5 |
| Module 3.3 (LLM Deployment) | Transformers knowledge | Module 2.3, 2.4 |
| Module 3.5 (RAG) | Embeddings, tokenizers | Module 2.3, 2.5 |
| Module 4.1 (Multimodal) | Vision + language | Modules 2.2, 2.3 |

---

## DGX Spark Advantages in This Domain

| Module | DGX Spark Benefit |
|--------|-------------------|
| 2.1 PyTorch | Large batch sizes (128GB), native BF16 AMP |
| 2.2 Computer Vision | SAM ViT-H fits easily, train ViT-Large |
| 2.3 NLP | Fine-tune without gradient checkpointing |
| 2.4 Efficient | Mamba 100K+ context, full MoE models (Mixtral) |
| 2.5 Hugging Face | Load 50B+ models in BF16 directly |
| 2.6 Diffusion | SDXL + Refiner, train LoRAs at full precision |

---

## Domain Completion Checklist

### Module Milestones
- [ ] **2.1**: ResNet-18 + AMP training + profiling report
- [ ] **2.2**: CNN comparison + ViT implementation + SAM demo
- [ ] **2.3**: Transformer encoder + tokenizer + text generation
- [ ] **2.4**: Mamba benchmark + MoE analysis + comparison report
- [ ] **2.5**: Pipeline demos + Trainer fine-tuning + Hub upload
- [ ] **2.6**: DDPM from scratch + SDXL/Flux generation + LoRA training

### Skills Verification
- [ ] Can explain attention mechanism and implement from scratch
- [ ] Can fine-tune pre-trained models on custom datasets
- [ ] Can profile and optimize PyTorch training loops
- [ ] Can choose appropriate architecture for different tasks
- [ ] Can navigate HF Hub and use Trainer API effectively

---

## Recommended Study Order

### Standard Path (8 weeks)
```
Week 8-9:   Module 2.1 (PyTorch) - Foundation
Week 10-11: Module 2.2 (Computer Vision) - Images
Week 12:    Module 2.3 (NLP & Transformers) - Text
Week 13:    Module 2.4 (Efficient Architectures) - Modern
Week 14:    Module 2.5 (Hugging Face) - Ecosystem
Week 15:    Module 2.6 (Diffusion Models) - Generation
```

### Accelerated Path (5 weeks, strong prerequisites)
```
Week 1: 2.1 - Focus on AMP and profiling (skip basic PyTorch)
Week 2: 2.2 + 2.3 - ViT connects CV to NLP
Week 3: 2.4 - Mamba and MoE critical for modern LLMs
Week 4: 2.5 - Essential for Domain 3
Week 5: 2.6 - If time permits
```

### Deep-Dive Path (10+ weeks)
```
Week 8-9:   2.1 + implement additional architectures
Week 10-11: 2.2 + train on custom dataset
Week 12-13: 2.3 + build mini-GPT
Week 14:    2.4 + in-depth Mamba paper study
Week 15:    2.5 + advanced PEFT techniques
Week 16:    2.6 + train custom Flux LoRA
```

---

## Domain Project Ideas

After completing this domain, try these projects:

### 1. Custom Image Classifier
Build an end-to-end image classification system:
- Use Module 2.2 CNN/ViT architectures
- Apply Module 2.5 Trainer API
- Deploy with Module 2.1 profiling optimizations

### 2. Document Question Answering
Create a system that answers questions about documents:
- Use Module 2.3 attention/transformers
- Apply Module 2.5 BERT fine-tuning
- Extend in Domain 3 with RAG

### 3. Art Style Generator
Build a custom art style generator:
- Use Module 2.6 LoRA training
- Apply Module 2.5 Hub upload
- Create Gradio demo (Domain 4)

### 4. Efficient Long-Context Summarizer
Build a summarization system for long documents:
- Use Module 2.4 Mamba for long contexts
- Apply Module 2.3 transformer knowledge
- Compare architectures systematically

---

## Resources by Topic

### PyTorch Mastery
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [d2l.ai - Dive into Deep Learning](https://d2l.ai/)

### Computer Vision
- [CS231n - Stanford](http://cs231n.stanford.edu/)
- [timm library](https://github.com/huggingface/pytorch-image-models)

### Transformers & NLP
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Efficient Architectures
- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [Mixtral Paper](https://arxiv.org/abs/2401.04088)

### Hugging Face
- [HF Course](https://huggingface.co/learn/nlp-course)
- [PEFT Documentation](https://huggingface.co/docs/peft)

### Diffusion Models
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)

---

## Module Navigation

| Previous Domain | Current | Next Domain |
|-----------------|---------|-------------|
| [Domain 1: Platform Foundations](../domain-1-platform-foundations/) | **Domain 2: Deep Learning Frameworks** | [Domain 3: LLM Systems](../domain-3-llm-systems/) |
