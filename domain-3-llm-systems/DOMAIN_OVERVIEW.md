# Domain 3: LLM Systems - Overview

## 🎯 What This Domain Covers

Domain 3 takes you from understanding LLMs to building complete, production-ready AI systems. You'll learn to fine-tune models, optimize them for deployment, enhance their reasoning capabilities, build knowledge-grounded applications with RAG, and create autonomous AI agents.

**In one sentence**: Everything you need to go from a pre-trained model to a deployed, intelligent application.

---

## 📊 Domain at a Glance

| Attribute | Value |
|-----------|-------|
| **Modules** | 6 |
| **Estimated Time** | 70-90 hours |
| **Prerequisites** | Domains 1-2 (Hardware + ML Fundamentals) |
| **Key Outcome** | Build and deploy complete LLM-powered applications |

---

## 🗺️ Module Overview

```
Domain 3: LLM Systems
│
├── Module 3.1: LLM Fine-Tuning (Weeks 17-19)
│   └── Customize models for your use case
│
├── Module 3.2: Quantization (Weeks 20-21)
│   └── Compress models to fit on DGX Spark
│
├── Module 3.3: Deployment (Weeks 22-23)
│   └── Serve models efficiently with vLLM, SGLang
│
├── Module 3.4: Test-Time Compute (Week 23.5)
│   └── Enhance reasoning with Chain-of-Thought
│
├── Module 3.5: RAG Systems (Weeks 24-25)
│   └── Ground models in your documents
│
└── Module 3.6: AI Agents (Week 26)
    └── Build autonomous tool-using agents
```

---

## 📦 Module Details

### Module 3.1: LLM Fine-Tuning
**Time**: 15-20 hours | **Priority**: P0 Critical

Fine-tune LLMs to follow instructions, adopt specific styles, or excel at domain tasks.

**Key Topics**:
- LoRA, QLoRA, DoRA (parameter-efficient fine-tuning)
- SFT (Supervised Fine-Tuning) for instruction following
- DPO, SimPO, ORPO (preference alignment)
- Dataset preparation and formatting

**Deliverables**:
- Fine-tuned model with custom style
- Preference-aligned model

→ [Go to Module 3.1](./module-3.1-llm-finetuning/)

---

### Module 3.2: Quantization
**Time**: 10-12 hours | **Priority**: P0 Critical

Compress models to fit in memory while maintaining quality.

**Key Topics**:
- NVFP4, FP8 (Blackwell-native formats)
- GPTQ, AWQ, GGUF quantization
- Quality vs memory tradeoffs
- Model conversion and optimization

**Deliverables**:
- 70B model running in <50GB memory
- Quantization quality comparison

→ [Go to Module 3.2](./module-3.2-quantization/)

---

### Module 3.3: Model Deployment
**Time**: 12-15 hours | **Priority**: P0 Critical

Serve models efficiently with production-grade inference engines.

**Key Topics**:
- vLLM with PagedAttention
- SGLang with RadixAttention
- TensorRT-LLM optimization
- Speculative decoding (Medusa, EAGLE)
- Ollama for development

**Deliverables**:
- OpenAI-compatible API endpoint
- Benchmarked inference performance

→ [Go to Module 3.3](./module-3.3-deployment/)

---

### Module 3.4: Test-Time Compute
**Time**: 6-8 hours | **Priority**: P1 Important

Enhance model reasoning at inference time.

**Key Topics**:
- Chain-of-Thought prompting
- Self-Consistency (multiple reasoning paths)
- DeepSeek-R1 reasoning patterns
- Compute-optimal scaling

**Deliverables**:
- Reasoning benchmark comparison
- Optimal prompting strategies

→ [Go to Module 3.4](./module-3.4-test-time-compute/)

---

### Module 3.5: RAG Systems
**Time**: 12-15 hours | **Priority**: P0 Critical

Build knowledge-grounded applications with retrieval.

**Key Topics**:
- Vector embeddings and similarity search
- ChromaDB, FAISS, Qdrant databases
- Chunking strategies
- Hybrid search (dense + sparse)
- Reranking with cross-encoders
- RAGAS evaluation

**Deliverables**:
- Production RAG pipeline
- Evaluated retrieval system

→ [Go to Module 3.5](./module-3.5-rag-systems/)

---

### Module 3.6: AI Agents
**Time**: 10-12 hours | **Priority**: P0 Critical

Build autonomous agents that use tools and collaborate.

**Key Topics**:
- ReAct pattern (Reasoning + Acting)
- Custom tool creation
- LangGraph for complex workflows
- Multi-agent systems (CrewAI)
- Human-in-the-loop patterns

**Deliverables**:
- Multi-tool ReAct agent
- Multi-agent content generation system

→ [Go to Module 3.6](./module-3.6-ai-agents/)

---

## 🔗 Domain Connections

### Prerequisites (Domains 1-2)
- **Domain 1**: Hardware understanding (GPU memory, CUDA)
- **Domain 2**: ML fundamentals (transformers, training)

### What This Enables (Domain 4+)
- **Domain 4**: Production deployment, monitoring, scaling
- **Domain 5**: Real-world applications and demos

---

## 🛤️ Recommended Learning Paths

### Standard Path (Complete Coverage)
Follow modules in order: 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6

Best for: Building comprehensive LLM engineering skills

### Deployment-First Path
3.3 (Deployment) → 3.5 (RAG) → 3.6 (Agents) → 3.2 → 3.1

Best for: Quickly deploying applications, fine-tuning later

### Research-Focused Path
3.1 (Fine-tuning) → 3.4 (Reasoning) → 3.2 (Quantization)

Best for: Model development and optimization research

---

## 💡 Key Concepts Across Domain 3

| Concept | First Introduced | Used Throughout |
|---------|------------------|-----------------|
| LoRA adapters | 3.1 | 3.2, 3.3 |
| Quantization | 3.2 | 3.3, 3.5, 3.6 |
| Inference servers | 3.3 | 3.5, 3.6 |
| Embeddings | 3.5 | 3.6 |
| Tool calling | 3.6 | Applications |

---

## 🎯 Domain Learning Outcomes

By completing Domain 3, you will be able to:

1. **Fine-tune** LLMs for custom tasks using parameter-efficient methods
2. **Quantize** models to fit on DGX Spark while maintaining quality
3. **Deploy** models with production-grade inference engines
4. **Enhance** model reasoning with test-time compute techniques
5. **Build** RAG systems that ground responses in your documents
6. **Create** AI agents that use tools and work autonomously

---

## 🛠️ DGX Spark Capabilities Used

| Capability | Modules |
|------------|---------|
| 128GB Unified Memory | All modules |
| Blackwell GPU | 3.1, 3.2, 3.3 |
| NVFP4/FP8 Support | 3.2 |
| CUDA Acceleration | 3.3, 3.5 |
| Local LLM Serving | 3.3, 3.5, 3.6 |

---

## 📚 Study Materials by Module

Each module includes these documentation files:

| Document | Purpose |
|----------|---------|
| QUICKSTART.md | 5-minute working demo |
| ELI5.md | Plain-language explanations |
| PREREQUISITES.md | Skills check before starting |
| STUDY_GUIDE.md | Learning path and objectives |
| QUICK_REFERENCE.md | Commands and patterns |
| LAB_PREP.md | Environment setup |
| TROUBLESHOOTING.md | Error solutions and FAQs |

---

## ▶️ Getting Started

1. **Verify Prerequisites**: Complete Domains 1-2
2. **Choose Your Path**: Standard, Deployment-First, or Research-Focused
3. **Start First Module**: Begin with Module 3.1 or your chosen entry point
4. **Track Progress**: Use milestone checklists in each module

Ready to begin? Start with [Module 3.1: LLM Fine-Tuning](./module-3.1-llm-finetuning/)
