# Domain 4: Production AI - Overview

## 🎯 Domain Learning Objectives

By completing this domain, you will:

1. **Build multimodal AI systems** that understand images, generate content, and process audio
2. **Implement comprehensive AI safety measures** including guardrails, red teaming, and bias evaluation
3. **Master MLOps practices** for experiment tracking, benchmarking, and reproducibility
4. **Deploy AI to production** using Docker, Kubernetes, and cloud platforms
5. **Create polished demos** that showcase your AI work effectively
6. **Complete a capstone project** synthesizing all curriculum learnings

---

## 🗺️ Domain Roadmap

```
Week 27        Weeks 28-29      Weeks 30-31      Weeks 32-33      Week 34        Weeks 35-40
   │               │                │                │               │               │
   ▼               ▼                ▼                ▼               ▼               ▼
┌──────┐       ┌──────┐          ┌──────┐        ┌──────┐       ┌──────┐       ┌──────┐
│ 4.1  │──────►│ 4.2  │─────────►│ 4.3  │───────►│ 4.4  │──────►│ 4.5  │──────►│ 4.6  │
└──────┘       └──────┘          └──────┘        └──────┘       └──────┘       └──────┘
Multimodal     AI Safety        MLOps &         Container &    Demo           Capstone
AI             & Alignment      Evaluation      Deployment     Building       Project
```

---

## 📚 Module Summaries

### Module 4.1: Multimodal AI
**Focus**: Vision-language models, image generation, audio processing, multimodal pipelines
**Key Deliverables**: VLM demo, image generation pipeline, multimodal RAG, document AI
**Time**: ~8-10 hours (Week 27)
**DGX Spark Advantage**: Run LLaVA-13B, Qwen2-VL-72B (4-bit), SDXL, and Whisper simultaneously

### Module 4.2: AI Safety & Alignment
**Focus**: Guardrails, red teaming, safety benchmarks, responsible AI practices
**Key Deliverables**: NeMo Guardrails chatbot, Llama Guard integration, red team report, model card
**Time**: ~12-15 hours (Weeks 28-29)
**Priority**: P0 Critical - Industry compliance requirement

### Module 4.3: MLOps & Experiment Tracking
**Focus**: MLflow, W&B, LLM benchmarks, drift detection, reproducibility
**Key Deliverables**: Tracking server, benchmark results, evaluation framework, model registry
**Time**: ~12-15 hours (Weeks 30-31)
**Priority**: P0/P1 - Foundation for production ML

### Module 4.4: Containerization & Cloud Deployment
**Focus**: Docker, Kubernetes, AWS SageMaker, GCP Vertex AI, Gradio/Streamlit
**Key Deliverables**: Optimized Docker images, K8s manifests, cloud endpoints, deployed demos
**Time**: ~12-15 hours (Weeks 32-33)
**Priority**: P0/P1 - Docker critical, cloud high priority

### Module 4.5: Demo Building & Prototyping
**Focus**: Advanced Gradio, multi-page Streamlit, rapid prototyping, deployment
**Key Deliverables**: RAG demo, agent playground, portfolio demo, video walkthrough
**Time**: ~6-8 hours (Week 34)
**Priority**: P2 - Portfolio building

### Module 4.6: Capstone Project
**Focus**: End-to-end AI system integrating all curriculum learnings
**Key Deliverables**: Technical report, code repository, demo, presentation, model card
**Time**: ~40-50 hours (Weeks 35-40)
**Priority**: Synthesis of entire curriculum

---

## 🔗 Prerequisites from Previous Domains

| This Domain Needs | From Domain | Specifically |
|-------------------|-------------|--------------|
| LLM fine-tuning | Domain 3 | Module 3.1: LoRA, QLoRA techniques |
| Quantization | Domain 3 | Module 3.2: FP8, INT4, NVFP4 formats |
| RAG systems | Domain 3 | Module 3.5: Vector DBs, retrieval |
| AI Agents | Domain 3 | Module 3.6: LangChain, tool calling |
| PyTorch proficiency | Domain 2 | Module 2.1: Model loading, inference |
| Transformers | Domain 2 | Module 2.3: Attention, tokenization |

---

## 🎯 What This Domain Prepares You For

| Career Path | Relevant Skills from This Domain |
|-------------|----------------------------------|
| **ML Engineer** | Docker, K8s, MLOps, deployment pipelines |
| **AI Safety Engineer** | Guardrails, red teaming, benchmarks, model cards |
| **Applied AI Scientist** | Multimodal systems, evaluation, capstone |
| **AI Product Engineer** | Demo building, prototyping, stakeholder communication |
| **MLOps Engineer** | Experiment tracking, versioning, reproducibility |

---

## ✅ Domain Completion Checklist

### Module Completion
- [ ] 4.1: Multimodal AI - All 5 labs complete
- [ ] 4.2: AI Safety - All 6 labs complete, model card created
- [ ] 4.3: MLOps - Tracking server running, benchmarks documented
- [ ] 4.4: Containerization - Docker + cloud deployment working
- [ ] 4.5: Demo Building - At least one public demo deployed
- [ ] 4.6: Capstone - All deliverables submitted

### Key Artifacts
- [ ] Working multimodal pipeline (4.1)
- [ ] Safety evaluation report with red teaming results (4.2)
- [ ] MLflow/W&B experiment tracking setup (4.3)
- [ ] Docker Compose stack for ML (4.4)
- [ ] Public demo on Hugging Face Spaces (4.5)
- [ ] Complete capstone with video demo (4.6)

### Self-Assessment
- [ ] Can explain multimodal AI architectures
- [ ] Can implement guardrails and perform red teaming
- [ ] Can set up reproducible ML experiments
- [ ] Can containerize and deploy ML models
- [ ] Can create polished demos for stakeholders
- [ ] Ready to build production AI systems

---

## 📖 Recommended Study Order

**Standard Path** (most students):
1. 4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
2. Follow the weekly schedule in each module
3. Complete all labs before moving on

**Accelerated Path** (strong ML background):
1. Skim 4.1 (multimodal may be new)
2. 4.2 (safety is critical, don't skip)
3. 4.3 (focus on MLflow setup, skip W&B if familiar)
4. 4.4 (Docker critical, cloud optional if familiar)
5. 4.5 (quick pass - choose one demo framework)
6. 4.6 (full capstone)

**Deep-Dive Path** (extra time available):
1. 4.1 + experiment with all VLM options
2. 4.2 + implement all OWASP LLM Top 10 mitigations
3. 4.3 + set up full CI/CD pipeline
4. 4.4 + deploy to both AWS and GCP
5. 4.5 + build demos for each framework
6. 4.6 + attempt stretch goals

---

## 🏆 Domain Project Ideas

After completing this domain, try:

1. **Enterprise Document Assistant**
   - Multimodal RAG over company documents (PDFs with tables/charts)
   - Safety guardrails for PII and confidential information
   - Deployed on internal Kubernetes cluster
   - MLflow tracking for continuous improvement

2. **AI Code Review Agent**
   - Multi-agent system for code analysis
   - Red-teamed for prompt injection via code
   - Integrated with GitHub Actions
   - Gradio interface for ad-hoc reviews

3. **Personalized Learning Tutor**
   - Multimodal input (student work photos)
   - Safety filters for age-appropriate content
   - Progress tracking with W&B
   - Mobile-friendly Streamlit demo

---

## 📊 Time Investment Summary

| Module | Hours | Priority | Key Skills |
|--------|-------|----------|------------|
| 4.1 Multimodal | 8-10 | Medium | VLMs, diffusion, audio |
| 4.2 AI Safety | 12-15 | **Critical** | Guardrails, red teaming |
| 4.3 MLOps | 12-15 | High | Tracking, benchmarks |
| 4.4 Containerization | 12-15 | **Critical** | Docker, K8s, cloud |
| 4.5 Demo Building | 6-8 | Medium | Gradio, Streamlit |
| 4.6 Capstone | 40-50 | **Critical** | Integration |
| **Total** | **90-113** | - | Production AI |

---

## 🔗 Quick Links

| Resource | Description |
|----------|-------------|
| [GLOSSARY.md](./GLOSSARY.md) | Domain 4 terminology |
| [Module 4.1](./module-4.1-multimodal/) | Multimodal AI |
| [Module 4.2](./module-4.2-ai-safety/) | AI Safety |
| [Module 4.3](./module-4.3-mlops/) | MLOps |
| [Module 4.4](./module-4.4-containerization-deployment/) | Containerization |
| [Module 4.5](./module-4.5-demo-building/) | Demo Building |
| [Module 4.6](./module-4.6-capstone-project/) | Capstone |

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Domain 3: LLM Systems](../domain-3-llm-systems/) | **Domain 4: Production AI** | Curriculum Complete! |
