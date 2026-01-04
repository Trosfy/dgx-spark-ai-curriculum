# Module 4.6: Capstone Project

**Domain:** 4 - Production AI
**Duration:** Weeks 35-40 (40-50 hours)
**Prerequisites:** All previous modules (Domains 1-4)

---

## Overview

Congratulations on reaching the capstone! This is your opportunity to synthesize everything you've learned into a substantial, end-to-end AI project. Choose a project that interests you and demonstrates mastery of the DGX Spark platform.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Design and implement an end-to-end AI system
- ✅ Apply all techniques learned throughout the curriculum
- ✅ Document and present your work professionally
- ✅ Deploy a production-ready AI application

---

## Project Options

Choose ONE of the following:

### Option A: Domain-Specific AI Assistant

Build a complete AI assistant for a specific domain with safety guardrails.

**Requirements:**
- Fine-tuned base model (70B with QLoRA)
- RAG system with domain knowledge base
- Custom tools for domain operations
- **NeMo Guardrails for safety**
- Production API with streaming
- Comprehensive evaluation (benchmark + safety)
- **Gradio demo**
- Complete documentation with model card

**Example domains:** AWS infrastructure, trading analysis, code review, medical literature, legal documents

---

### Option B: Multimodal Document Intelligence

Build a system that processes and understands complex documents.

**Requirements:**
- Document ingestion pipeline (PDF, images, diagrams)
- Vision-language model integration
- Structured information extraction
- Multimodal RAG
- Question answering over documents
- Export to structured formats
- Evaluation on document benchmarks
- **Interactive demo**

---

### Option C: AI Agent Swarm with Safety

Build a multi-agent system with comprehensive safety measures.

**Requirements:**
- Minimum 4 specialized agents
- Agent communication protocol
- Task decomposition and planning
- Human-in-the-loop approval
- **Safety guardrails on agent actions**
- Error recovery and fallbacks
- Performance benchmarking
- **Red teaming and safety evaluation**

---

### Option D: Custom Training Pipeline

Build infrastructure for continuous model improvement.

**Requirements:**
- Data collection and curation pipeline
- Multiple fine-tuning approaches (SFT, DPO)
- Automated evaluation
- Model versioning and comparison
- A/B testing framework
- Deployment automation

---

### Option E: Browser-Deployed Fine-Tuned LLM (NEW)

Build a complete pipeline that fine-tunes a small language model for a specific domain, optimizes it for browser deployment, and creates a zero-cost static web application.

**Why This Capstone Is Unique:**
- Demonstrates the FULL ML lifecycle: data → training → optimization → deployment
- Zero ongoing costs (no GPU servers, no API fees)
- Privacy-preserving (all inference on user's device)
- Showcases DGX Spark for training + edge deployment
- Applies skills from Modules 3.1 (Fine-tuning), 3.2 (Quantization), 4.4 (Deployment)

**Requirements:**
- **Dataset Creation** (50-200 high-quality examples)
  - Messages format with system/user/assistant roles
  - Domain-focused (Matcha expertise for this project)
- **QLoRA Fine-Tuning on DGX Spark**
  - Base model: Gemma 3 270M (~540MB)
  - Training with Unsloth for 2x speed
  - MLflow experiment tracking
- **Model Optimization Pipeline**
  - Merge LoRA adapters in BF16 precision
  - Export to ONNX format
  - INT4 block quantization (browser-compatible)
- **Browser Integration**
  - Transformers.js for inference
  - WebGPU acceleration with WASM fallback
  - React component with streaming
- **Static Deployment**
  - AWS S3 + CloudFront for model files (CDN)
  - Any static host for web app (Vercel/Netlify/etc)
  - Complete model card and documentation

**Deliverables:**
1. Fine-tuned model files (GGUF + ONNX INT4)
2. Training code with MLflow logs
3. React web application
4. Technical report
5. Model card with safety evaluation
6. Video demo

---

## Project Phases

| Phase | Duration | Activities |
|-------|----------|------------|
| **Planning** | Week 35 | Requirements, architecture, tech selection |
| **Foundation** | Weeks 36-37 | Core components, data prep, initial models |
| **Integration** | Week 38 | Component integration, API development |
| **Optimization** | Week 39 | Performance tuning, quantization, safety evaluation |
| **Documentation** | Week 40 | Docs, demo, presentation, model card |

---

## Deliverables

### 1. Technical Report (15-20 pages)
- Problem statement and motivation
- System architecture
- Implementation details
- Evaluation results (performance + safety)
- Lessons learned

### 2. Code Repository
- Well-organized, documented code
- README with setup instructions
- Requirements and environment files
- Unit tests
- Dockerfile for deployment

### 3. Demo (Gradio/Streamlit + Video)
- Working interactive demonstration
- Video walkthrough (5-10 minutes)
- Deployed to Hugging Face Spaces or similar

### 4. Presentation
- Slide deck (15-20 slides)
- Technical deep-dives
- Results and impact

### 5. Model Card (with Safety Evaluation)
- Model description and intended use
- Training data documentation
- Safety evaluation results
- Limitations and biases

---

## Grading Rubric (Self-Assessment)

| Criteria | Points |
|----------|--------|
| Technical complexity and correctness | 20 |
| Effective use of DGX Spark capabilities | 15 |
| **Safety implementation and evaluation** | 15 |
| Code quality and organization | 15 |
| Documentation and presentation | 15 |
| Evaluation and benchmarking | 10 |
| Innovation and creativity | 10 |
| **Total** | **100** |

---

## Milestone Checklist

- [ ] Project proposal approved
- [ ] Architecture design documented
- [ ] Core components implemented
- [ ] Integration complete
- [ ] **Safety evaluation completed**
- [ ] Evaluation results collected
- [ ] Technical report written
- [ ] **Model card created**
- [ ] Demo video created
- [ ] Presentation ready
- [ ] Code repository finalized

---

## Templates

Use these templates in the `templates/` folder:
- `project-proposal.md` - Initial proposal format
- `technical-report.md` - Report structure
- `presentation-outline.md` - Slide organization

---

## Tips for Success

1. **Start with a clear scope** - Better to complete a focused project than abandon an ambitious one

2. **Leverage DGX Spark strengths** - Show what's possible with 128GB unified memory and Blackwell FP4

3. **Document as you go** - Don't leave documentation for the end

4. **Test early and often** - Build evaluation into your development process

5. **Ask for feedback** - Share progress and get input

---

## 🎉 Congratulations!

Upon completing the capstone, you will have:

- ✅ Deep understanding of AI/ML from fundamentals to production
- ✅ Hands-on experience with cutting-edge hardware (DGX Spark)
- ✅ Portfolio project demonstrating advanced skills
- ✅ Expertise in LLM fine-tuning, quantization, and deployment
- ✅ Experience building AI agents and multimodal systems

**You are now AI-ready!** 🚀

---

## What's Next?

- Contribute your capstone to the curriculum examples
- Share your learnings with the community
- Continue exploring new AI developments
- Consider NVIDIA Deep Learning Institute certifications
- Join AI/ML communities and keep learning!

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 4.5: Demo Building](../module-4.5-demo-building/) | **Module 4.6: Capstone Project** | Curriculum Complete! |

---

## Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Validate your project idea with a quick MVP |
| [PREREQUISITES.md](./PREREQUISITES.md) | Self-assessment for required skills from all domains |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Week-by-week plan and deliverables checklist |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Code patterns for all four project options |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and resource planning |
| [WORKFLOWS.md](./WORKFLOWS.md) | Step-by-step workflows for planning, development, and documentation |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors, solutions, and FAQs for capstone development |

---

## Resources

- [DGX Spark Playbooks](https://build.nvidia.com/spark)
- [Hugging Face Hub](https://huggingface.co/)
- [Papers With Code](https://paperswithcode.com/)
- [AI Community Discord servers](../../docs/RESOURCES.md)
