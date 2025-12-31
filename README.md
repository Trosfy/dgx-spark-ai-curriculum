# 🚀 DGX Spark AI Curriculum v2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DGX Spark](https://img.shields.io/badge/NVIDIA-DGX%20Spark-76B900?logo=nvidia)](https://www.nvidia.com/dgx-spark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive **32-40 week AI/ML curriculum** specifically optimized for the **NVIDIA DGX Spark** (Grace Blackwell GB10 Superchip with 128GB unified memory).

**25 core modules + 5 optional** covering neural network fundamentals to fine-tuning 100B+ parameter models. Leverages DGX Spark's unique capabilities: **NVFP4 quantization**, **Mamba/MoE architectures**, **production RAG systems**, and **AI safety guardrails**.

## 🎯 What You'll Learn

| Domain | Duration | Topics |
|--------|----------|--------|
| **Platform Foundations** | Weeks 1-7 | DGX Spark, Python, **CUDA Python** ⭐, Math, Neural Networks, **Classical ML** ⭐, MicroGrad+ |
| **Deep Learning Frameworks** | Weeks 8-15 | PyTorch, CV (ViT, YOLO), NLP & Transformers, **Mamba/MoE** ⭐, Hugging Face, **Diffusion** ⭐ |
| **LLM Systems** | Weeks 16-26 | Fine-tuning (DoRA, SimPO, ORPO), **NVFP4 Quantization** ⭐, SGLang/Medusa, **RAG Systems** ⭐, Agents |
| **Production AI** | Weeks 27-40 | Multimodal, **AI Safety** ⭐, MLOps, **Docker/K8s/Cloud** ⭐, Demos, Capstone |
| **Optional Modules** | Self-paced | Learning Theory, Recommender Systems, Interpretability, RL, GNNs |

⭐ = New in v2.0

## 🔥 DGX Spark Advantages

This curriculum specifically leverages DGX Spark's unique capabilities:

| Capability | What It Enables |
|------------|-----------------|
| **128GB Unified Memory** | Fine-tune 100B+ models with QLoRA locally |
| **NVFP4 Quantization** | Run ~200B parameter models (Blackwell exclusive!) |
| **FP8 Native** | 90-100B inference without quantization loss |
| **10,000+ tok/s Prefill** | TensorRT-LLM optimized inference |
| **RAPIDS/cuML** | GPU-accelerated classical ML and data processing |
| **Desktop Form Factor** | All of this on your desk!

## 📋 Prerequisites

- Basic Python programming knowledge
- High school level mathematics
- NVIDIA DGX Spark (or adapt for other hardware)
- Enthusiasm to learn! 🎉

## 🚀 Quick Start

1. **Clone the repository**
```bash
   git clone https://github.com/Trosfy/dgx-spark-ai-curriculum.git
   cd dgx-spark-ai-curriculum
```

2. **Set up your environment**
```bash
   # See docs/SETUP.md for complete instructions
   docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

3. **Start with Module 1**
```bash
   cd domain-1-platform-foundations/module-1.1-dgx-spark-platform
   jupyter lab
```

## 📚 Curriculum Structure

<details>
<summary><b>Domain 1: Platform Foundations (Weeks 1-7)</b></summary>

- [Module 1.1: DGX Spark Platform Mastery](domain-1-platform-foundations/module-1.1-dgx-spark-platform/)
- [Module 1.2: Python for AI/ML](domain-1-platform-foundations/module-1.2-python-for-ai/)
- [Module 1.3: CUDA Python & GPU Programming](domain-1-platform-foundations/module-1.3-cuda-python/) ⭐ **NEW [P0]**
- [Module 1.4: Mathematics for Deep Learning](domain-1-platform-foundations/module-1.4-math-foundations/)
- [Module 1.5: Neural Network Fundamentals](domain-1-platform-foundations/module-1.5-neural-networks/)
- [Module 1.6: Classical ML Foundations](domain-1-platform-foundations/module-1.6-classical-ml/) ⭐ **NEW [P2]**
- [Module 1.7: Capstone — MicroGrad+](domain-1-platform-foundations/module-1.7-capstone-micrograd/)

</details>

<details>
<summary><b>Domain 2: Deep Learning Frameworks (Weeks 8-15)</b></summary>

- [Module 2.1: Deep Learning with PyTorch](domain-2-deep-learning-frameworks/module-2.1-pytorch/)
- [Module 2.2: Computer Vision](domain-2-deep-learning-frameworks/module-2.2-computer-vision/) (ViT, YOLO) **[P2 Expanded]**
- [Module 2.3: NLP & Transformers](domain-2-deep-learning-frameworks/module-2.3-nlp-transformers/) (Tokenizer Training) **[P2 Expanded]**
- [Module 2.4: Efficient Architectures](domain-2-deep-learning-frameworks/module-2.4-efficient-architectures/) ⭐ **NEW [P1]** (Mamba, MoE)
- [Module 2.5: Hugging Face Ecosystem](domain-2-deep-learning-frameworks/module-2.5-huggingface/)
- [Module 2.6: Diffusion Models](domain-2-deep-learning-frameworks/module-2.6-diffusion-models/) ⭐ **NEW [P1]**

</details>

<details>
<summary><b>Domain 3: LLM Systems (Weeks 16-26)</b></summary>

- [Module 3.1: LLM Fine-Tuning](domain-3-llm-systems/module-3.1-llm-finetuning/) ⭐ 100B QLoRA! (DoRA, NEFTune, SimPO, ORPO) **[P1 Expanded]**
- [Module 3.2: Quantization & Optimization](domain-3-llm-systems/module-3.2-quantization/) ⭐ NVFP4/FP8! **[P0 Expanded]**
- [Module 3.3: Deployment & Inference](domain-3-llm-systems/module-3.3-deployment/) (SGLang, Medusa) **[P1 Expanded]**
- [Module 3.4: Test-Time Compute & Reasoning](domain-3-llm-systems/module-3.4-test-time-compute/) ⭐ **NEW [P1]**
- [Module 3.5: RAG Systems & Vector Databases](domain-3-llm-systems/module-3.5-rag-systems/) ⭐ **NEW [P0]**
- [Module 3.6: AI Agents & Agentic Systems](domain-3-llm-systems/module-3.6-ai-agents/)

</details>

<details>
<summary><b>Domain 4: Production AI (Weeks 27-40)</b></summary>

- [Module 4.1: Multimodal AI](domain-4-production-ai/module-4.1-multimodal/)
- [Module 4.2: AI Safety & Alignment](domain-4-production-ai/module-4.2-ai-safety/) ⭐ **NEW [P0]**
- [Module 4.3: MLOps & Experiment Tracking](domain-4-production-ai/module-4.3-mlops/) **[P0/P1 Expanded]**
- [Module 4.4: Containerization & Deployment](domain-4-production-ai/module-4.4-containerization-deployment/) ⭐ **NEW [P0/P1]** (Docker, K8s, Cloud)
- [Module 4.5: Demo Building & Prototyping](domain-4-production-ai/module-4.5-demo-building/) ⭐ **NEW [P2]**
- [Module 4.6: Capstone Project](domain-4-production-ai/module-4.6-capstone-project/)

</details>

<details>
<summary><b>Optional Modules [P3] (Self-paced)</b></summary>

- [Optional A: Learning Theory Deep Dive](optional-modules/optional-a-learning-theory/)
- [Optional B: Recommender Systems](optional-modules/optional-b-recommender-systems/)
- [Optional C: Mechanistic Interpretability](optional-modules/optional-c-mechanistic-interpretability/)
- [Optional D: Reinforcement Learning Fundamentals](optional-modules/optional-d-reinforcement-learning/)
- [Optional E: Graph Neural Networks](optional-modules/optional-e-graph-neural-networks/)

</details>

## 📁 Repository Structure

```
dgx-spark-ai-curriculum/
│
├── README.md                    # Main repo README (you are here)
├── CURRICULUM_V2.md             # Full v2 curriculum documentation
├── content-prompt.md            # Content generator prompt
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
├── .gitignore                   # Comprehensive gitignore
├── requirements.txt             # Python dependencies
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── question.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── workflows/
│   │   └── validate_notebooks.yml
│   └── mlc_config.json
│
├── docs/
│   ├── SETUP.md                # Environment setup guide
│   ├── NGC_CONTAINERS.md       # NGC container deep-dive
│   ├── TROUBLESHOOTING.md      # Common issues & solutions
│   └── RESOURCES.md            # External learning resources
│
├── review/
│   ├── PROMPT.md               # Content review prompt
│   └── gather_module_for_review.py
│
├── utils/
│   ├── __init__.py             # Package init with exports
│   ├── dgx_spark_utils.py      # System info, environment check
│   ├── memory_utils.py         # Memory tracking, estimation
│   └── benchmark_utils.py      # Direct API benchmarking
│
├── templates/
│   ├── notebook_template.ipynb      # Template for creating labs
│   └── module_readme_template.md    # Template for module READMEs
│
├── domain-1-platform-foundations/       # Weeks 1-7
│   ├── module-1.1-dgx-spark-platform/
│   ├── module-1.2-python-for-ai/
│   ├── module-1.3-cuda-python/          ⭐ NEW [P0]
│   ├── module-1.4-math-foundations/
│   ├── module-1.5-neural-networks/
│   ├── module-1.6-classical-ml/         ⭐ NEW [P2]
│   └── module-1.7-capstone-micrograd/
│
├── domain-2-deep-learning-frameworks/   # Weeks 8-15
│   ├── module-2.1-pytorch/
│   ├── module-2.2-computer-vision/      [P2 Expanded: ViT, YOLO]
│   ├── module-2.3-nlp-transformers/     [P2 Expanded: Tokenizer Training]
│   ├── module-2.4-efficient-architectures/  ⭐ NEW [P1] Mamba, MoE
│   ├── module-2.5-huggingface/
│   └── module-2.6-diffusion-models/     ⭐ NEW [P1]
│
├── domain-3-llm-systems/                # Weeks 16-26
│   ├── module-3.1-llm-finetuning/       [P1 Expanded: DoRA, SimPO, ORPO]
│   ├── module-3.2-quantization/         [P0 Expanded: NVFP4, FP8]
│   ├── module-3.3-deployment/           [P1 Expanded: SGLang, Medusa]
│   ├── module-3.4-test-time-compute/    ⭐ NEW [P1]
│   ├── module-3.5-rag-systems/          ⭐ NEW [P0]
│   └── module-3.6-ai-agents/
│
├── domain-4-production-ai/              # Weeks 27-40
│   ├── module-4.1-multimodal/
│   ├── module-4.2-ai-safety/            ⭐ NEW [P0]
│   ├── module-4.3-mlops/                [P0/P1 Expanded]
│   ├── module-4.4-containerization-deployment/  ⭐ NEW [P0/P1]
│   ├── module-4.5-demo-building/        ⭐ NEW [P2]
│   └── module-4.6-capstone-project/
│
└── optional-modules/                    # Self-paced [P3]
    ├── optional-a-learning-theory/
    ├── optional-b-recommender-systems/
    ├── optional-c-mechanistic-interpretability/
    ├── optional-d-reinforcement-learning/
    └── optional-e-graph-neural-networks/
```

## 📊 Progress Tracking

Track your progress using the [Progress Tracker](CURRICULUM_V2.md#appendix-d-progress-tracking) in the curriculum document.

## 🆕 What's New in v2.0

Based on comprehensive [curriculum research](docs/curriculum/curriculum-research.md), v2.0 adds:

**P0 Critical Additions:**
- CUDA Python & GPU Programming (2 weeks)
- NVFP4/FP8 Quantization Workflows (expanded)
- RAG Systems & Vector Databases (2 weeks)
- AI Safety & Alignment (2 weeks)
- Docker & Containerization (1.5 weeks)

**P1 High Priority:**
- Mamba/State Space Models, MoE Architectures
- Modern Fine-Tuning (DoRA, NEFTune, SimPO, ORPO)
- SGLang, Speculative Decoding (Medusa/EAGLE)
- Test-Time Compute & Reasoning (DeepSeek-R1)
- Diffusion Models (SDXL, Flux)

**P2 Medium Priority:**
- Classical ML (XGBoost, RAPIDS cuML)
- Object Detection (YOLO, Faster R-CNN), Vision Transformers
- Tokenizer Training from Scratch
- Kubernetes Basics, Gradio/Streamlit

**P3 Optional:**
- Learning Theory, Recommender Systems, Mechanistic Interpretability, RL, GNNs

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- 🐛 Found a bug? [Open an issue](https://github.com/Trosfy/dgx-spark-ai-curriculum/issues)
- 💡 Have an idea? [Start a discussion](https://github.com/Trosfy/dgx-spark-ai-curriculum/discussions)
- 📝 Want to contribute? [Submit a PR](https://github.com/Trosfy/dgx-spark-ai-curriculum/pulls)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- NVIDIA for the DGX Spark platform
- Hugging Face for their incredible ecosystem
- Fast.ai for teaching philosophy inspiration
- The open-source AI community

---

<p align="center">
  <b>Built for the DGX Spark community 🚀</b><br>
  <a href="https://github.com/Trosfy/dgx-spark-ai-curriculum">Star this repo</a> if you find it helpful!
</p>