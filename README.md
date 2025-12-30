# 🚀 DGX Spark AI Curriculum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DGX Spark](https://img.shields.io/badge/NVIDIA-DGX%20Spark-76B900?logo=nvidia)](https://www.nvidia.com/dgx-spark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive **24-week AI/ML curriculum** specifically optimized for the **NVIDIA DGX Spark** (Grace Blackwell GB10 Superchip with 128GB unified memory).

From neural network fundamentals to fine-tuning 70B parameter models, this curriculum leverages the unique capabilities of the DGX Spark platform.

## 🎯 What You'll Learn

| Domain | Duration | Topics |
|--------|----------|--------|
| **Platform Foundations** | Weeks 1-6 | DGX Spark platform, Python for AI, Math for DL, Neural Networks, MicroGrad+ |
| **Deep Learning Frameworks** | Weeks 7-14 | PyTorch, Computer Vision, NLP, Transformers, Hugging Face |
| **LLM Systems** | Weeks 15-22 | LLM Fine-tuning (70B QLoRA!), Quantization (FP4!), Deployment, AI Agents |
| **Production AI** | Weeks 23-32 | Multimodal AI, Benchmarking & MLOps, Capstone Project |

## 🔥 DGX Spark Advantages

This curriculum specifically leverages DGX Spark's unique capabilities:

- **128GB Unified Memory**: Fine-tune 70B models with QLoRA locally
- **Blackwell FP4**: Exclusive NVFP4 quantization for 3.5× memory reduction
- **10,000+ tok/s Prefill**: TensorRT-LLM optimized inference
- **Desktop Form Factor**: All of this on your desk!

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
<summary><b>Domain 1: Platform Foundations (Weeks 1-6)</b></summary>

- [Module 1.1: DGX Spark Platform Mastery](domain-1-platform-foundations/module-1.1-dgx-spark-platform/)
- [Module 1.2: Python for AI/ML](domain-1-platform-foundations/module-1.2-python-for-ai/)
- [Module 1.3: Mathematics for Deep Learning](domain-1-platform-foundations/module-1.3-math-foundations/)
- [Module 1.4: Neural Network Fundamentals](domain-1-platform-foundations/module-1.4-neural-networks/)
- [Module 1.5: Capstone — MicroGrad+](domain-1-platform-foundations/module-1.5-capstone-micrograd/)

</details>

<details>
<summary><b>Domain 2: Deep Learning Frameworks (Weeks 7-14)</b></summary>

- [Module 2.1: Deep Learning with PyTorch](domain-2-deep-learning-frameworks/module-2.1-pytorch/)
- [Module 2.2: Computer Vision](domain-2-deep-learning-frameworks/module-2.2-computer-vision/)
- [Module 2.3: NLP & Transformers](domain-2-deep-learning-frameworks/module-2.3-nlp-transformers/)
- [Module 2.4: Hugging Face Ecosystem](domain-2-deep-learning-frameworks/module-2.4-huggingface/)

</details>

<details>
<summary><b>Domain 3: LLM Systems (Weeks 15-22)</b></summary>

- [Module 3.1: LLM Fine-Tuning](domain-3-llm-systems/module-3.1-llm-finetuning/) ⭐ 70B QLoRA!
- [Module 3.2: Quantization & Optimization](domain-3-llm-systems/module-3.2-quantization/) ⭐ FP4!
- [Module 3.3: Deployment & Inference](domain-3-llm-systems/module-3.3-deployment/)
- [Module 3.4: AI Agents](domain-3-llm-systems/module-3.4-ai-agents/)

</details>

<details>
<summary><b>Domain 4: Production AI (Weeks 23-32)</b></summary>

- [Module 4.1: Multimodal AI](domain-4-production-ai/module-4.1-multimodal/)
- [Module 4.2: Benchmarking & MLOps](domain-4-production-ai/module-4.2-mlops/)
- [Module 4.3: Capstone Project](domain-4-production-ai/module-4.3-capstone-project/)

</details>

## 📁 Repository Structure

```
dgx-spark-ai-curriculum/
│
├── README.md                    # Main repo README with badges, overview
├── CURRICULUM.md                # Full curriculum documentation
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
├── utils/
│   ├── __init__.py             # Package init with exports
│   ├── dgx_spark_utils.py      # System info, environment check
│   ├── memory_utils.py         # Memory tracking, estimation
│   └── benchmark_utils.py      # Direct API benchmarking
│
├── templates/
│   ├── notebook_template.ipynb
│   ├── module_readme_template.md
│   ├── project-proposal.md
│   ├── technical-report.md
│   └── presentation-outline.md
│
├── domain-1-platform-foundations/
│   ├── module-1.1-dgx-spark-platform/
│   ├── module-1.2-python-for-ai/
│   ├── module-1.3-math-foundations/
│   ├── module-1.4-neural-networks/
│   └── module-1.5-capstone-micrograd/
│
├── domain-2-deep-learning-frameworks/
│   ├── module-2.1-pytorch/
│   ├── module-2.2-computer-vision/
│   ├── module-2.3-nlp-transformers/
│   └── module-2.4-huggingface/
│
├── domain-3-llm-systems/
│   ├── module-3.1-llm-finetuning/         ⭐ 70B QLoRA
│   ├── module-3.2-quantization/           ⭐ NVFP4
│   ├── module-3.3-deployment/
│   └── module-3.4-ai-agents/
│
└── domain-4-production-ai/
    ├── module-4.1-multimodal/
    ├── module-4.2-mlops/
    └── module-4.3-capstone-project/
```

## 📊 Progress Tracking

Track your progress using the [Progress Tracker](CURRICULUM.md#appendix-c-progress-tracking) in the curriculum document.

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