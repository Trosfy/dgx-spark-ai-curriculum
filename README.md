# 🚀 DGX Spark AI Curriculum

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DGX Spark](https://img.shields.io/badge/NVIDIA-DGX%20Spark-76B900?logo=nvidia)](https://www.nvidia.com/dgx-spark)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A comprehensive **24-week AI/ML curriculum** specifically optimized for the **NVIDIA DGX Spark** (Grace Blackwell GB10 Superchip with 128GB unified memory).

From neural network fundamentals to fine-tuning 70B parameter models, this curriculum leverages the unique capabilities of the DGX Spark platform.

![Curriculum Overview](assets/curriculum-overview.png)

## 🎯 What You'll Learn

| Phase | Duration | Topics |
|-------|----------|--------|
| **Foundations** | Weeks 1-6 | DGX Spark platform, Python for AI, Math for DL, Neural Networks |
| **Intermediate** | Weeks 7-14 | PyTorch, Computer Vision, NLP, Transformers, Hugging Face |
| **Advanced** | Weeks 15-26 | LLM Fine-tuning (70B QLoRA!), Quantization (FP4!), Deployment, AI Agents, Multimodal |
| **Capstone** | Weeks 27-32 | End-to-end AI project |

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
   git clone https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum.git
   cd dgx-spark-ai-curriculum
```

2. **Set up your environment**
```bash
   # See docs/SETUP.md for complete instructions
   docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

3. **Start with Module 1**
```bash
   cd phase-1-foundations/module-01-dgx-spark-platform
   jupyter lab
```

## 📚 Curriculum Structure

<details>
<summary><b>Phase 1: Foundations (Weeks 1-6)</b></summary>

- [Module 1: DGX Spark Platform Mastery](phase-1-foundations/module-01-dgx-spark-platform/)
- [Module 2: Python for AI/ML](phase-1-foundations/module-02-python-for-ai/)
- [Module 3: Mathematics for Deep Learning](phase-1-foundations/module-03-math-for-dl/)
- [Module 4: Neural Network Fundamentals](phase-1-foundations/module-04-neural-network-fundamentals/)
- [Module 5: Phase 1 Capstone](phase-1-foundations/module-05-phase1-capstone/)

</details>

<details>
<summary><b>Phase 2: Intermediate (Weeks 7-14)</b></summary>

- [Module 6: Deep Learning with PyTorch](phase-2-intermediate/module-06-pytorch-deep-learning/)
- [Module 7: Computer Vision](phase-2-intermediate/module-07-computer-vision/)
- [Module 8: NLP & Transformers](phase-2-intermediate/module-08-nlp-transformers/)
- [Module 9: Hugging Face Ecosystem](phase-2-intermediate/module-09-huggingface-ecosystem/)

</details>

<details>
<summary><b>Phase 3: Advanced (Weeks 15-26)</b></summary>

- [Module 10: LLM Fine-Tuning](phase-3-advanced/module-10-llm-finetuning/) ⭐ 70B QLoRA!
- [Module 11: Quantization & Optimization](phase-3-advanced/module-11-quantization/) ⭐ FP4!
- [Module 12: Deployment & Inference](phase-3-advanced/module-12-deployment-inference/)
- [Module 13: AI Agents](phase-3-advanced/module-13-ai-agents/)
- [Module 14: Multimodal AI](phase-3-advanced/module-14-multimodal/)
- [Module 15: Benchmarking & MLOps](phase-3-advanced/module-15-benchmarking-mlops/)

</details>

<details>
<summary><b>Phase 4: Capstone (Weeks 27-32)</b></summary>

- [Capstone Project Options](phase-4-capstone/)

</details>

## 📊 Progress Tracking

Track your progress using the [Progress Tracker](CURRICULUM.md#appendix-c-progress-tracking) in the curriculum document.

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- 🐛 Found a bug? [Open an issue](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/issues)
- 💡 Have an idea? [Start a discussion](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/discussions)
- 📝 Want to contribute? [Submit a PR](https://github.com/YOUR_USERNAME/dgx-spark-ai-curriculum/pulls)

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