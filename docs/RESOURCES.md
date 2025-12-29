# External Learning Resources

Curated resources to supplement the DGX Spark AI Curriculum.

---

## Table of Contents

- [Official NVIDIA Resources](#official-nvidia-resources)
- [Foundational Courses](#foundational-courses)
- [Deep Learning Frameworks](#deep-learning-frameworks)
- [LLM & NLP Resources](#llm--nlp-resources)
- [Computer Vision](#computer-vision)
- [AI Agents & RAG](#ai-agents--rag)
- [MLOps & Production](#mlops--production)
- [Research Papers](#research-papers)
- [Books](#books)
- [YouTube Channels](#youtube-channels)
- [Podcasts](#podcasts)
- [Communities](#communities)
- [Tools & Libraries](#tools--libraries)

---

## Official NVIDIA Resources

### DGX Spark Specific

| Resource | Description | Link |
|----------|-------------|------|
| DGX Spark User Guide | Official documentation | [docs.nvidia.com](https://docs.nvidia.com/dgx/dgx-spark/) |
| DGX Spark Playbooks | Ready-to-run AI workflows | [build.nvidia.com/spark](https://build.nvidia.com/spark) |
| NGC Catalog | Container registry | [catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/) |
| DGX Spark Porting Guide | Migration guide | [docs.nvidia.com](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/) |

### NVIDIA AI Courses

| Course | Topics | Link |
|--------|--------|------|
| Deep Learning Institute | Comprehensive AI training | [nvidia.com/dli](https://www.nvidia.com/en-us/training/) |
| Generative AI Explained | GenAI fundamentals | [courses.nvidia.com](https://courses.nvidia.com/) |
| Building RAG Agents | RAG with NVIDIA tools | [nvidia.com/dli](https://www.nvidia.com/en-us/training/) |

---

## Foundational Courses

### Machine Learning Fundamentals

| Course | Provider | Level | Notes |
|--------|----------|-------|-------|
| [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) | DeepLearning.AI / Stanford | Beginner | Andrew Ng's updated ML course |
| [Mathematics for Machine Learning](https://www.coursera.org/specializations/mathematics-machine-learning) | Imperial College London | Beginner | Linear algebra, calculus, PCA |
| [CS229: Machine Learning](https://cs229.stanford.edu/) | Stanford | Intermediate | Full Stanford course materials |

### Deep Learning

| Course | Provider | Level | Notes |
|--------|----------|-------|-------|
| [Practical Deep Learning for Coders](https://course.fast.ai/) | fast.ai | Beginner-Intermediate | **Highly recommended** - Top-down approach |
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | DeepLearning.AI | Intermediate | Comprehensive 5-course series |
| [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) | Stanford | Intermediate | Best for computer vision fundamentals |
| [CS224n: NLP with Deep Learning](http://cs224n.stanford.edu/) | Stanford | Intermediate | Best for NLP/transformers |

### Neural Networks from Scratch

| Resource | Description |
|----------|-------------|
| [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Andrej Karpathy's YouTube series |
| [micrograd](https://github.com/karpathy/micrograd) | Tiny autograd engine implementation |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | GPT training in ~300 lines |

---

## Deep Learning Frameworks

### PyTorch

| Resource | Description |
|----------|-------------|
| [Official PyTorch Tutorials](https://pytorch.org/tutorials/) | Comprehensive tutorial collection |
| [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) | High-level PyTorch wrapper |
| [torchvision](https://pytorch.org/vision/stable/) | Computer vision library |
| [torchaudio](https://pytorch.org/audio/stable/) | Audio processing library |

### Hugging Face

| Resource | Description |
|----------|-------------|
| [HF Course](https://huggingface.co/learn/nlp-course) | Official Transformers course |
| [LLM Course](https://huggingface.co/learn/llm-course) | Complete LLM curriculum |
| [Diffusion Course](https://huggingface.co/learn/diffusion-course) | Image generation |
| [Transformers Documentation](https://huggingface.co/docs/transformers/) | Library reference |
| [PEFT Documentation](https://huggingface.co/docs/peft/) | Parameter-efficient fine-tuning |

---

## LLM & NLP Resources

### LLM Fundamentals

| Resource | Description |
|----------|-------------|
| [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) | Visual explanation of transformers |
| [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) | Visual explanation of GPT |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original transformer paper |
| [LLM Visualization](https://bbycroft.net/llm) | Interactive 3D visualization |

### Fine-Tuning

| Resource | Description |
|----------|-------------|
| [LoRA Paper](https://arxiv.org/abs/2106.09685) | Low-Rank Adaptation |
| [QLoRA Paper](https://arxiv.org/abs/2305.14314) | Quantized LoRA |
| [Unsloth](https://github.com/unslothai/unsloth) | Fast LoRA training |
| [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) | Fine-tuning GUI |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | Fine-tuning framework |

### Quantization

| Resource | Description |
|----------|-------------|
| [GPTQ Paper](https://arxiv.org/abs/2210.17323) | Post-training quantization |
| [AWQ Paper](https://arxiv.org/abs/2306.00978) | Activation-aware quantization |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | GGUF format, CPU/GPU inference |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit/4-bit quantization |

### Alignment & RLHF

| Resource | Description |
|----------|-------------|
| [RLHF Blog](https://huggingface.co/blog/rlhf) | Hugging Face RLHF guide |
| [DPO Paper](https://arxiv.org/abs/2305.18290) | Direct Preference Optimization |
| [TRL Library](https://huggingface.co/docs/trl/) | Transformer Reinforcement Learning |

---

## Computer Vision

### Core Resources

| Resource | Description |
|----------|-------------|
| [CS231n](http://cs231n.stanford.edu/) | Stanford CNN course |
| [timm](https://github.com/huggingface/pytorch-image-models) | PyTorch Image Models |
| [torchvision](https://pytorch.org/vision/) | PyTorch vision library |

### Vision Transformers

| Resource | Description |
|----------|-------------|
| [ViT Paper](https://arxiv.org/abs/2010.11929) | Original Vision Transformer |
| [DeiT Paper](https://arxiv.org/abs/2012.12877) | Data-efficient ViT training |
| [Segment Anything](https://segment-anything.com/) | SAM model and demo |

### Object Detection & Segmentation

| Resource | Description |
|----------|-------------|
| [YOLOv8](https://docs.ultralytics.com/) | Ultralytics detection framework |
| [Detectron2](https://detectron2.readthedocs.io/) | Facebook AI detection library |
| [MMDetection](https://mmdetection.readthedocs.io/) | OpenMMLab detection toolbox |

---

## AI Agents & RAG

### LangChain

| Resource | Description |
|----------|-------------|
| [LangChain Docs](https://python.langchain.com/docs/) | Official documentation |
| [LangChain Hub](https://smith.langchain.com/hub) | Prompt templates |
| [LangGraph Docs](https://langchain-ai.github.io/langgraph/) | Stateful agent workflows |
| [LangSmith](https://smith.langchain.com/) | Tracing and debugging |

### LlamaIndex

| Resource | Description |
|----------|-------------|
| [LlamaIndex Docs](https://docs.llamaindex.ai/) | Official documentation |
| [LlamaIndex Blog](https://www.llamaindex.ai/blog) | Tutorials and guides |

### RAG Resources

| Resource | Description |
|----------|-------------|
| [RAG Paper](https://arxiv.org/abs/2005.11401) | Original RAG paper |
| [Advanced RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) | Comprehensive RAG guide |
| [ChromaDB](https://docs.trychroma.com/) | Vector database |
| [FAISS](https://faiss.ai/) | Facebook similarity search |

### Multi-Agent Systems

| Resource | Description |
|----------|-------------|
| [CrewAI](https://docs.crewai.com/) | Multi-agent framework |
| [AutoGen](https://microsoft.github.io/autogen/) | Microsoft agent framework |
| [Swarm](https://github.com/openai/swarm) | OpenAI agent patterns |

---

## MLOps & Production

### Experiment Tracking

| Resource | Description |
|----------|-------------|
| [MLflow](https://mlflow.org/docs/latest/) | Experiment tracking, model registry |
| [Weights & Biases](https://docs.wandb.ai/) | Experiment tracking, visualization |
| [Neptune.ai](https://docs.neptune.ai/) | ML metadata store |

### Model Serving

| Resource | Description |
|----------|-------------|
| [vLLM](https://docs.vllm.ai/) | High-throughput LLM serving |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) | NVIDIA optimized inference |
| [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/) | Hugging Face serving |
| [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/) | NVIDIA model serving |

### Evaluation

| Resource | Description |
|----------|-------------|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Standard LLM benchmarks |
| [HELM](https://crfm.stanford.edu/helm/) | Stanford holistic evaluation |
| [OpenCompass](https://opencompass.org.cn/) | Open evaluation platform |

---

## Research Papers

### Must-Read Papers

| Paper | Year | Topic |
|-------|------|-------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer architecture |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Bidirectional pretraining |
| [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | Language model scaling |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 | In-context learning |
| [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Parameter-efficient fine-tuning |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | RLHF for instruction following |
| [LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | Open foundation models |
| [QLoRA](https://arxiv.org/abs/2305.14314) | 2023 | 4-bit fine-tuning |

### Staying Current

| Resource | Description |
|----------|-------------|
| [Arxiv Sanity](https://arxiv-sanity-lite.com/) | Paper discovery |
| [Papers With Code](https://paperswithcode.com/) | Papers with implementations |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI-powered paper search |
| [Connected Papers](https://www.connectedpapers.com/) | Paper relationship graphs |

---

## Books

### Fundamentals

| Book | Author | Notes |
|------|--------|-------|
| Deep Learning | Goodfellow, Bengio, Courville | Classic textbook, free online |
| Dive into Deep Learning | Zhang et al. | Interactive, code-focused, free |
| Hands-On Machine Learning | Aurélien Géron | Practical, updated for TF2/PyTorch |

### Advanced

| Book | Author | Notes |
|------|--------|-------|
| Understanding Deep Learning | Simon Prince | Modern, comprehensive |
| Probabilistic Machine Learning | Kevin Murphy | Advanced probabilistic methods |
| Designing Machine Learning Systems | Chip Huyen | MLOps and production |

### LLM Specific

| Book | Author | Notes |
|------|--------|-------|
| Build a Large Language Model | Sebastian Raschka | Hands-on LLM building |
| Natural Language Processing with Transformers | Lewis Tunstall et al. | Hugging Face focused |

---

## YouTube Channels

### Educational

| Channel | Focus |
|---------|-------|
| [3Blue1Brown](https://www.youtube.com/@3blue1brown) | Math intuition, neural networks |
| [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) | Neural networks from scratch |
| [Yannic Kilcher](https://www.youtube.com/@YannicKilcher) | Paper explanations |
| [AI Explained](https://www.youtube.com/@aiexplained-official) | AI news and concepts |
| [StatQuest](https://www.youtube.com/@statquest) | Statistics and ML basics |

### Technical Deep-Dives

| Channel | Focus |
|---------|-------|
| [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) | Research highlights |
| [The AI Epiphany](https://www.youtube.com/@TheAIEpiphany) | Implementation tutorials |
| [DeepLearning.AI](https://www.youtube.com/@Deeplearningai) | Course content |

---

## Podcasts

| Podcast | Focus |
|---------|-------|
| Lex Fridman Podcast | AI researcher interviews |
| Latent Space | AI engineering deep-dives |
| Gradient Dissent | Weights & Biases podcast |
| Practical AI | Applied AI topics |
| The TWIML AI Podcast | ML trends and interviews |

---

## Communities

### Forums & Discussion

| Community | Platform |
|-----------|----------|
| [r/MachineLearning](https://reddit.com/r/MachineLearning) | Reddit |
| [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) | Reddit - Local LLM focus |
| [Hugging Face Forums](https://discuss.huggingface.co/) | HF community |
| [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) | NVIDIA support |

### Discord

| Community | Focus |
|-----------|-------|
| Hugging Face | Transformers, Diffusers |
| LangChain | Agents, RAG |
| Unsloth | Fast fine-tuning |
| EleutherAI | Open research |
| LAION | Open datasets, models |

### Twitter/X Accounts to Follow

| Account | Focus |
|---------|-------|
| @kaborge | Transformers, PEFT |
| @_philschmid | HF, deployment |
| @rasaboroghazian | RAG, agents |
| @TheAIEpiphany | Implementations |
| @TimDettmers | Quantization |

---

## Tools & Libraries

### Essential Libraries

| Library | Purpose |
|---------|---------|
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [Transformers](https://huggingface.co/transformers) | Pre-trained models |
| [Accelerate](https://huggingface.co/accelerate) | Distributed training |
| [PEFT](https://github.com/huggingface/peft) | Parameter-efficient fine-tuning |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | Quantization |

### Inference Engines

| Tool | Best For |
|------|----------|
| [Ollama](https://ollama.com/) | Easy local inference |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | CPU/GPU inference, GGUF |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput serving |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA optimized |

### Development Tools

| Tool | Purpose |
|------|---------|
| [Jupyter](https://jupyter.org/) | Interactive notebooks |
| [VS Code](https://code.visualstudio.com/) | Code editor |
| [Docker](https://www.docker.com/) | Containerization |
| [Git](https://git-scm.com/) | Version control |

---

## Contribution

Found a great resource? Submit a PR to add it!

Guidelines:
- Ensure resources are high-quality and actively maintained
- Include a brief description
- Categorize appropriately
- Prefer free resources when available

---

*Last updated: January 2025*
