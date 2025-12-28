# NVIDIA DGX Spark AI Mastery Curriculum

## Curriculum Overview

| Attribute | Details |
|-----------|---------|
| **Target Hardware** | NVIDIA DGX Spark (GB10 Superchip, 128GB Unified Memory) |
| **Duration** | 24-32 weeks (standard pace) / 12-16 weeks (intensive) |
| **Prerequisites** | Basic Python knowledge, high school mathematics |
| **Primary Environment** | JupyterLab (native on DGX Spark) |
| **Testing Platform** | Your custom Ollama Web UI |

---

## Curriculum Structure

```
PHASE 1: FOUNDATIONS (Weeks 1-6)
├── Module 1: DGX Spark Platform Mastery
├── Module 2: Python for AI/ML
├── Module 3: Mathematics for Deep Learning
└── Module 4: Neural Network Fundamentals

PHASE 2: INTERMEDIATE (Weeks 7-14)
├── Module 5: Deep Learning with PyTorch
├── Module 6: Computer Vision
├── Module 7: Natural Language Processing & Transformers
└── Module 8: Hugging Face Ecosystem

PHASE 3: ADVANCED (Weeks 15-24)
├── Module 9: Large Language Model Fine-Tuning
├── Module 10: Model Quantization & Optimization
├── Module 11: Model Deployment & Inference Engines
├── Module 12: AI Agents & Agentic Systems
├── Module 13: Multimodal AI
└── Module 14: Benchmarking, Evaluation & MLOps

PHASE 4: CAPSTONE (Weeks 25-28)
└── Module 15: Capstone Project
```

---

# PHASE 1: FOUNDATIONS

## Module 1: DGX Spark Platform Mastery

**Duration:** Week 1 (5-8 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Explain the DGX Spark hardware architecture and its advantages for AI workloads
- Navigate the DGX OS environment and utilize pre-installed AI tools
- Configure JupyterLab for optimal AI development workflows
- Identify which open-source projects are compatible with DGX Spark

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.1 | Describe the Grace Blackwell GB10 architecture including CPU cores, GPU specs, and unified memory | Understand |
| 1.2 | Execute system monitoring commands to verify GPU and memory status | Apply |
| 1.3 | Configure NGC containers for PyTorch and other frameworks | Apply |
| 1.4 | Differentiate between compatible and incompatible open-source tools | Analyze |

### Topics

1. **Hardware Architecture Deep-Dive**
   - Grace Blackwell GB10 superchip (20 ARM cores, 6144 CUDA cores, 192 Tensor Cores)
   - 128GB LPDDR5X unified memory architecture
   - NVLink-C2C interconnect (273 GB/s bandwidth)
   - FP4/FP8/BF16/FP32 compute capabilities

2. **Software Environment**
   - DGX OS (Ubuntu 24.04 LTS)
   - CUDA 13.0.2, cuDNN, TensorRT
   - Pre-installed JupyterLab configuration
   - NGC container ecosystem

3. **Ecosystem Compatibility**
   - Fully compatible: Ollama, llama.cpp, NeMo
   - NGC required: PyTorch, JAX, Hugging Face
   - Partial support: vLLM, TensorRT-LLM, DeepSpeed

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 1.1 | **System Exploration** | Run `nvidia-smi`, `lscpu`, `free -h` and document your system specs. Create a Jupyter notebook summarizing your DGX Spark configuration. | 1 hour |
| 1.2 | **Memory Architecture Lab** | Write a Python script that allocates tensors of increasing size, monitoring memory with `torch.cuda.memory_summary()`. Document when CPU vs GPU memory is used. | 1.5 hours |
| 1.3 | **NGC Container Setup** | Pull and configure the PyTorch NGC container. Create a docker-compose.yml that mounts your home directory and Hugging Face cache. | 1.5 hours |
| 1.4 | **Compatibility Matrix** | Research and create a markdown table of 20 popular AI tools with their DGX Spark compatibility status and workarounds. | 2 hours |
| 1.5 | **Ollama Integration** | Configure Ollama, pull 3 different model sizes (7B, 13B, 70B), benchmark each with your Web UI, document performance. | 2 hours |

### Guidance

> **Critical:** Standard PyTorch pip wheels do NOT work on DGX Spark (ARM64 + CUDA). Always use NGC containers:
> ```bash
> docker pull nvcr.io/nvidia/pytorch:25.11-py3
> docker run --gpus all -v $HOME:/workspace -it nvcr.io/nvidia/pytorch:25.11-py3
> ```

> **Buffer Cache Management:** Before memory-intensive operations, clear the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> **JupyterLab Best Practice:** Create a startup script that initializes your preferred environment and mounts necessary volumes.

### Milestone Checklist

- [ ] Successfully ran nvidia-smi showing GB10 GPU
- [ ] Created system specification notebook
- [ ] NGC PyTorch container running with GPU access
- [ ] Ollama serving models through your Web UI
- [ ] Completed compatibility matrix document
- [ ] Documented unified memory behavior with tensor allocation test

---

## Module 2: Python for AI/ML

**Duration:** Week 2 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Write efficient NumPy code for tensor operations
- Manipulate datasets using Pandas for ML preprocessing
- Create publication-quality visualizations with Matplotlib/Seaborn
- Profile and optimize Python code for performance

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.1 | Implement vectorized operations using NumPy broadcasting | Apply |
| 2.2 | Transform and clean datasets using Pandas operations | Apply |
| 2.3 | Create multi-panel visualizations for model analysis | Create |
| 2.4 | Profile Python code and identify performance bottlenecks | Analyze |

### Topics

1. **NumPy Essentials**
   - Array creation, indexing, slicing
   - Broadcasting rules and vectorization
   - Linear algebra operations (dot, matmul, einsum)
   - Memory layout (C-contiguous vs F-contiguous)

2. **Pandas for ML**
   - DataFrame operations and transformations
   - Handling missing data
   - Feature engineering patterns
   - Efficient I/O (parquet, feather)

3. **Visualization**
   - Matplotlib fundamentals
   - Seaborn statistical plots
   - Training curve visualization
   - Attention heatmaps and activation maps

4. **Performance Optimization**
   - Profiling with cProfile and line_profiler
   - Numba JIT compilation
   - Memory profiling

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 2.1 | **NumPy Broadcasting Lab** | Implement matrix operations (batch matrix multiply, outer products) using broadcasting. Compare loop vs vectorized performance. Deliverable: Notebook with timing comparisons. | 2 hours |
| 2.2 | **Dataset Preprocessing Pipeline** | Download a real dataset (e.g., Titanic, Housing), implement complete preprocessing pipeline: handling nulls, encoding categoricals, feature scaling. Deliverable: Reusable preprocessing class. | 2 hours |
| 2.3 | **Visualization Dashboard** | Create a multi-panel figure showing: (1) training curves, (2) confusion matrix, (3) feature importance, (4) prediction distribution. Use synthetic data if needed. | 2 hours |
| 2.4 | **Einsum Mastery** | Implement attention mechanism using `np.einsum`. Compare with explicit loop and matmul versions. Document einsum notation. | 2 hours |
| 2.5 | **Profiling Exercise** | Profile a slow function, identify bottlenecks, optimize with vectorization/Numba. Achieve 10x+ speedup. Document before/after. | 2 hours |

### Guidance

> **Einsum Tip:** Master einsum notation early—it's essential for understanding attention mechanisms:
> ```python
> # Batch matrix multiply: (B, M, K) @ (B, K, N) -> (B, M, N)
> np.einsum('bmk,bkn->bmn', A, B)
> ```

> **Memory Efficiency:** Use `np.float32` instead of `np.float64` for ML workloads—halves memory with negligible precision loss.

### Milestone Checklist

- [ ] Vectorized batch operations notebook complete
- [ ] Reusable preprocessing pipeline class created
- [ ] Multi-panel visualization dashboard created
- [ ] Einsum attention implementation working
- [ ] Achieved 10x+ speedup in profiling exercise
- [ ] All notebooks documented with markdown explanations

---

## Module 3: Mathematics for Deep Learning

**Duration:** Week 3 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Compute gradients for neural network operations manually and verify with autograd
- Explain and implement common optimization algorithms
- Interpret loss landscapes and understand convergence behavior
- Apply linear algebra concepts to neural network operations

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.1 | Compute partial derivatives for composite functions using chain rule | Apply |
| 3.2 | Implement gradient descent variants from scratch | Apply |
| 3.3 | Visualize and interpret loss landscapes | Analyze |
| 3.4 | Perform matrix calculus for backpropagation derivation | Understand |

### Topics

1. **Linear Algebra for Neural Networks**
   - Matrix/vector operations and their gradients
   - Eigenvalues/eigenvectors (for PCA, weight initialization)
   - Singular Value Decomposition (for LoRA understanding)
   - Tensor operations and reshaping

2. **Calculus for Backpropagation**
   - Chain rule for composite functions
   - Partial derivatives and gradients
   - Jacobian and Hessian matrices
   - Computational graphs

3. **Optimization Theory**
   - Gradient descent and learning rates
   - Momentum and adaptive methods (Adam, AdamW)
   - Learning rate schedules
   - Loss landscape geometry

4. **Probability for ML**
   - Probability distributions (Gaussian, Categorical)
   - Maximum likelihood estimation
   - Cross-entropy and KL divergence
   - Bayesian basics

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 3.1 | **Manual Backprop** | Implement forward and backward pass for a 3-layer MLP from scratch (no autograd). Verify gradients match PyTorch autograd. | 3 hours |
| 3.2 | **Optimizer Implementation** | Implement SGD, Momentum, and Adam optimizers from scratch. Train on a simple problem, compare convergence curves. | 2 hours |
| 3.3 | **Loss Landscape Visualization** | Create 2D and 3D visualizations of loss landscapes for a simple network. Identify local minima, saddle points. | 2 hours |
| 3.4 | **SVD for LoRA Intuition** | Decompose a weight matrix using SVD, reconstruct with varying ranks, visualize information loss. Connect to LoRA concept. | 2 hours |
| 3.5 | **Probability Distributions Lab** | Implement and visualize common distributions. Derive cross-entropy loss from maximum likelihood principle. | 2 hours |

### Guidance

> **Gradient Checking:** Always verify manual gradients with numerical approximation:
> ```python
> def grad_check(f, x, eps=1e-5):
>     return (f(x + eps) - f(x - eps)) / (2 * eps)
> ```

> **SVD Connection to LoRA:** Understanding that any matrix W can be decomposed as W = UΣV^T helps you understand why LoRA's low-rank adaptation (W + BA) works—we're adding a low-rank perturbation.

### Milestone Checklist

- [ ] Manual backprop implementation matches autograd within 1e-6
- [ ] Three optimizers implemented and compared
- [ ] Loss landscape visualizations created
- [ ] SVD decomposition notebook complete
- [ ] Cross-entropy derivation documented
- [ ] All concepts connected to neural network applications

---

## Module 4: Neural Network Fundamentals

**Duration:** Weeks 4-5 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build neural networks from scratch using only NumPy
- Explain the purpose of each neural network component
- Train networks on real datasets and diagnose common issues
- Implement regularization and normalization techniques

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.1 | Implement fully-connected layers with forward and backward passes | Apply |
| 4.2 | Explain vanishing/exploding gradients and implement solutions | Understand |
| 4.3 | Apply regularization techniques (L2, dropout) to prevent overfitting | Apply |
| 4.4 | Diagnose training issues from loss curves and metrics | Analyze |

### Topics

1. **Perceptron to MLP**
   - Single neuron and activation functions
   - Multi-layer perceptrons
   - Universal approximation theorem

2. **Activation Functions**
   - Sigmoid, Tanh, ReLU, GELU, SiLU
   - Vanishing gradient problem
   - Choosing activations for different tasks

3. **Loss Functions**
   - MSE for regression
   - Cross-entropy for classification
   - Custom losses

4. **Regularization**
   - L1/L2 regularization
   - Dropout
   - Early stopping
   - Data augmentation concepts

5. **Normalization**
   - Batch normalization
   - Layer normalization
   - RMSNorm (for transformers)

6. **Weight Initialization**
   - Xavier/Glorot initialization
   - He initialization
   - Impact on training dynamics

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 4.1 | **NumPy Neural Network** | Build a complete MLP from scratch: Linear, ReLU, Softmax, CrossEntropy, SGD. Train on MNIST to >95% accuracy. | 4 hours |
| 4.2 | **Activation Function Study** | Implement 6 activation functions, visualize their outputs and gradients, train same network with each, compare results. | 2 hours |
| 4.3 | **Regularization Experiments** | Train networks with varying L2 strength and dropout rates. Create visualization showing underfitting → good fit → overfitting. | 2 hours |
| 4.4 | **Normalization Comparison** | Implement BatchNorm and LayerNorm from scratch. Compare training dynamics with/without normalization. | 2 hours |
| 4.5 | **Training Diagnostics Lab** | Deliberately create training problems (bad LR, vanishing gradients, overfitting), document symptoms, implement fixes. | 2 hours |
| 4.6 | **GPU Acceleration** | Port your NumPy MLP to PyTorch, compare training speed CPU vs GPU on DGX Spark. | 2 hours |

### Guidance

> **Learning Rate Selection:** Start with 1e-3 for Adam, 1e-2 for SGD with momentum. If loss explodes, reduce by 10x. If loss decreases too slowly, increase by 3x.

> **Debugging Checklist:**
> 1. Can your network overfit a single batch? (If not, bug in forward/backward)
> 2. Does loss decrease at all? (If not, check learning rate, gradients)
> 3. Does validation loss increase while training loss decreases? (Overfitting)

### Milestone Checklist

- [ ] NumPy MLP achieving >95% on MNIST
- [ ] Activation function comparison notebook complete
- [ ] Regularization experiments documented
- [ ] BatchNorm and LayerNorm implementations working
- [ ] Training diagnostics guide created
- [ ] GPU vs CPU speedup measured and documented

---

## Module 5: Phase 1 Capstone — From-Scratch Neural Network Library

**Duration:** Week 6 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Design and implement a modular neural network library
- Create reusable, well-documented code following software engineering practices
- Benchmark your implementation against PyTorch

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 5.1 | Design a modular architecture for neural network components | Create |
| 5.2 | Implement automatic differentiation for common operations | Apply |
| 5.3 | Write comprehensive unit tests for neural network operations | Apply |
| 5.4 | Document code with docstrings and usage examples | Create |

### Capstone Project: MicroGrad+

Build an extended version of Andrej Karpathy's micrograd with additional features:

**Required Components:**
1. Tensor class with automatic differentiation
2. Layer classes: Linear, Conv2D (basic), ReLU, Softmax
3. Loss functions: MSE, CrossEntropy
4. Optimizers: SGD, Adam
5. Training loop abstraction

**Deliverables:**
1. Python package with proper structure (`micrograd_plus/`)
2. Unit tests with >80% coverage
3. Documentation with API reference
4. Example notebooks training on MNIST and CIFAR-10
5. Performance benchmark vs PyTorch

### Milestone Checklist

- [ ] Package structure created with proper `__init__.py`
- [ ] All core components implemented
- [ ] Unit tests passing with >80% coverage
- [ ] MNIST example achieving >95% accuracy
- [ ] Documentation complete
- [ ] Benchmark notebook comparing with PyTorch

---

# PHASE 2: INTERMEDIATE

## Module 6: Deep Learning with PyTorch

**Duration:** Week 7-8 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build complex neural networks using PyTorch's nn.Module
- Implement custom datasets and data loaders
- Utilize PyTorch's autograd for custom operations
- Debug and profile PyTorch models effectively

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 6.1 | Create custom nn.Module classes with proper initialization | Apply |
| 6.2 | Implement Dataset and DataLoader for custom data | Apply |
| 6.3 | Use hooks for model introspection and debugging | Apply |
| 6.4 | Profile models with PyTorch Profiler | Analyze |

### Topics

1. **PyTorch Fundamentals**
   - Tensors and operations
   - Autograd mechanics
   - GPU memory management
   - Mixed precision with AMP

2. **Building Models**
   - nn.Module architecture
   - Sequential vs functional API
   - Parameter registration
   - State dict and checkpointing

3. **Data Pipeline**
   - Dataset class implementation
   - DataLoader with workers
   - Transforms and augmentation
   - Efficient data loading on DGX Spark

4. **Training Infrastructure**
   - Training loops
   - Validation and metrics
   - Learning rate scheduling
   - Gradient clipping and accumulation

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 6.1 | **Custom Module Lab** | Implement a ResNet block as custom nn.Module. Stack blocks to create ResNet-18. | 2 hours |
| 6.2 | **Dataset Pipeline** | Create custom Dataset for a local image folder. Implement DataLoader with augmentation. Benchmark loading speed. | 2 hours |
| 6.3 | **Autograd Deep Dive** | Implement custom autograd Function for a novel activation. Verify gradients with gradcheck. | 2 hours |
| 6.4 | **Mixed Precision Training** | Train a model with AMP (torch.cuda.amp). Compare memory usage and speed vs FP32. | 2 hours |
| 6.5 | **Profiling Workshop** | Profile a training loop with PyTorch Profiler. Identify bottlenecks, optimize, measure improvement. | 2 hours |
| 6.6 | **Checkpointing System** | Implement robust checkpointing: save/resume training, best model tracking, early stopping. | 2 hours |

### Guidance

> **DGX Spark Memory Tip:** The unified memory means you can load larger batches than traditional GPUs. Start with batch_size=64 and increase until you see slowdowns.

> **NGC Container Usage:**
> ```bash
> docker run --gpus all --ipc=host --net=host \
>   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
>   -v $PWD:/workspace -w /workspace \
>   nvcr.io/nvidia/pytorch:25.11-py3 \
>   jupyter lab --ip=0.0.0.0 --allow-root
> ```

### Milestone Checklist

- [ ] ResNet-18 implemented from scratch
- [ ] Custom dataset pipeline working
- [ ] Custom autograd function with verified gradients
- [ ] AMP training notebook with memory comparison
- [ ] Profiling report with optimization applied
- [ ] Checkpointing system tested (interrupt and resume training)

---

## Module 7: Computer Vision

**Duration:** Weeks 9-10 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Implement and train CNN architectures for image classification
- Apply transfer learning for custom image tasks
- Perform object detection using pre-trained models
- Understand and implement image segmentation

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 7.1 | Explain the evolution from LeNet to modern architectures | Understand |
| 7.2 | Implement data augmentation pipelines for image data | Apply |
| 7.3 | Fine-tune pre-trained models on custom datasets | Apply |
| 7.4 | Evaluate model performance using appropriate CV metrics | Analyze |

### Topics

1. **CNN Architectures**
   - Convolution and pooling operations
   - LeNet, AlexNet, VGG
   - ResNet and skip connections
   - Modern architectures: EfficientNet, ConvNeXt

2. **Transfer Learning**
   - Pre-trained model selection
   - Feature extraction vs fine-tuning
   - Learning rate strategies for transfer learning

3. **Object Detection**
   - Region-based methods (R-CNN family)
   - Single-shot detectors (YOLO, SSD)
   - Using pre-trained detectors

4. **Image Segmentation**
   - Semantic vs instance segmentation
   - U-Net architecture
   - Segment Anything Model (SAM)

5. **Vision Transformers**
   - ViT architecture
   - Patch embeddings
   - DeiT and training tricks

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 7.1 | **CNN Architecture Study** | Implement LeNet, AlexNet (simplified), and ResNet-18. Train on CIFAR-10, compare accuracy and training dynamics. | 3 hours |
| 7.2 | **Transfer Learning Project** | Fine-tune a pre-trained EfficientNet on a custom dataset (e.g., food classification). Achieve >90% accuracy. | 3 hours |
| 7.3 | **Object Detection Demo** | Use YOLOv8 to detect objects in custom images/video. Document inference speed on DGX Spark. | 2 hours |
| 7.4 | **Segmentation Lab** | Implement U-Net for semantic segmentation. Train on a simple dataset (e.g., pet segmentation). | 3 hours |
| 7.5 | **Vision Transformer** | Implement ViT from scratch. Train on CIFAR-10 and compare with CNN. | 3 hours |
| 7.6 | **SAM Integration** | Use Segment Anything Model for interactive segmentation. Create a demo notebook. | 2 hours |

### Guidance

> **Data Augmentation Best Practices:**
> ```python
> transforms.Compose([
>     transforms.RandomResizedCrop(224),
>     transforms.RandomHorizontalFlip(),
>     transforms.ColorJitter(0.4, 0.4, 0.4),
>     transforms.ToTensor(),
>     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
> ])
> ```

> **Transfer Learning LR Strategy:** Use 10x smaller LR for pre-trained layers vs new layers.

### Milestone Checklist

- [ ] Three CNN architectures implemented and compared
- [ ] Transfer learning project achieving >90% accuracy
- [ ] Object detection demo working
- [ ] U-Net segmentation model trained
- [ ] Vision Transformer implemented from scratch
- [ ] SAM demo notebook complete

---

## Module 8: Natural Language Processing & Transformers

**Duration:** Weeks 11-12 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Implement the Transformer architecture from scratch
- Explain attention mechanisms and their variations
- Apply tokenization strategies for different tasks
- Fine-tune language models for downstream tasks

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 8.1 | Implement multi-head self-attention from scratch | Apply |
| 8.2 | Explain positional encoding strategies (sinusoidal, rotary) | Understand |
| 8.3 | Tokenize text using BPE and SentencePiece | Apply |
| 8.4 | Fine-tune BERT for classification tasks | Apply |

### Topics

1. **Attention Mechanisms**
   - Scaled dot-product attention
   - Multi-head attention
   - Cross-attention
   - Attention visualization

2. **Transformer Architecture**
   - Encoder and decoder blocks
   - Layer normalization placement (Pre-LN vs Post-LN)
   - Feed-forward networks
   - Residual connections

3. **Positional Encodings**
   - Sinusoidal encoding
   - Learned embeddings
   - Rotary Position Embeddings (RoPE)
   - ALiBi

4. **Tokenization**
   - Word-level vs subword tokenization
   - BPE algorithm
   - SentencePiece and tokenizers library

5. **Pre-trained Models**
   - BERT and masked language modeling
   - GPT and causal language modeling
   - T5 and encoder-decoder models

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 8.1 | **Attention from Scratch** | Implement scaled dot-product and multi-head attention. Visualize attention patterns on sample sentences. | 2 hours |
| 8.2 | **Transformer Block** | Implement a complete Transformer encoder block. Stack 6 blocks to create encoder. | 3 hours |
| 8.3 | **Positional Encoding Study** | Implement sinusoidal and RoPE encodings. Visualize and compare their properties. | 2 hours |
| 8.4 | **Tokenization Lab** | Train a BPE tokenizer on custom text. Compare with pre-trained tokenizers (GPT-2, LLaMA). | 2 hours |
| 8.5 | **BERT Fine-tuning** | Fine-tune BERT for sentiment classification. Evaluate on test set, analyze errors. | 2 hours |
| 8.6 | **GPT Text Generation** | Load GPT-2, implement different decoding strategies (greedy, beam search, sampling). Test with your Ollama UI. | 2 hours |

### Guidance

> **Attention Dimension Formula:**
> ```python
> # d_k = d_model // num_heads
> attention = softmax(Q @ K.T / sqrt(d_k)) @ V
> ```

> **RoPE vs Sinusoidal:** RoPE applies rotation to Q/K based on position, allowing relative position information. Better for long sequences.

### Milestone Checklist

- [ ] Multi-head attention implementation complete
- [ ] Full Transformer encoder working
- [ ] Both positional encoding types implemented
- [ ] Custom BPE tokenizer trained
- [ ] BERT fine-tuning achieving good accuracy
- [ ] Text generation with multiple decoding strategies

---

## Module 9: Hugging Face Ecosystem

**Duration:** Weeks 13-14 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Navigate and utilize the Hugging Face Hub effectively
- Use Transformers library for various NLP tasks
- Load and preprocess datasets with the Datasets library
- Apply the Trainer API for efficient model training

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 9.1 | Load and use pre-trained models from Hugging Face Hub | Apply |
| 9.2 | Preprocess datasets using datasets library transformations | Apply |
| 9.3 | Configure and use the Trainer API for fine-tuning | Apply |
| 9.4 | Evaluate models using the evaluate library | Analyze |

### Topics

1. **Hugging Face Hub**
   - Model cards and documentation
   - Model discovery and selection
   - Uploading models and datasets

2. **Transformers Library**
   - Auto classes (AutoModel, AutoTokenizer)
   - Pipeline API for quick inference
   - Model-specific classes

3. **Datasets Library**
   - Loading datasets
   - Map and filter operations
   - Streaming for large datasets

4. **Training with HF**
   - Trainer and TrainingArguments
   - Custom training loops with Accelerate
   - Evaluation metrics

5. **PEFT Library**
   - Introduction to parameter-efficient fine-tuning
   - LoRA configuration
   - Merging adapters

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 9.1 | **Hub Exploration** | Explore HF Hub, find 10 interesting models, document their use cases, download and test 3 locally. | 2 hours |
| 9.2 | **Pipeline Showcase** | Create notebook demonstrating 5 different pipelines: text-generation, sentiment, NER, QA, summarization. | 2 hours |
| 9.3 | **Dataset Processing** | Load a large dataset (>1M samples), apply preprocessing with map(), create train/val/test splits. | 2 hours |
| 9.4 | **Trainer Fine-tuning** | Use Trainer API to fine-tune a model for text classification. Implement custom metrics callback. | 2 hours |
| 9.5 | **LoRA Introduction** | Apply LoRA to a small model using PEFT. Compare trainable parameters and memory usage vs full fine-tuning. | 2 hours |
| 9.6 | **Model Upload** | Fine-tune a model, create model card, upload to HF Hub (can be private). | 2 hours |

### Guidance

> **Model Selection on DGX Spark:** With 128GB unified memory, you can load models up to ~70B parameters in FP16, or ~140B in 8-bit quantization.

> **Accelerate for Multi-GPU:** If you connect two DGX Sparks via ConnectX-7:
> ```bash
> accelerate config  # Set up distributed training
> accelerate launch train.py
> ```

### Milestone Checklist

- [ ] 10 models documented from HF Hub
- [ ] 5 pipeline demonstrations complete
- [ ] Large dataset processing pipeline working
- [ ] Trainer fine-tuning complete with custom metrics
- [ ] LoRA fine-tuning comparison documented
- [ ] Model uploaded to HF Hub with model card

---

# PHASE 3: ADVANCED

## Module 10: Large Language Model Fine-Tuning

**Duration:** Weeks 15-17 (15-18 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Fine-tune LLMs using LoRA, QLoRA, and full fine-tuning
- Prepare datasets for instruction tuning and chat formats
- Select appropriate fine-tuning strategies based on resources and goals
- Evaluate fine-tuned models for task performance

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 10.1 | Explain the mathematical foundations of LoRA | Understand |
| 10.2 | Configure and execute QLoRA fine-tuning for 70B models | Apply |
| 10.3 | Prepare datasets in instruction-following formats | Apply |
| 10.4 | Evaluate and compare fine-tuned models | Evaluate |

### Topics

1. **Fine-Tuning Strategies**
   - Full fine-tuning
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
   - DoRA, rsLoRA variants
   - When to use each approach

2. **LoRA Deep Dive**
   - Low-rank decomposition theory
   - Rank selection (r parameter)
   - Alpha scaling
   - Target modules selection

3. **Dataset Preparation**
   - Instruction format (Alpaca, ShareGPT)
   - Chat templates
   - Data quality considerations
   - Synthetic data generation

4. **Training Infrastructure**
   - Gradient checkpointing
   - Memory optimization
   - Unsloth acceleration
   - LLaMA Factory GUI

5. **RLHF and Preference Optimization**
   - Reward modeling
   - PPO training
   - DPO (Direct Preference Optimization)
   - ORPO, SimPO variants

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 10.1 | **LoRA Theory Notebook** | Implement LoRA from scratch on a small transformer. Visualize weight updates. Document rank vs performance tradeoff. | 3 hours |
| 10.2 | **8B Model LoRA Fine-tuning** | Fine-tune Llama 3.1 8B with LoRA on a custom dataset. Use Unsloth for acceleration. | 3 hours |
| 10.3 | **70B Model QLoRA** | Fine-tune a 70B model using QLoRA. This is where DGX Spark shines—document memory usage, demonstrate capability. | 4 hours |
| 10.4 | **Dataset Preparation** | Create an instruction dataset from raw data. Implement Alpaca and ChatML formats. Include data cleaning. | 2 hours |
| 10.5 | **DPO Training** | Implement preference optimization using DPO. Compare with SFT-only baseline. | 3 hours |
| 10.6 | **LLaMA Factory Exploration** | Use LLaMA Factory web UI for fine-tuning. Document the workflow and compare with script-based approach. | 2 hours |
| 10.7 | **Integration with Ollama** | Convert fine-tuned model to GGUF, import to Ollama, test in your Web UI. | 2 hours |

### Guidance

> **DGX Spark QLoRA Config for 70B:**
> ```python
> bnb_config = BitsAndBytesConfig(
>     load_in_4bit=True,
>     bnb_4bit_quant_type="nf4",
>     bnb_4bit_compute_dtype=torch.bfloat16,
>     bnb_4bit_use_double_quant=True
> )
> # Expect ~45-55GB memory usage
> ```

> **LoRA Rank Guidelines:**
> - r=8: Quick experiments, minimal quality impact
> - r=16: Good balance for most tasks
> - r=32-64: Complex tasks requiring more adaptation
> - r=128+: Approaching full fine-tuning capacity

> **Clear cache between runs:**
> ```python
> import torch, gc
> torch.cuda.empty_cache()
> gc.collect()
> ```

### Milestone Checklist

- [ ] LoRA theory notebook with from-scratch implementation
- [ ] 8B model successfully fine-tuned with LoRA
- [ ] 70B model fine-tuned with QLoRA (DGX Spark capability demonstration)
- [ ] Custom instruction dataset created
- [ ] DPO preference optimization completed
- [ ] LLaMA Factory workflow documented
- [ ] Fine-tuned model running in Ollama via your Web UI

---

## Module 11: Model Quantization & Optimization

**Duration:** Weeks 18-19 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Apply various quantization techniques (GPTQ, AWQ, GGUF, FP4/FP8)
- Optimize models for inference using TensorRT
- Evaluate quantization impact on model quality
- Select optimal quantization strategy for deployment requirements

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 11.1 | Explain different quantization methods and their tradeoffs | Understand |
| 11.2 | Quantize models using GPTQ, AWQ, and GGUF | Apply |
| 11.3 | Apply Blackwell-exclusive FP4 quantization | Apply |
| 11.4 | Measure and compare quality degradation from quantization | Evaluate |

### Topics

1. **Quantization Fundamentals**
   - Data types: FP32, FP16, BF16, INT8, INT4, FP8, FP4
   - Post-training quantization vs quantization-aware training
   - Calibration datasets

2. **Quantization Methods**
   - GPTQ (GPU-optimized PTQ)
   - AWQ (Activation-aware Quantization)
   - GGUF format for llama.cpp
   - bitsandbytes NF4

3. **Blackwell-Specific Optimization**
   - NVFP4 format with dual-level scaling
   - MXFP4 (Open Compute Project format)
   - FP8 training and inference
   - TensorRT optimization

4. **Quality Evaluation**
   - Perplexity measurement
   - Task-specific benchmarks
   - Output comparison

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 11.1 | **Quantization Overview** | Create comparison notebook: load same model in FP16, INT8, INT4. Measure size, speed, perplexity. | 2 hours |
| 11.2 | **GPTQ Quantization** | Quantize a 7B model with GPTQ. Compare different group sizes (32, 64, 128). | 2 hours |
| 11.3 | **AWQ Quantization** | Quantize same model with AWQ. Compare with GPTQ results. | 1.5 hours |
| 11.4 | **GGUF Conversion** | Convert model to GGUF format. Test with llama.cpp, measure performance on DGX Spark. | 2 hours |
| 11.5 | **FP4 Deep Dive** | Use TensorRT Model Optimizer for NVFP4 quantization. Benchmark against FP16 baseline. Document DGX Spark's FP4 advantage. | 3 hours |
| 11.6 | **Quality Benchmark Suite** | Create standardized benchmark: perplexity, MMLU sample, generation quality. Run on all quantization variants. | 2 hours |

### Guidance

> **NVFP4 Quantization Workflow:**
> ```python
> from modelopt.torch.quantization import quantize
> model = quantize(model, quant_cfg="nvfp4", calibration_dataloader=calib_dl)
> # Then export to TensorRT-LLM
> ```

> **Perplexity Target:** Less than 0.5 perplexity increase from FP16 baseline indicates acceptable quantization quality.

> **DGX Spark FP4 Performance:** Expect ~10,000 tok/s prefill for 8B models in NVFP4—this is the Blackwell advantage.

### Milestone Checklist

- [ ] Quantization comparison table created (size, speed, perplexity)
- [ ] GPTQ quantization with multiple configurations
- [ ] AWQ quantization completed
- [ ] GGUF conversion and llama.cpp testing done
- [ ] NVFP4 quantization demonstrating DGX Spark capability
- [ ] Quality benchmark suite created and run on all variants

---

## Module 12: Model Deployment & Inference Engines

**Duration:** Weeks 20-21 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Deploy models using various inference engines
- Optimize inference for latency and throughput
- Implement serving APIs for production use
- Select the right inference engine for different requirements

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 12.1 | Compare inference engines (Ollama, llama.cpp, vLLM, TensorRT-LLM) | Analyze |
| 12.2 | Deploy models as REST APIs | Apply |
| 12.3 | Implement continuous batching for throughput optimization | Apply |
| 12.4 | Configure and use speculative decoding | Apply |

### Topics

1. **Inference Engine Overview**
   - Ollama (user-friendly, optimized for DGX Spark)
   - llama.cpp (fastest decode, GGUF format)
   - vLLM (PagedAttention, continuous batching)
   - TensorRT-LLM (best prefill, NVIDIA optimized)
   - SGLang (speculative decoding)

2. **Optimization Techniques**
   - Continuous batching
   - KV cache optimization
   - Speculative decoding
   - Tensor parallelism

3. **Serving Infrastructure**
   - REST API design
   - Streaming responses
   - Load balancing
   - Health checks and monitoring

4. **Production Considerations**
   - Latency vs throughput tradeoffs
   - Memory management
   - Scaling strategies

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 12.1 | **Engine Benchmark** | Benchmark same model on Ollama, llama.cpp, vLLM, TensorRT-LLM. Measure prefill, decode, memory. Create comparison report. | 3 hours |
| 12.2 | **vLLM Deployment** | Deploy model with vLLM. Implement continuous batching. Measure throughput under concurrent requests. | 2 hours |
| 12.3 | **TensorRT-LLM Optimization** | Build TensorRT-LLM engine for a model. Compare with other engines on prefill speed. | 3 hours |
| 12.4 | **Speculative Decoding** | Configure SGLang with EAGLE-3 speculative decoding. Measure speedup vs standard decoding. | 2 hours |
| 12.5 | **Production API** | Create FastAPI wrapper around your preferred engine. Implement streaming, error handling, monitoring endpoint. | 2 hours |
| 12.6 | **Ollama Web UI Integration** | Optimize your existing Ollama Web UI based on learnings. Add model switching, performance metrics display. | 2 hours |

### Guidance

> **Engine Selection Guide:**
> - **Highest decode speed:** llama.cpp (~59 tok/s on DGX Spark)
> - **Best prefill:** TensorRT-LLM (~10,000 tok/s for 8B FP4)
> - **Easiest setup:** Ollama (pre-optimized for DGX Spark)
> - **Best throughput under load:** vLLM with continuous batching

> **vLLM on DGX Spark:**
> ```bash
> docker pull nvcr.io/nvidia/vllm:spark
> # Use --enforce-eager if you encounter issues
> ```

### Milestone Checklist

- [ ] Comprehensive engine benchmark report created
- [ ] vLLM deployment with continuous batching working
- [ ] TensorRT-LLM engine built and tested
- [ ] Speculative decoding speedup measured
- [ ] Production FastAPI server implemented
- [ ] Ollama Web UI enhanced with new features

---

## Module 13: AI Agents & Agentic Systems

**Duration:** Weeks 22-23 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build AI agents using LangChain and LlamaIndex
- Implement Retrieval-Augmented Generation (RAG) systems
- Create multi-agent systems for complex tasks
- Design and implement tool-using agents

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 13.1 | Implement RAG pipeline with vector database | Apply |
| 13.2 | Create agents with custom tools | Apply |
| 13.3 | Design multi-agent architectures | Create |
| 13.4 | Evaluate agent performance and reliability | Evaluate |

### Topics

1. **RAG Fundamentals**
   - Document loading and chunking
   - Embedding models
   - Vector databases (ChromaDB, FAISS, Qdrant)
   - Retrieval strategies

2. **LangChain Framework**
   - Chains and composition
   - Agents and tools
   - Memory systems
   - Callbacks and tracing

3. **LlamaIndex**
   - Index types
   - Query engines
   - Response synthesis
   - Advanced retrieval

4. **LangGraph & Agentic Workflows**
   - Stateful agents
   - Graph-based orchestration
   - Human-in-the-loop
   - Error recovery

5. **Multi-Agent Systems**
   - Agent communication patterns
   - CrewAI framework
   - AutoGen for coding agents
   - Task decomposition

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 13.1 | **RAG Pipeline** | Build complete RAG system: document ingestion, embedding, ChromaDB storage, retrieval, generation. Test on technical documentation. | 3 hours |
| 13.2 | **Custom Tools** | Create 5 custom tools (web search, calculator, code executor, file reader, API caller). Build agent that uses them. | 3 hours |
| 13.3 | **LlamaIndex Query Engine** | Build advanced query engine with hybrid search, reranking, and source citations. | 2 hours |
| 13.4 | **LangGraph Workflow** | Implement multi-step workflow with branching logic, error handling, human approval step. | 3 hours |
| 13.5 | **Multi-Agent System** | Create 3-agent system (researcher, writer, reviewer) for content generation. Use CrewAI or custom orchestration. | 3 hours |
| 13.6 | **Agent Benchmark** | Create evaluation framework: task completion rate, quality, token efficiency. Test your agents systematically. | 2 hours |

### Guidance

> **Local Agent Stack on DGX Spark:**
> - Embeddings: Run `nomic-embed-text` or `bge-large` locally via Ollama
> - LLM: 70B model via Ollama for best agent performance
> - Vector DB: ChromaDB or Qdrant locally

> **RAG Chunking Strategy:**
> - Chunk size: 512-1024 tokens
> - Overlap: 10-20%
> - Use semantic chunking for better retrieval

> **Agent Reliability:** Implement retry logic and fallbacks. Agents will fail—design for graceful degradation.

### Milestone Checklist

- [ ] RAG pipeline working with technical docs
- [ ] 5 custom tools implemented and tested
- [ ] LlamaIndex query engine with citations
- [ ] LangGraph workflow with branching and human-in-the-loop
- [ ] Multi-agent content generation system working
- [ ] Agent evaluation framework created

---

## Module 14: Multimodal AI

**Duration:** Week 24 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Work with vision-language models for image understanding
- Implement image generation with diffusion models
- Build multimodal pipelines combining vision and language
- Fine-tune multimodal models

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 14.1 | Use vision-language models for image analysis | Apply |
| 14.2 | Generate images using Stable Diffusion | Apply |
| 14.3 | Build multimodal RAG systems | Apply |
| 14.4 | Fine-tune vision-language models | Apply |

### Topics

1. **Vision-Language Models**
   - LLaVA architecture
   - CLIP and embeddings
   - Qwen-VL, InternVL
   - Document understanding (OCR, layout)

2. **Image Generation**
   - Stable Diffusion fundamentals
   - ControlNet for guided generation
   - SDXL and Flux models
   - LoRA for style transfer

3. **Audio Models**
   - Whisper for transcription
   - Text-to-speech models
   - Audio understanding

4. **Multimodal Pipelines**
   - Document AI
   - Video understanding
   - Multimodal RAG

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 14.1 | **Vision-Language Demo** | Deploy LLaVA or Qwen-VL on DGX Spark. Create notebook demonstrating image understanding, VQA, OCR. | 2 hours |
| 14.2 | **Image Generation** | Run SDXL or Flux on DGX Spark. Experiment with prompts, negative prompts, ControlNet. | 2 hours |
| 14.3 | **Multimodal RAG** | Build RAG system that indexes images (using CLIP) and text. Query with natural language. | 2 hours |
| 14.4 | **Document AI Pipeline** | Create pipeline for PDF processing: OCR, layout analysis, question answering over documents. | 2 hours |
| 14.5 | **Audio Transcription** | Deploy Whisper, create transcription pipeline. Combine with LLM for audio Q&A. | 2 hours |

### Guidance

> **VLM Memory on DGX Spark:** LLaVA-34B fits comfortably. Qwen2-VL-72B works with quantization.

> **Image Generation Performance:** SDXL generates 1024x1024 images in ~5-8 seconds on DGX Spark. Flux is slower but higher quality.

### Milestone Checklist

- [ ] Vision-language model running with demo notebook
- [ ] Image generation working with various techniques
- [ ] Multimodal RAG indexing images and text
- [ ] Document AI pipeline processing PDFs
- [ ] Audio transcription and Q&A pipeline working

---

## Module 15: Benchmarking, Evaluation & MLOps

**Duration:** Week 25-26 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Evaluate LLMs using standard benchmarks
- Implement comprehensive evaluation frameworks
- Set up experiment tracking and model versioning
- Create reproducible ML pipelines

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 15.1 | Run standard LLM benchmarks (MMLU, HellaSwag, etc.) | Apply |
| 15.2 | Design custom evaluation suites for specific use cases | Create |
| 15.3 | Track experiments with MLflow or Weights & Biases | Apply |
| 15.4 | Version datasets and models systematically | Apply |

### Topics

1. **LLM Benchmarks**
   - MMLU, HellaSwag, ARC, WinoGrande
   - HumanEval for code
   - MT-Bench for chat
   - LM Evaluation Harness

2. **Custom Evaluation**
   - Task-specific metrics
   - LLM-as-judge
   - Human evaluation protocols
   - A/B testing

3. **Experiment Tracking**
   - MLflow setup and usage
   - Weights & Biases
   - Hyperparameter logging
   - Artifact management

4. **MLOps Practices**
   - Model versioning (DVC, HF Hub)
   - Dataset versioning
   - Reproducibility
   - CI/CD for ML

### Tasks

| # | Task | Description & Deliverable | Est. Time |
|---|------|---------------------------|-----------|
| 15.1 | **Benchmark Suite** | Run lm-evaluation-harness on your models. Create comparison tables across model sizes and quantization levels. | 3 hours |
| 15.2 | **Custom Eval Framework** | Design evaluation for your specific use case. Include automatic metrics and LLM-as-judge. | 2 hours |
| 15.3 | **MLflow Setup** | Set up MLflow on DGX Spark. Log training runs, parameters, metrics, artifacts. Create dashboard. | 2 hours |
| 15.4 | **Model Registry** | Create model versioning workflow. Track base models, fine-tuned versions, quantized variants. | 2 hours |
| 15.5 | **Reproducibility Audit** | Document complete environment, dependencies, random seeds. Verify you can reproduce a previous training run. | 2 hours |

### Guidance

> **lm-evaluation-harness:**
> ```bash
> pip install lm-eval
> lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B \
>   --tasks mmlu,hellaswag,arc_easy --batch_size 8
> ```

> **MLflow on DGX Spark:**
> ```bash
> mlflow server --host 0.0.0.0 --port 5000
> # Access at http://dgx-spark-ip:5000
> ```

### Milestone Checklist

- [ ] Benchmark results for multiple models collected
- [ ] Custom evaluation framework implemented
- [ ] MLflow tracking server running
- [ ] Model registry workflow established
- [ ] Reproducibility verified for training pipeline

---

# PHASE 4: CAPSTONE

## Module 16: Capstone Project

**Duration:** Weeks 27-32 (40-50 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Design and implement an end-to-end AI system
- Apply all techniques learned throughout the curriculum
- Document and present your work professionally
- Deploy a production-ready AI application

### Project Options

Choose ONE of the following capstone projects:

#### Option A: Domain-Specific AI Assistant
Build a complete AI assistant for a specific domain (e.g., AWS infrastructure, trading analysis, code review).

**Requirements:**
- Fine-tuned base model (70B with QLoRA)
- RAG system with domain knowledge base
- Custom tools for domain-specific operations
- Production API with streaming
- Evaluation framework
- Documentation

#### Option B: Multimodal Document Intelligence
Build a system that processes and understands complex documents (PDFs, images, diagrams).

**Requirements:**
- Document ingestion pipeline (PDF, images)
- Vision-language model integration
- Structured information extraction
- Question answering over documents
- Export to structured formats
- Evaluation on document benchmarks

#### Option C: AI Agent Swarm
Build a multi-agent system that collaborates to solve complex tasks.

**Requirements:**
- Minimum 4 specialized agents
- Agent communication protocol
- Task decomposition and planning
- Human-in-the-loop approval
- Error recovery and fallbacks
- Performance benchmarking

#### Option D: Custom Training Pipeline
Build a complete fine-tuning pipeline for continuous model improvement.

**Requirements:**
- Data collection and curation pipeline
- Multiple fine-tuning approaches (SFT, DPO)
- Automated evaluation
- Model versioning and comparison
- A/B testing framework
- Deployment automation

### Project Phases

| Phase | Duration | Activities |
|-------|----------|------------|
| **Planning** | Week 27 | Requirements, architecture design, tech selection |
| **Foundation** | Week 28-29 | Core components, data preparation, initial models |
| **Integration** | Week 30 | Component integration, API development |
| **Optimization** | Week 31 | Performance tuning, quantization, benchmarking |
| **Documentation** | Week 32 | Documentation, demo, presentation |

### Deliverables

1. **Technical Report** (15-20 pages)
   - Problem statement and motivation
   - System architecture
   - Implementation details
   - Evaluation results
   - Lessons learned

2. **Code Repository**
   - Well-organized, documented code
   - README with setup instructions
   - Requirements and environment files
   - Unit tests

3. **Demo**
   - Working demonstration
   - Video walkthrough (5-10 minutes)
   - Live demo capability

4. **Presentation**
   - Slide deck (15-20 slides)
   - Technical deep-dive sections
   - Results and impact

### Milestone Checklist

- [ ] Project proposal approved
- [ ] Architecture design documented
- [ ] Core components implemented
- [ ] Integration complete
- [ ] Evaluation results collected
- [ ] Documentation written
- [ ] Demo video created
- [ ] Presentation ready

---

# Appendices

## Appendix A: DGX Spark Quick Reference

### System Specs
| Component | Specification |
|-----------|---------------|
| CPU | 20 ARM v9.2 cores (10 Cortex-X925 + 10 Cortex-A725) |
| GPU | Blackwell (6144 CUDA cores, 192 Tensor Cores) |
| Memory | 128GB LPDDR5X unified (273 GB/s bandwidth) |
| Storage | 1TB or 4TB NVMe (PCIe Gen 5) |
| Networking | 10GbE + 200Gbps ConnectX-7 |
| Power | 140W TDP |

### Performance Benchmarks
| Model | Precision | Prefill (tok/s) | Decode (tok/s) |
|-------|-----------|-----------------|----------------|
| Llama 3.1 8B | NVFP4 | 10,257 | 38.7 |
| GPT-OSS 20B | MXFP4 | 4,500 | 59.0 |
| Llama 3.1 70B | Q4 | 800 | 15.2 |

### Common Commands
```bash
# System info
nvidia-smi
lscpu
free -h

# Clear buffer cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# NGC container
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker run --gpus all --ipc=host -v $HOME:/workspace -it nvcr.io/nvidia/pytorch:25.11-py3

# Ollama
ollama list
ollama run llama3.1:70b
```

## Appendix B: Recommended Resources

### Official Documentation
- [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [DGX Spark Playbooks](https://build.nvidia.com/spark)
- [NGC Container Catalog](https://catalog.ngc.nvidia.com/)

### Courses
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face Course](https://huggingface.co/learn)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Stanford CS224n](http://cs224n.stanford.edu/)

### Tools
- [Unsloth](https://github.com/unslothai/unsloth) - Fast LoRA training
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) - Fine-tuning GUI
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Benchmarking

## Appendix C: Progress Tracking

### Weekly Log Template
```markdown
## Week [N] - [Date Range]

### Completed
- [ ] Task 1
- [ ] Task 2

### In Progress
- Task 3 (60%)

### Challenges
- Description of challenge and resolution

### Key Learnings
- Learning point 1
- Learning point 2

### Next Week
- Planned task 1
- Planned task 2
```

### Module Completion Tracker

| Module | Status | Start Date | End Date | Notes |
|--------|--------|------------|----------|-------|
| 1. DGX Spark Platform | ⬜ | | | |
| 2. Python for AI/ML | ⬜ | | | |
| 3. Mathematics | ⬜ | | | |
| 4. Neural Network Fundamentals | ⬜ | | | |
| 5. Phase 1 Capstone | ⬜ | | | |
| 6. PyTorch Deep Learning | ⬜ | | | |
| 7. Computer Vision | ⬜ | | | |
| 8. NLP & Transformers | ⬜ | | | |
| 9. Hugging Face Ecosystem | ⬜ | | | |
| 10. LLM Fine-Tuning | ⬜ | | | |
| 11. Quantization & Optimization | ⬜ | | | |
| 12. Deployment & Inference | ⬜ | | | |
| 13. AI Agents | ⬜ | | | |
| 14. Multimodal AI | ⬜ | | | |
| 15. Benchmarking & MLOps | ⬜ | | | |
| 16. Capstone Project | ⬜ | | | |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial curriculum release |

---

*This curriculum is optimized for NVIDIA DGX Spark with 128GB unified memory. Adjust batch sizes and model sizes if using different hardware.*