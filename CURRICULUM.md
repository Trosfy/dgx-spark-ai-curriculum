# NVIDIA DGX Spark AI Mastery Curriculum

## Curriculum Overview

| Attribute | Details |
|-----------|---------|
| **Target Hardware** | NVIDIA DGX Spark (GB10 Superchip, 128GB Unified Memory) |
| **Duration** | 32-40 weeks (standard pace) / 16-20 weeks (intensive) |
| **Prerequisites** | Basic Python knowledge, high school mathematics |
| **Primary Environment** | JupyterLab (native on DGX Spark) |
| **Testing Platform** | Your custom Ollama Web UI |

---

## What's New in

Based on comprehensive curriculum research, adds significant new content organized by priority:

### Critical Additions
| Addition | Duration | Rationale |
|----------|----------|-----------|
| **CUDA Python & GPU Programming** | 2 weeks | Essential for DGX Spark hardware mastery |
| **NVFP4/FP8 Quantization Workflows** | Expanded | Blackwell's key differentiator |
| **RAG Systems & Vector Databases** | 2 weeks | Industry-demanded skill (majority of LLM job postings) |
| **Docker & Containerization** | 1.5 weeks | Production deployment prerequisite |
| **AI Safety & Alignment** | 2 weeks | Regulatory compliance (EU AI Act), industry demand |
| **Experiment Tracking (MLflow/W&B)** | Expanded | Industry-standard MLOps |

### High Priority Additions
| Addition | Duration | Rationale |
|----------|----------|-----------|
| **State Space Models (Mamba)** | 1.5 weeks | Linear memory scaling, leverages 128GB unified memory |
| **Mixture of Experts (MoE)** | 1 week | Top 10 open models use MoE; DeepSeekMoE accessible |
| **Modern Inference Engines (SGLang)** | Expanded | 29-45% faster than vLLM |
| **Speculative Decoding (Medusa)** | 1 week | 2-3x interactive speedup |
| **Modern Fine-Tuning (DoRA, NEFTune, SimPO, ORPO)** | Expanded | High-impact improvements |
| **Test-Time Compute & Reasoning** | 1 week | DeepSeek-R1 techniques |
| **Diffusion Models (Stable Diffusion, Flux)** | 2 weeks | Fast.ai gap, growing demand |
| **Cloud Deployment (AWS/GCP)** | 1.5 weeks | 35% of ML job postings require AWS |
| **Model Monitoring & Drift Detection** | Expanded | Production readiness |

### Medium Priority Additions
| Addition | Duration | Rationale |
|----------|----------|-----------|
| **Classical ML Overview (XGBoost, RF)** | 1 week | Baseline comparison, 17-26% job mention rate |
| **Object Detection (YOLO, Faster R-CNN)** | Expanded | Complete CV coverage |
| **Vision Transformers (ViT, DeiT)** | Expanded | Modern CV architectures |
| **Kubernetes Basics for ML** | 0.5 weeks | 58% of orgs use K8s for AI |
| **Tokenizer Training from Scratch** | Expanded | Deep NLP understanding |
| **Gradio/Streamlit Demo Building** | 0.5 weeks | Rapid prototyping |
| **KTO (Kahneman-Tversky Optimization)** | Expanded | Binary feedback scenarios |

### Optional Enhancements
| Addition | Duration | Rationale |
|----------|----------|-----------|
| **Learning Theory Foundations** | Optional | VC dimension, bias-variance tradeoff |
| **Recommender Systems** | Optional | Specialized application domain |
| **Mechanistic Interpretability** | Optional | Advanced research topic |
| **Reinforcement Learning Fundamentals** | Optional | Foundation for RLHF understanding |
| **Graph Neural Networks** | Optional | Specialized architecture |

---

## Curriculum Structure

```
DOMAIN 1: PLATFORM FOUNDATIONS (Weeks 1-7)
├── Module 1.1: DGX Spark Platform Mastery
├── Module 1.2: Python for AI/ML
├── Module 1.3: CUDA Python & GPU Programming
├── Module 1.4: Mathematics for Deep Learning
├── Module 1.5: Neural Network Fundamentals
├── Module 1.6: Classical ML Foundations
└── Module 1.7: Capstone — MicroGrad+

DOMAIN 2: DEEP LEARNING FRAMEWORKS (Weeks 8-15)
├── Module 2.1: Deep Learning with PyTorch
├── Module 2.2: Computer Vision (expanded: ViT, YOLO)
├── Module 2.3: NLP & Transformers (expanded: Tokenizer Training)
├── Module 2.4: Efficient Architectures (Mamba, MoE)
├── Module 2.5: Hugging Face Ecosystem
└── Module 2.6: Diffusion Models

DOMAIN 3: LLM SYSTEMS (Weeks 16-26)
├── Module 3.1: LLM Fine-Tuning (DoRA, NEFTune, SimPO, ORPO, KTO)
├── Module 3.2: Quantization & Optimization (NVFP4, FP8)
├── Module 3.3: Deployment & Inference (SGLang, Speculative Decoding)
├── Module 3.4: Test-Time Compute & Reasoning
├── Module 3.5: RAG Systems & Vector Databases
└── Module 3.6: AI Agents & Agentic Systems

DOMAIN 4: PRODUCTION AI (Weeks 27-40)
├── Module 4.1: Multimodal AI
├── Module 4.2: AI Safety & Alignment
├── Module 4.3: MLOps & Experiment Tracking 
├── Module 4.4: Containerization & Deployment  (Docker, K8s, Cloud)
├── Module 4.5: Demo Building & Prototyping
└── Module 4.6: Capstone Project

OPTIONAL MODULES
├── Optional A: Learning Theory Deep Dive
├── Optional B: Recommender Systems
├── Optional C: Mechanistic Interpretability
├── Optional D: Reinforcement Learning Fundamentals
└── Optional E: Graph Neural Networks
```

---

## DGX Spark Model Capacity Matrix

Understanding what's possible on your 128GB unified memory system:

| Scenario | Maximum Model Size | Memory Usage | Notes |
|----------|-------------------|--------------|-------|
| Full Fine-Tuning (FP16) | **12-16B** | ~100-128GB | With gradient checkpointing |
| QLoRA Fine-Tuning | **100-120B** | ~50-70GB | 4-bit quantized + adapters |
| FP16 Inference | **50-55B** | ~110-120GB | Including KV cache headroom |
| FP8 Inference | **90-100B** | ~90-100GB | Native Blackwell support |
| **NVFP4 Inference** | **~200B** | ~100GB | Blackwell exclusive |
| Dual Spark (256GB) FP4 | **~405B** | ~200GB | Model parallelism via NVLink |

---

# DOMAIN 1: PLATFORM FOUNDATIONS

## Module 1.1: DGX Spark Platform Mastery

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
| 1.1.1 | Describe the Grace Blackwell GB10 architecture including CPU cores, GPU specs, and unified memory | Understand |
| 1.1.2 | Execute system monitoring commands to verify GPU and memory status | Apply |
| 1.1.3 | Configure NGC containers for PyTorch and other frameworks | Apply |
| 1.1.4 | Differentiate between compatible and incompatible open-source tools | Analyze |

### Topics

1. **Hardware Architecture Deep-Dive**
   - Grace Blackwell GB10 superchip (20 ARM cores, 6144 CUDA cores, 192 Tensor Cores)
   - 128GB LPDDR5X unified memory architecture
   - NVLink-C2C interconnect (273 GB/s bandwidth)
   - FP4/FP8/BF16/FP32 compute capabilities
   - Comparison with consumer GPUs (RTX 4090, 5090)

2. **Software Environment**
   - DGX OS (Ubuntu 24.04 LTS)
   - CUDA 13.0.2, cuDNN, TensorRT
   - Pre-installed JupyterLab configuration
   - NGC container ecosystem

3. **Ecosystem Compatibility**
   - Fully compatible: Ollama, llama.cpp, NeMo, SGLang
   - NGC required: PyTorch, JAX, Hugging Face
   - Partial support: vLLM (--enforce-eager), TensorRT-LLM
   - RAPIDS/cuML for data acceleration

4. **Dual DGX Spark Configuration** (Optional)
   - NVLink connection for 256GB combined memory
   - Model parallelism setup
   - Distributed training basics

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 1.1.1 | **System Exploration** | Run `nvidia-smi`, `lscpu`, `free -h` and document your system specs. Create a Jupyter notebook summarizing your DGX Spark configuration. | 1 hour |
| 1.1.2 | **Memory Architecture Lab** | Write a Python script that allocates tensors of increasing size, monitoring memory with `torch.cuda.memory_summary()`. Document when CPU vs GPU memory is used. | 1.5 hours |
| 1.1.3 | **NGC Container Setup** | Pull and configure the PyTorch NGC container. Create a docker-compose.yml that mounts your home directory and Hugging Face cache. | 1.5 hours |
| 1.1.4 | **Compatibility Matrix** | Research and create a markdown table of 20 popular AI tools with their DGX Spark compatibility status and workarounds. | 2 hours |
| 1.1.5 | **Ollama Integration** | Configure Ollama, pull 3 different model sizes (7B, 13B, 70B), benchmark each with your Web UI, document performance. | 2 hours |

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

### Milestone Checklist

- [ ] Successfully ran nvidia-smi showing GB10 GPU
- [ ] Created system specification notebook
- [ ] NGC PyTorch container running with GPU access
- [ ] Ollama serving models through your Web UI
- [ ] Completed compatibility matrix document
- [ ] Documented unified memory behavior with tensor allocation test

---

## Module 1.2: Python for AI/ML

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
| 1.2.1 | Implement vectorized operations using NumPy broadcasting | Apply |
| 1.2.2 | Transform and clean datasets using Pandas operations | Apply |
| 1.2.3 | Create multi-panel visualizations for model analysis | Create |
| 1.2.4 | Profile Python code and identify performance bottlenecks | Analyze |

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

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 1.2.1 | **NumPy Broadcasting Lab** | Implement matrix operations (batch matrix multiply, outer products) using broadcasting. Compare loop vs vectorized performance. | 2 hours |
| 1.2.2 | **Dataset Preprocessing Pipeline** | Download a real dataset (e.g., Titanic, Housing), implement complete preprocessing pipeline. | 2 hours |
| 1.2.3 | **Visualization Dashboard** | Create a multi-panel figure showing training curves, confusion matrix, feature importance, prediction distribution. | 2 hours |
| 1.2.4 | **Einsum Mastery** | Implement attention mechanism using `np.einsum`. Compare with explicit loop and matmul versions. | 2 hours |
| 1.2.5 | **Profiling Exercise** | Profile a slow function, identify bottlenecks, optimize with vectorization/Numba. Achieve 10x+ speedup. | 2 hours |

### Guidance

> **Einsum Tip:** Master einsum notation early—it's essential for understanding attention mechanisms:
> ```python
> # Batch matrix multiply: (B, M, K) @ (B, K, N) -> (B, M, N)
> np.einsum('bmk,bkn->bmn', A, B)
> ```

### Milestone Checklist

- [ ] Vectorized batch operations notebook complete
- [ ] Reusable preprocessing pipeline class created
- [ ] Multi-panel visualization dashboard created
- [ ] Einsum attention implementation working
- [ ] Achieved 10x+ speedup in profiling exercise

---

## Module 1.3: CUDA Python & GPU Programming

**Duration:** Week 3 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Explain GPU architecture and parallel computing principles
- Write CUDA kernels using Numba and CuPy
- Optimize memory access patterns for GPU performance
- Profile and debug GPU code using NVIDIA tools

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.3.1 | Explain GPU memory hierarchy (global, shared, registers) | Understand |
| 1.3.2 | Write parallel algorithms using CUDA Python (Numba) | Apply |
| 1.3.3 | Optimize memory coalescing and reduce bank conflicts | Apply |
| 1.3.4 | Profile GPU code with Nsight Systems/Compute | Analyze |

### Topics

1. **GPU Architecture Fundamentals**
   - SIMT execution model
   - Streaming Multiprocessors (SMs)
   - Warp execution and divergence
   - Blackwell-specific features (Tensor Cores, FP4 units)

2. **Memory Hierarchy**
   - Global memory and coalescing
   - Shared memory and bank conflicts
   - Constant and texture memory
   - Unified memory on DGX Spark

3. **CUDA Python with Numba**
   - @cuda.jit decorator
   - Thread indexing and grid configuration
   - Synchronization primitives
   - Atomic operations

4. **CuPy for Array Operations**
   - NumPy-compatible GPU arrays
   - Custom kernels with RawKernel
   - Memory management
   - Interoperability with PyTorch

5. **Profiling and Optimization**
   - Nsight Systems for timeline analysis
   - Nsight Compute for kernel profiling
   - Occupancy optimization
   - Common performance pitfalls

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 1.3.1 | **Parallel Reduction** | Implement parallel sum reduction with progressively optimized versions (naive → shared memory → warp shuffle). Measure speedup at each stage. | 3 hours |
| 1.3.2 | **Matrix Multiplication** | Implement tiled matrix multiplication using shared memory. Compare with cuBLAS. Document speedup vs naive implementation. | 3 hours |
| 1.3.3 | **Custom Embedding Lookup** | Write a CUDA kernel for batched embedding lookup (common in LLMs). Optimize memory access patterns. | 2 hours |
| 1.3.4 | **CuPy Integration** | Port NumPy preprocessing pipeline to CuPy. Measure speedup on large datasets. | 2 hours |
| 1.3.5 | **Profiling Workshop** | Profile a PyTorch training loop with Nsight Systems. Identify bottlenecks, optimize data loading/transfers. | 2 hours |

### Guidance

> **DGX Spark Specifics:** The unified memory architecture means CPU↔GPU transfers are faster than discrete GPUs, but understanding access patterns still matters for Tensor Core utilization.

> **Thread Configuration:**
> ```python
> from numba import cuda
> 
> @cuda.jit
> def kernel(data):
>     idx = cuda.grid(1)
>     if idx < data.size:
>         data[idx] *= 2
> 
> # Launch with appropriate grid size
> threads_per_block = 256
> blocks = (data.size + threads_per_block - 1) // threads_per_block
> kernel[blocks, threads_per_block](data)
> ```

### Milestone Checklist

- [ ] Parallel reduction achieving >100x speedup vs CPU
- [ ] Tiled matrix multiplication within 2x of cuBLAS
- [ ] Custom embedding kernel working correctly
- [ ] CuPy pipeline with measured speedup
- [ ] Nsight profiling report with optimization recommendations

---

## Module 1.4: Mathematics for Deep Learning

**Duration:** Week 4 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Compute gradients for neural network operations manually and verify with autograd
- Explain and implement common optimization algorithms
- Interpret loss landscapes and understand convergence behavior
- Apply linear algebra concepts to neural network operations

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.4.1 | Compute partial derivatives for composite functions using chain rule | Apply |
| 1.4.2 | Implement gradient descent variants from scratch | Apply |
| 1.4.3 | Visualize and interpret loss landscapes | Analyze |
| 1.4.4 | Perform matrix calculus for backpropagation derivation | Understand |

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
   - Learning rate schedules (cosine, warmup)
   - Loss landscape geometry

4. **Probability for ML**
   - Probability distributions (Gaussian, Categorical)
   - Maximum likelihood estimation
   - Cross-entropy and KL divergence
   - Bayesian basics

5. **Learning Theory Foundations**
   - Bias-variance tradeoff
   - VC dimension concepts
   - Generalization bounds intuition
   - PAC learning basics

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 1.4.1 | **Manual Backprop** | Implement forward and backward pass for a 3-layer MLP from scratch (no autograd). Verify gradients match PyTorch autograd. | 3 hours |
| 1.4.2 | **Optimizer Implementation** | Implement SGD, Momentum, and Adam optimizers from scratch. Train on a simple problem, compare convergence curves. | 2 hours |
| 1.4.3 | **Loss Landscape Visualization** | Create 2D and 3D visualizations of loss landscapes for a simple network. Identify local minima, saddle points. | 2 hours |
| 1.4.4 | **SVD for LoRA Intuition** | Decompose a weight matrix using SVD, reconstruct with varying ranks, visualize information loss. | 2 hours |
| 1.4.5 | **Probability Distributions Lab** | Implement and visualize common distributions. Derive cross-entropy loss from maximum likelihood principle. | 2 hours |

### Guidance

> **SVD Connection to LoRA:** Understanding that any matrix W can be decomposed as W = UΣV^T helps you understand why LoRA's low-rank adaptation (W + BA) works—we're adding a low-rank perturbation.

### Milestone Checklist

- [ ] Manual backprop implementation matches autograd within 1e-6
- [ ] Three optimizers implemented and compared
- [ ] Loss landscape visualizations created
- [ ] SVD decomposition notebook complete
- [ ] Cross-entropy derivation documented

---

## Module 1.5: Neural Network Fundamentals

**Duration:** Week 5 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build neural networks from scratch using only NumPy
- Explain the purpose of each neural network component
- Train networks on real datasets and diagnose common issues
- Implement regularization and normalization techniques

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.5.1 | Implement fully-connected layers with forward and backward passes | Apply |
| 1.5.2 | Explain vanishing/exploding gradients and implement solutions | Understand |
| 1.5.3 | Apply regularization techniques (L2, dropout) to prevent overfitting | Apply |
| 1.5.4 | Diagnose training issues from loss curves and metrics | Analyze |

### Topics

1. **Perceptron to MLP**
   - Single neuron and activation functions
   - Multi-layer perceptrons
   - Universal approximation theorem

2. **Activation Functions**
   - Sigmoid, Tanh, ReLU, GELU, SiLU/Swish
   - Vanishing gradient problem
   - Choosing activations for different tasks

3. **Loss Functions**
   - MSE for regression
   - Cross-entropy for classification
   - Focal loss for imbalanced data
   - Custom losses

4. **Regularization**
   - L1/L2 regularization
   - Dropout and DropPath
   - Early stopping
   - Data augmentation concepts

5. **Normalization**
   - Batch normalization
   - Layer normalization
   - RMSNorm (for transformers)
   - Group normalization

6. **Weight Initialization**
   - Xavier/Glorot initialization
   - He/Kaiming initialization
   - Impact on training dynamics

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 1.5.1 | **NumPy Neural Network** | Build a complete MLP from scratch: Linear, ReLU, Softmax, CrossEntropy, SGD. Train on MNIST to >95% accuracy. | 4 hours |
| 1.5.2 | **Activation Function Study** | Implement 6 activation functions, visualize their outputs and gradients, train same network with each. | 2 hours |
| 1.5.3 | **Regularization Experiments** | Train networks with varying L2 strength and dropout rates. Visualize underfitting → good fit → overfitting. | 2 hours |
| 1.5.4 | **Normalization Comparison** | Implement BatchNorm and LayerNorm from scratch. Compare training dynamics. | 2 hours |
| 1.5.5 | **Training Diagnostics Lab** | Deliberately create training problems, document symptoms, implement fixes. | 2 hours |

### Guidance

> **Learning Rate Selection:** Start with 1e-3 for Adam, 1e-2 for SGD with momentum. If loss explodes, reduce by 10x.

### Milestone Checklist

- [ ] NumPy MLP achieving >95% on MNIST
- [ ] Activation function comparison notebook complete
- [ ] Regularization experiments documented
- [ ] BatchNorm and LayerNorm implementations working
- [ ] Training diagnostics guide created

---

## Module 1.6: Classical ML Foundations

**Duration:** Week 6 (6-8 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Apply classical ML algorithms as baselines for comparison
- Explain when classical ML outperforms deep learning
- Use scikit-learn and XGBoost effectively
- Accelerate classical ML with RAPIDS cuML on DGX Spark

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.6.1 | Train and evaluate tree-based models (Random Forest, XGBoost) | Apply |
| 1.6.2 | Explain bias-variance tradeoff in classical ML context | Understand |
| 1.6.3 | Perform hyperparameter tuning with cross-validation | Apply |
| 1.6.4 | Accelerate classical ML with RAPIDS cuML | Apply |

### Topics

1. **Tree-Based Methods**
   - Decision trees and information gain
   - Random Forests and bagging
   - Gradient Boosting (XGBoost, LightGBM, CatBoost)
   - Feature importance analysis

2. **Linear Models**
   - Logistic regression
   - Support Vector Machines
   - Regularized regression (Ridge, Lasso)

3. **Model Selection**
   - Cross-validation strategies
   - Hyperparameter tuning (GridSearch, Optuna)
   - When to use classical ML vs deep learning

4. **GPU Acceleration with RAPIDS**
   - cuML drop-in replacements for scikit-learn
   - cuDF for data preprocessing
   - Performance comparison CPU vs GPU

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 1.6.1 | **Tabular Data Challenge** | Train XGBoost on a tabular dataset. Compare with simple neural network. Document when each excels. | 2 hours |
| 1.6.2 | **Hyperparameter Optimization** | Use Optuna to tune XGBoost hyperparameters. Visualize optimization history and feature importance. | 2 hours |
| 1.6.3 | **RAPIDS Acceleration** | Port scikit-learn pipeline to cuML. Benchmark on large dataset (1M+ rows). | 2 hours |
| 1.6.4 | **Baseline Comparison Framework** | Create reusable framework that trains classical ML baseline before any deep learning experiment. | 2 hours |

### Guidance

> **When to Use Classical ML:**
> - Tabular data with <100K samples: XGBoost often wins
> - Interpretability required: Decision trees, linear models
> - Limited compute: Classical ML trains in seconds
> - Always train as baseline before deep learning!

> **RAPIDS on DGX Spark:**
> ```python
> import cudf
> import cuml
> from cuml.ensemble import RandomForestClassifier
> 
> # GPU-accelerated Random Forest
> rf = RandomForestClassifier(n_estimators=100)
> rf.fit(X_train_gpu, y_train_gpu)  # 10-100x faster than scikit-learn
> ```

### Milestone Checklist

- [ ] XGBoost model trained and evaluated
- [ ] Optuna hyperparameter optimization complete
- [ ] RAPIDS cuML benchmark showing speedup
- [ ] Baseline comparison framework created
- [ ] Documentation of when to use classical vs deep learning

---

## Module 1.7: Capstone — MicroGrad+

**Duration:** Week 7 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Design and implement a modular neural network library
- Create reusable, well-documented code following software engineering practices
- Benchmark your implementation against PyTorch

### Capstone Project: MicroGrad+

Build an extended version of Andrej Karpathy's micrograd with additional features:

**Required Components:**
1. Tensor class with automatic differentiation
2. Layer classes: Linear, Conv2D (basic), ReLU, Softmax, LayerNorm
3. Loss functions: MSE, CrossEntropy
4. Optimizers: SGD, Adam
5. Training loop abstraction
6. GPU support via CuPy (optional but recommended)

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

# DOMAIN 2: DEEP LEARNING FRAMEWORKS

## Module 2.1: Deep Learning with PyTorch

**Duration:** Weeks 8-9 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build complex neural networks using PyTorch's nn.Module
- Implement custom datasets and data loaders
- Utilize PyTorch's autograd for custom operations
- Debug and profile PyTorch models effectively

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.1.1 | Create custom nn.Module classes with proper initialization | Apply |
| 2.1.2 | Implement Dataset and DataLoader for custom data | Apply |
| 2.1.3 | Use hooks for model introspection and debugging | Apply |
| 2.1.4 | Profile models with PyTorch Profiler | Analyze |

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
   - Learning rate scheduling (cosine, warmup)
   - Gradient clipping and accumulation

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 2.1.1 | **Custom Module Lab** | Implement a ResNet block as custom nn.Module. Stack blocks to create ResNet-18. | 2 hours |
| 2.1.2 | **Dataset Pipeline** | Create custom Dataset for a local image folder. Implement DataLoader with augmentation. | 2 hours |
| 2.1.3 | **Autograd Deep Dive** | Implement custom autograd Function for a novel activation. Verify gradients. | 2 hours |
| 2.1.4 | **Mixed Precision Training** | Train a model with AMP. Compare memory usage and speed vs FP32. | 2 hours |
| 2.1.5 | **Profiling Workshop** | Profile a training loop with PyTorch Profiler. Identify and fix bottlenecks. | 2 hours |
| 2.1.6 | **Checkpointing System** | Implement robust checkpointing: save/resume training, best model tracking, early stopping. | 2 hours |

### Milestone Checklist

- [ ] ResNet-18 implemented from scratch
- [ ] Custom dataset pipeline working
- [ ] Custom autograd function with verified gradients
- [ ] AMP training notebook with memory comparison
- [ ] Profiling report with optimization applied
- [ ] Checkpointing system tested

---

## Module 2.2: Computer Vision (Expanded)

**Duration:** Weeks 10-11 (14-16 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Implement and train CNN architectures for image classification
- Apply transfer learning for custom image tasks
- Perform object detection using YOLO and Faster R-CNN
- Understand and implement Vision Transformers (ViT)

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.2.1 | Explain the evolution from LeNet to modern architectures | Understand |
| 2.2.2 | Train and deploy YOLO for object detection | Apply |
| 2.2.3 | Implement Vision Transformer (ViT) from scratch | Apply |
| 2.2.4 | Fine-tune pre-trained models on custom datasets | Apply |

### Topics

1. **CNN Architectures**
   - LeNet, AlexNet, VGG evolution
   - ResNet and skip connections
   - Modern architectures: EfficientNet, ConvNeXt

2. **Transfer Learning**
   - Pre-trained model selection
   - Feature extraction vs fine-tuning
   - Learning rate strategies

3. **Object Detection**
   - Region-based methods (R-CNN family, Faster R-CNN)
   - Single-shot detectors (YOLO family, SSD)
   - Anchor-free detectors (FCOS, CenterNet)
   - Using YOLOv8/YOLOv11 on DGX Spark

4. **Image Segmentation**
   - Semantic vs instance segmentation
   - U-Net architecture
   - Segment Anything Model (SAM)

5. **Vision Transformers**
   - ViT architecture and patch embeddings
   - Positional embeddings for images
   - DeiT training tricks
   - Swin Transformer (hierarchical)
   - Comparison with CNNs

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 2.2.1 | **CNN Architecture Study** | Implement LeNet and ResNet-18. Train on CIFAR-10, compare accuracy and training dynamics. | 2 hours |
| 2.2.2 | **Transfer Learning Project** | Fine-tune EfficientNet on a custom dataset. Achieve >90% accuracy. | 2 hours |
| 2.2.3 | **YOLO Object Detection** | Train YOLOv8 on custom objects. Create real-time detection demo. Document inference speed on DGX Spark. | 3 hours |
| 2.2.4 | **Faster R-CNN Lab** | Use torchvision Faster R-CNN for detection. Compare accuracy vs speed with YOLO. | 2 hours |
| 2.2.5 | **Vision Transformer from Scratch** | Implement ViT from scratch. Train on CIFAR-10, compare with CNN. | 3 hours |
| 2.2.6 | **SAM Integration** | Use Segment Anything Model for interactive segmentation. Create demo notebook. | 2 hours |

### Milestone Checklist

- [ ] CNN architectures implemented and compared
- [ ] Transfer learning project achieving >90% accuracy
- [ ] YOLOv8 object detection working with custom training
- [ ] Faster R-CNN comparison documented
- [ ] Vision Transformer implemented from scratch
- [ ] SAM demo notebook complete

---

## Module 2.3: NLP & Transformers (Expanded)

**Duration:** Week 12 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Implement the Transformer architecture from scratch
- Train custom tokenizers using BPE and SentencePiece
- Fine-tune language models for downstream tasks
- Explain attention mechanisms and their variations

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.3.1 | Implement multi-head self-attention from scratch | Apply |
| 2.3.2 | Train custom tokenizers from scratch | Apply |
| 2.3.3 | Explain positional encoding strategies (sinusoidal, RoPE, ALiBi) | Understand |
| 2.3.4 | Fine-tune BERT and GPT models for downstream tasks | Apply |

### Topics

1. **Attention Mechanisms**
   - Scaled dot-product attention
   - Multi-head attention
   - Cross-attention
   - Attention visualization

2. **Transformer Architecture**
   - Encoder and decoder blocks
   - Layer normalization placement (Pre-LN vs Post-LN)
   - Feed-forward networks (and GLU variants)
   - Residual connections

3. **Positional Encodings**
   - Sinusoidal encoding
   - Learned embeddings
   - Rotary Position Embeddings (RoPE)
   - ALiBi (Attention with Linear Biases)

4. **Tokenization**
   - Word-level vs subword tokenization
   - BPE algorithm implementation from scratch
   - SentencePiece and HuggingFace tokenizers
   - Training tokenizers on custom corpora
   - Vocabulary size tradeoffs

5. **Pre-trained Models**
   - BERT and masked language modeling
   - GPT and causal language modeling
   - T5 and encoder-decoder models
   - Modern architectures: Llama, Mistral

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 2.3.1 | **Attention from Scratch** | Implement scaled dot-product and multi-head attention. Visualize attention patterns. | 2 hours |
| 2.3.2 | **Transformer Block** | Implement complete Transformer encoder block. Stack 6 blocks to create encoder. | 2 hours |
| 2.3.3 | **Positional Encoding Study** | Implement sinusoidal and RoPE encodings. Visualize and compare properties. | 2 hours |
| 2.3.4 | **Tokenizer Training from Scratch** | Implement BPE algorithm from scratch. Train on custom corpus. Compare with HuggingFace tokenizer. | 3 hours |
| 2.3.5 | **BERT Fine-tuning** | Fine-tune BERT for sentiment classification. Analyze errors. | 2 hours |
| 2.3.6 | **GPT Text Generation** | Load GPT-2, implement greedy, beam search, and sampling decoding. | 2 hours |

### Milestone Checklist

- [ ] Multi-head attention implementation complete
- [ ] Full Transformer encoder working
- [ ] Both positional encoding types implemented
- [ ] Custom BPE tokenizer trained from scratch
- [ ] BERT fine-tuning achieving good accuracy
- [ ] Text generation with multiple decoding strategies

---

## Module 2.4: Efficient Architectures

**Duration:** Week 13 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Explain State Space Models (Mamba) and their advantages
- Understand Mixture of Experts (MoE) architecture
- Compare transformer vs alternative architectures
- Run and fine-tune efficient architecture models on DGX Spark

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.4.1 | Explain selective state space mechanism in Mamba | Understand |
| 2.4.2 | Describe MoE architecture with gating and load balancing | Understand |
| 2.4.3 | Compare memory/compute tradeoffs of different architectures | Analyze |
| 2.4.4 | Run inference and fine-tune Mamba/MoE models | Apply |

### Topics

1. **Limitations of Transformers**
   - Quadratic attention complexity O(n²)
   - KV cache memory growth
   - Computational bottlenecks for long sequences

2. **State Space Models (Mamba)**
   - Linear time complexity O(n)
   - Selective state space mechanism
   - Hardware-aware parallel algorithms
   - No KV cache (constant memory per token)
   - Mamba-2 improvements

3. **Mixture of Experts (MoE)**
   - Sparse activation patterns
   - Router/gating mechanisms (top-k)
   - Load balancing losses
   - Expert parallelism
   - DeepSeekMoE, Mixtral architectures

4. **Hybrid Architectures**
   - Jamba (Mamba + Attention)
   - IBM Granite 4.0 architecture
   - When to use hybrid approaches

5. **Efficient Attention Variants**
   - Flash Attention
   - Linear attention
   - Grouped Query Attention (GQA)
   - Multi-Query Attention (MQA)

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 2.4.1 | **Mamba Inference** | Run Mamba-2.8B on DGX Spark. Compare memory usage and throughput vs same-size transformer. Document advantages. | 2 hours |
| 2.4.2 | **Mamba Architecture Study** | Study Mamba's selective scan algorithm. Implement simplified version. Visualize state evolution. | 2 hours |
| 2.4.3 | **MoE Exploration** | Run DeepSeekMoE-16B on DGX Spark. Analyze which experts activate for different prompts. | 2 hours |
| 2.4.4 | **MoE Router Analysis** | Visualize expert selection patterns. Understand load balancing. | 2 hours |
| 2.4.5 | **Architecture Comparison** | Benchmark Mamba vs Transformer vs MoE on same task (perplexity, speed, memory). Create comparison report. | 2 hours |
| 2.4.6 | **Mamba Fine-tuning** | Fine-tune small Mamba model on custom dataset using LoRA. | 2 hours |

### Guidance

> **Why Mamba on DGX Spark?** Mamba's linear memory scaling means the 128GB unified memory can support much longer contexts than transformers. A 70B transformer might max out at 32K context; Mamba can handle 100K+.

> **MoE Memory:** DeepSeekMoE-16B has 16B total parameters but only ~2.5B active per token. This makes inference very efficient despite the large total size.

### Milestone Checklist

- [ ] Mamba inference running with benchmark data
- [ ] Selective scan understanding documented
- [ ] MoE expert activation patterns visualized
- [ ] Architecture comparison report complete
- [ ] Mamba fine-tuning demonstrated
- [ ] Clear understanding of when to use each architecture

---

## Module 2.5: Hugging Face Ecosystem

**Duration:** Week 14 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Navigate and utilize the Hugging Face Hub effectively
- Use Transformers library for various NLP tasks
- Load and preprocess datasets with the Datasets library
- Apply the Trainer API for efficient model training

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.5.1 | Load and use pre-trained models from Hugging Face Hub | Apply |
| 2.5.2 | Preprocess datasets using datasets library | Apply |
| 2.5.3 | Configure and use the Trainer API | Apply |
| 2.5.4 | Use PEFT library for parameter-efficient fine-tuning | Apply |

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
   - LoRA configuration basics
   - Merging adapters

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 2.5.1 | **Hub Exploration** | Find 10 interesting models, document use cases, test 3 locally. | 2 hours |
| 2.5.2 | **Pipeline Showcase** | Demonstrate 5 different pipelines: text-generation, sentiment, NER, QA, summarization. | 2 hours |
| 2.5.3 | **Dataset Processing** | Load large dataset (>1M samples), apply preprocessing with map(), create splits. | 2 hours |
| 2.5.4 | **Trainer Fine-tuning** | Use Trainer API for text classification. Implement custom metrics callback. | 2 hours |
| 2.5.5 | **LoRA Introduction** | Apply LoRA to a small model using PEFT. Compare memory usage vs full fine-tuning. | 2 hours |
| 2.5.6 | **Model Upload** | Fine-tune a model, create model card, upload to HF Hub. | 2 hours |

### Milestone Checklist

- [ ] 10 models documented from HF Hub
- [ ] 5 pipeline demonstrations complete
- [ ] Large dataset processing pipeline working
- [ ] Trainer fine-tuning complete
- [ ] LoRA fine-tuning comparison documented
- [ ] Model uploaded to HF Hub with model card

---

## Module 2.6: Diffusion Models

**Duration:** Week 15 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Explain the theory behind diffusion models
- Generate images using Stable Diffusion and Flux
- Apply ControlNet for guided generation
- Fine-tune diffusion models with LoRA for style transfer

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 2.6.1 | Explain forward and reverse diffusion processes | Understand |
| 2.6.2 | Generate images with various guidance techniques | Apply |
| 2.6.3 | Use ControlNet for structured image generation | Apply |
| 2.6.4 | Train LoRA for custom styles | Apply |

### Topics

1. **Diffusion Model Theory**
   - Forward diffusion (adding noise)
   - Reverse diffusion (denoising)
   - Score matching and DDPM
   - Latent diffusion (efficiency)

2. **Stable Diffusion**
   - Architecture: VAE, U-Net, CLIP text encoder
   - Prompt engineering for better outputs
   - Negative prompts
   - Classifier-free guidance

3. **Advanced Generation**
   - ControlNet for pose, edge, depth control
   - IP-Adapter for image prompting
   - Inpainting and outpainting
   - Image-to-image generation

4. **Modern Architectures**
   - SDXL improvements (legacy)
   - FLUX.2 architecture (32B, current SOTA)
   - DiT (Diffusion Transformers)

5. **Fine-tuning Diffusion Models**
   - LoRA for style transfer
   - DreamBooth for subject learning
   - Textual inversion

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 2.6.1 | **Diffusion Theory Notebook** | Implement simple DDPM on MNIST. Visualize forward/reverse process step by step. | 2 hours |
| 2.6.2 | **Stable Diffusion Generation** | Run SDXL on DGX Spark. Experiment with prompts, negative prompts, guidance scale. Document quality vs speed. | 2 hours |
| 2.6.3 | **ControlNet Workshop** | Use ControlNet with pose, canny, and depth. Create consistent character generations. | 2 hours |
| 2.6.4 | **FLUX.2 Exploration** | Run FLUX.2 dev on DGX Spark. Compare quality with SDXL. Use inpainting/outpainting. | 2 hours |
| 2.6.5 | **LoRA Style Training** | Train a LoRA on ~20 style images. Generate images in your custom style. | 2 hours |
| 2.6.6 | **Image Generation Pipeline** | Build end-to-end pipeline: text prompt → image → upscaling → variations. | 2 hours |

### Guidance

> **DGX Spark Performance:** SDXL generates 1024x1024 images in ~5-8 seconds. FLUX.2 dev is slower (~15-20s) but significantly higher quality with better text rendering and multi-reference consistency.

> **Memory for Diffusion:** SDXL requires ~7GB VRAM minimum. With 128GB unified memory, you can run the largest models and multiple LoRAs simultaneously.

### Milestone Checklist

- [ ] Simple DDPM implementation complete
- [ ] SDXL generation with various techniques
- [ ] ControlNet demos working
- [ ] FLUX.2 comparison documented
- [ ] Custom LoRA style trained
- [ ] End-to-end generation pipeline created

---

# DOMAIN 3: LLM SYSTEMS

## Module 3.1: LLM Fine-Tuning

**Duration:** Weeks 16-18 (18-22 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Fine-tune LLMs using LoRA, QLoRA, DoRA, and full fine-tuning
- Apply modern alignment techniques (DPO, SimPO, ORPO, KTO)
- Prepare datasets for instruction tuning and preference learning
- Fine-tune 70B+ models on DGX Spark

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.1.1 | Explain mathematical foundations of LoRA and DoRA | Understand |
| 3.1.2 | Configure and execute QLoRA fine-tuning for 70B models | Apply |
| 3.1.3 | Implement preference optimization with DPO, SimPO, ORPO, KTO | Apply |
| 3.1.4 | Apply NEFTune for improved fine-tuning | Apply |

### Topics

1. **Fine-Tuning Strategies**
   - Full fine-tuning (when feasible)
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
   - When to use each approach

2. **Advanced LoRA Variants**
   - **DoRA**: Weight-decomposed low-rank adaptation (+3.7 points on commonsense reasoning)
   - **rsLoRA**: Rank-stabilized LoRA for higher ranks
   - **QA-LoRA**: Quantization-aware LoRA
   - Target module selection strategies

3. **Training Enhancements**
   - **NEFTune**: Noisy embeddings (29.8% → 64.7% on AlpacaEval!)
   - Gradient checkpointing
   - Flash Attention integration
   - Unsloth 2x speedup

4. **Dataset Preparation**
   - Instruction formats (Alpaca, ShareGPT, OpenAI)
   - Chat templates
   - Data quality filtering
   - Synthetic data generation basics

5. **Preference Optimization**
   - Reward modeling overview
   - **DPO** (Direct Preference Optimization)
   - **SimPO**: Simpler, no reference model needed (+6.4 points on AlpacaEval)
   - **ORPO**: Odds Ratio Preference Optimization (50% less memory)
   - **KTO**: Kahneman-Tversky Optimization (binary signals)
   - **IPO**: Identity Preference Optimization
   - Choosing the right method

6. **Training Infrastructure**
   - LLaMA Factory GUI
   - Axolotl configuration
   - TRL library usage

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 3.1.1 | **LoRA Theory Notebook** | Implement LoRA from scratch on a small transformer. Visualize weight updates. | 2 hours |
| 3.1.2 | **DoRA Comparison** | Fine-tune same model with LoRA vs DoRA. Compare quality on benchmark. | 2 hours |
| 3.1.3 | **NEFTune Magic** | Add NEFTune to training pipeline. Measure improvement on evaluation. | 1 hour |
| 3.1.4 | **8B Model LoRA Fine-tuning** | Fine-tune Llama 3.1 8B with LoRA + NEFTune on custom dataset. Use Unsloth. | 3 hours |
| 3.1.5 | **70B Model QLoRA** | Fine-tune 70B model using QLoRA. Document memory usage—this is the DGX Spark showcase! | 4 hours |
| 3.1.6 | **Dataset Preparation** | Create instruction dataset from raw data. Implement multiple formats. | 2 hours |
| 3.1.7 | **DPO Training** | Implement preference optimization with DPO. Compare with SFT baseline. | 2 hours |
| 3.1.8 | **SimPO vs ORPO** | Train same model with SimPO and ORPO. Compare quality and memory usage. | 2 hours |
| 3.1.9 | **KTO for Binary Feedback** | Train with KTO using thumbs up/down data. | 2 hours |
| 3.1.10 | **Integration with Ollama** | Convert fine-tuned model to GGUF, import to Ollama, test in your Web UI. | 2 hours |

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

> **NEFTune Implementation (5 lines!):**
> ```python
> def neftune_forward(self, input_ids):
>     embeddings = self.original_forward(input_ids)
>     if self.training:
>         noise = torch.randn_like(embeddings) * (self.neftune_alpha / embeddings.size(1)**0.5)
>         embeddings = embeddings + noise
>     return embeddings
> ```

> **Choosing Preference Method:**
> - **DPO**: Proven, well-understood, good default
> - **SimPO**: Better results, simpler, no reference model
> - **ORPO**: Memory constrained? Use ORPO (no ref model, single stage)
> - **KTO**: Only have binary feedback? Use KTO

### Milestone Checklist

- [ ] LoRA theory notebook with from-scratch implementation
- [ ] DoRA comparison showing improvement
- [ ] NEFTune improvement measured
- [ ] 8B model fine-tuned with LoRA + NEFTune
- [ ] 70B model fine-tuned with QLoRA (DGX Spark showcase!)
- [ ] Custom instruction dataset created
- [ ] DPO preference optimization completed
- [ ] SimPO and ORPO compared
- [ ] KTO trained with binary feedback
- [ ] Fine-tuned model running in Ollama

---

## Module 3.2: Quantization & Optimization

**Duration:** Weeks 19-20 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Apply Blackwell-exclusive NVFP4 and FP8 quantization
- Quantize models using GPTQ, AWQ, and GGUF
- Optimize models for inference using TensorRT-LLM
- Evaluate quantization impact on model quality

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.2.1 | Explain NVFP4 micro-block scaling and FP8 E4M3 format | Understand |
| 3.2.2 | Quantize models using NVFP4 with TensorRT Model Optimizer | Apply |
| 3.2.3 | Apply GPTQ, AWQ, and GGUF quantization | Apply |
| 3.2.4 | Measure and compare quality degradation from quantization | Evaluate |

### Topics

1. **Quantization Fundamentals**
   - Data types: FP32, FP16, BF16, INT8, INT4, FP8, FP4
   - Post-training quantization (PTQ) vs quantization-aware training (QAT)
   - Calibration datasets and techniques

2. **Blackwell-Specific Quantization**
   - **NVFP4**: NVIDIA's proprietary FP4 with dual-level scaling
     - Micro-block scaling for accuracy
     - 3.5× memory reduction vs FP16
     - Only ~0.1% accuracy loss on MMLU
   - **FP8 (E4M3/E5M2)**: Native Blackwell support
     - E4M3 for inference (higher precision)
     - E5M2 for training (larger range)
   - **MXFP4**: Open Compute Project format
   - TensorRT Model Optimizer workflow

3. **Standard Quantization Methods**
   - GPTQ (GPU-optimized PTQ)
   - AWQ (Activation-aware Quantization)
   - GGUF format for llama.cpp
   - bitsandbytes NF4 (for training)

4. **Quality Evaluation**
   - Perplexity measurement
   - Task-specific benchmarks (MMLU, HellaSwag)
   - Output comparison and human evaluation
   - Acceptable quality thresholds

5. **TensorRT-LLM Integration**
   - Building TensorRT engines
   - Weight-only vs full quantization
   - INT8 KV cache
   - Optimal configurations for DGX Spark

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 3.2.1 | **Data Type Exploration** | Create notebook showing FP32→FP16→FP8→FP4 representations. Visualize precision loss. | 1.5 hours |
| 3.2.2 | **NVFP4 Quantization** | Use TensorRT Model Optimizer for NVFP4 quantization of 70B model. Benchmark against FP16. This is the Blackwell showcase! | 3 hours |
| 3.2.3 | **FP8 Training and Inference** | Train a model with FP8. Compare training curves with FP16. Run FP8 inference. | 2 hours |
| 3.2.4 | **GPTQ Quantization** | Quantize 7B model with GPTQ. Compare different group sizes (32, 64, 128). | 2 hours |
| 3.2.5 | **AWQ Quantization** | Quantize same model with AWQ. Compare with GPTQ results. | 1.5 hours |
| 3.2.6 | **GGUF Conversion** | Convert model to GGUF format. Test with llama.cpp on DGX Spark. | 2 hours |
| 3.2.7 | **Quality Benchmark Suite** | Create standardized benchmark: perplexity, MMLU sample, generation quality. Run on all quantization variants. | 2 hours |
| 3.2.8 | **TensorRT-LLM Engine** | Build TensorRT-LLM engine with NVFP4. Measure prefill/decode performance. | 2 hours |

### Guidance

> **NVFP4 Quantization Workflow:**
> ```python
> from tensorrt_model_optimizer.torch.quantization import quantize
> 
> model = quantize(
>     model,
>     quant_cfg="nvfp4",
>     calibration_dataloader=calib_dl
> )
> # Then export to TensorRT-LLM
> ```

> **DGX Spark FP4 Performance:** Expect ~10,000+ tok/s prefill for 8B models in NVFP4—this is the Blackwell exclusive advantage!

### Milestone Checklist

- [ ] Data type precision visualization complete
- [ ] NVFP4 quantization of 70B model (DGX Spark showcase!)
- [ ] FP8 training and inference demonstrated
- [ ] GPTQ quantization with multiple configurations
- [ ] AWQ quantization completed
- [ ] GGUF conversion and llama.cpp testing done
- [ ] Quality benchmark suite created
- [ ] TensorRT-LLM engine built and benchmarked

---

## Module 3.3: Deployment & Inference

**Duration:** Weeks 21-22 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Deploy models using multiple inference engines (Ollama, vLLM, SGLang, TensorRT-LLM)
- Implement speculative decoding for faster inference
- Optimize inference for latency and throughput
- Select the right inference engine for different requirements

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.3.1 | Compare inference engines and select optimal for use case | Analyze |
| 3.3.2 | Implement speculative decoding with Medusa/EAGLE | Apply |
| 3.3.3 | Configure continuous batching and PagedAttention | Apply |
| 3.3.4 | Deploy production-ready REST APIs | Apply |

### Topics

1. **Inference Engine Overview**
   - Ollama (user-friendly, optimized for DGX Spark)
   - llama.cpp (fastest decode, GGUF format)
   - vLLM (PagedAttention, continuous batching)
   - TensorRT-LLM (best prefill, NVIDIA optimized)
   - **SGLang** (RadixAttention, 29-45% faster than vLLM)

2. **Speculative Decoding**
   - Theory: draft-verify paradigm
   - **Medusa**: Multi-head speculation (2-3x speedup, no draft model)
   - **EAGLE**: Efficient acceleration for LLMs
   - **EAGLE-3**: Latest improvements
   - Acceptance rate optimization
   - When speculative decoding helps/hurts

3. **Optimization Techniques**
   - Continuous batching
   - **PagedAttention** for efficient KV cache
   - **RadixAttention** (SGLang) for prefix caching
   - Tensor parallelism
   - KV cache quantization (INT8/FP8)

4. **Serving Infrastructure**
   - REST API design patterns
   - Streaming responses (SSE)
   - Load balancing strategies
   - Health checks and monitoring

5. **Production Considerations**
   - Latency vs throughput tradeoffs
   - Memory management and OOM prevention
   - Scaling strategies
   - Cost optimization

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 3.3.1 | **Engine Benchmark** | Benchmark same model on Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM. Create comprehensive comparison report. | 3 hours |
| 3.3.2 | **SGLang Deployment** | Deploy model with SGLang. Test RadixAttention with shared prefixes. Measure speedup. | 2 hours |
| 3.3.3 | **vLLM Continuous Batching** | Deploy with vLLM. Implement continuous batching. Measure throughput under load. | 2 hours |
| 3.3.4 | **Medusa Speculative Decoding** | Configure Medusa heads. Measure speedup vs standard decoding. Analyze acceptance rates. | 2 hours |
| 3.3.5 | **EAGLE-3 Implementation** | Set up EAGLE-3 speculative decoding. Compare with Medusa. | 2 hours |
| 3.3.6 | **TensorRT-LLM Optimization** | Build optimized TensorRT-LLM engine. Measure prefill performance. | 2 hours |
| 3.3.7 | **Production API** | Create FastAPI wrapper with streaming, error handling, metrics. | 2 hours |

### Guidance

> **Engine Selection Guide (DGX Spark):**
> | Engine | Best For | Prefill (tok/s) | Decode (tok/s) |
> |--------|----------|-----------------|----------------|
> | llama.cpp | Interactive chat | ~1,500 | ~59 |
> | Ollama | Easy setup | ~3,000 | ~45 |
> | vLLM | High throughput | ~5,000 | ~40 |
> | SGLang | Shared prefixes | ~6,000 | ~42 |
> | TensorRT-LLM | Batch inference | ~10,000+ | ~38 |

### Milestone Checklist

- [ ] Comprehensive engine benchmark report created
- [ ] SGLang deployment with RadixAttention tested
- [ ] vLLM deployment with continuous batching working
- [ ] Medusa speculative decoding speedup measured
- [ ] EAGLE-3 comparison completed
- [ ] TensorRT-LLM engine optimized
- [ ] Production FastAPI server implemented

---

## Module 3.4: Test-Time Compute & Reasoning

**Duration:** Week 23 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Explain test-time compute scaling and inference-time reasoning
- Implement Chain-of-Thought and reasoning strategies
- Use reasoning models (QwQ, Magistral, DeepSeek-R1) effectively
- Apply basic reward model concepts for output selection

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.4.1 | Explain test-time compute vs training-time compute tradeoffs | Understand |
| 3.4.2 | Implement Chain-of-Thought and reasoning prompting | Apply |
| 3.4.3 | Run and evaluate reasoning models (DeepSeek-R1) | Apply |
| 3.4.4 | Implement Best-of-N sampling with reward models | Apply |

### Topics

1. **Test-Time Compute Scaling**
   - Training compute vs inference compute
   - O1/O3 reasoning paradigm
   - When to invest in test-time compute
   - Cost-quality tradeoffs

2. **Reasoning Strategies**
   - Chain-of-Thought (CoT) prompting
   - Self-consistency (majority voting)
   - Tree-of-Thought exploration
   - Let's verify step by step

3. **Reasoning Models**
   - DeepSeek-R1 architecture and training (GRPO)
   - R1 distilled models (1.5B, 7B, 14B, 32B, 70B)
   - QwQ and other reasoning models
   - Running reasoning models on DGX Spark

4. **Reward Models and Selection**
   - Reward model basics
   - Best-of-N sampling
   - Process reward models (PRM) vs outcome reward models (ORM)
   - When reward guidance helps

5. **Practical Applications**
   - Math problem solving
   - Code generation with reasoning
   - Complex analysis tasks
   - Multi-step planning

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 3.4.1 | **CoT Prompting Workshop** | Implement CoT prompting for math and reasoning tasks. Measure accuracy improvement. | 2 hours |
| 3.4.2 | **Self-Consistency** | Implement majority voting with multiple samples. Measure accuracy vs single-shot. | 1.5 hours |
| 3.4.3 | **DeepSeek-R1 Exploration** | Run R1-distill-70B on DGX Spark. Test on math, coding, analysis tasks. Document thinking process. | 2 hours |
| 3.4.4 | **R1 vs Standard Model** | Compare R1 with standard Llama 70B on reasoning benchmarks. | 1.5 hours |
| 3.4.5 | **Best-of-N with Reward Model** | Implement Best-of-N sampling using a reward model. Measure quality improvement. | 2 hours |
| 3.4.6 | **Reasoning Pipeline** | Build pipeline that uses reasoning model for complex tasks, standard model for simple ones. | 2 hours |

### Guidance

> **Running R1 on DGX Spark:** DeepSeek-R1-distill-70B runs well at Q4 quantization. The explicit thinking tokens are valuable for understanding model reasoning.

### Milestone Checklist

- [ ] CoT prompting with measured improvement
- [ ] Self-consistency implementation working
- [ ] DeepSeek-R1 running and tested
- [ ] R1 vs standard model comparison complete
- [ ] Best-of-N sampling implemented
- [ ] Reasoning pipeline created

---

## Module 3.5: RAG Systems & Vector Databases

**Duration:** Weeks 24-25 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build production-ready RAG pipelines
- Select and use appropriate vector databases
- Optimize retrieval quality with advanced techniques
- Evaluate RAG system performance

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.5.1 | Design and implement complete RAG architectures | Create |
| 3.5.2 | Use vector databases (ChromaDB, FAISS, Qdrant) effectively | Apply |
| 3.5.3 | Implement advanced retrieval techniques (hybrid search, reranking) | Apply |
| 3.5.4 | Evaluate RAG systems with appropriate metrics | Evaluate |

### Topics

1. **RAG Fundamentals**
   - Retrieval-Augmented Generation concept
   - When to use RAG vs fine-tuning
   - RAG architecture patterns
   - Context window management

2. **Document Processing**
   - Document loading (PDF, DOCX, HTML, Markdown)
   - Chunking strategies
     - Fixed size with overlap
     - Semantic chunking
     - Sentence-based chunking
   - Metadata extraction and enrichment

3. **Embedding Models**
   - Sentence transformers
   - Local embedding models (BGE, Nomic, E5)
   - Running embeddings on DGX Spark
   - Embedding model selection criteria

4. **Vector Databases**
   - **ChromaDB**: Simple, local, Python-native
   - **FAISS**: Facebook's similarity search (GPU-accelerated)
   - **Qdrant**: Production-ready, filtering
   - **Milvus**: Distributed, scalable
   - Index types and tradeoffs (HNSW, IVF)

5. **Advanced Retrieval**
   - Hybrid search (dense + sparse)
   - BM25 for keyword matching
   - Query expansion
   - **Reranking** with cross-encoders
   - Hypothetical Document Embedding (HyDE)

6. **RAG Evaluation**
   - Retrieval metrics (Recall@K, MRR)
   - Answer quality metrics (RAGAS)
   - Faithfulness and groundedness
   - End-to-end evaluation

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 3.5.1 | **Basic RAG Pipeline** | Build RAG system: document ingestion → chunking → embedding → ChromaDB → retrieval → generation. | 3 hours |
| 3.5.2 | **Chunking Strategies** | Compare fixed, semantic, and sentence chunking. Measure retrieval quality for each. | 2 hours |
| 3.5.3 | **Vector Database Comparison** | Implement same RAG with ChromaDB, FAISS, Qdrant. Compare speed and features. | 2 hours |
| 3.5.4 | **Hybrid Search** | Implement BM25 + dense retrieval fusion. Measure improvement over dense-only. | 2 hours |
| 3.5.5 | **Reranking Pipeline** | Add cross-encoder reranking. Measure quality improvement. | 2 hours |
| 3.5.6 | **RAG Evaluation Framework** | Implement RAGAS evaluation. Create benchmark for your RAG system. | 2 hours |
| 3.5.7 | **Production RAG** | Build production-ready RAG with error handling, caching, monitoring. | 2 hours |

### Guidance

> **Local RAG Stack on DGX Spark:**
> - Embeddings: `bge-large-en-v1.5` via sentence-transformers (GPU accelerated)
> - Vector DB: ChromaDB or FAISS with GPU
> - LLM: 70B via Ollama
> - Reranker: `bge-reranker-large`

### Milestone Checklist

- [ ] Basic RAG pipeline working
- [ ] Chunking strategies compared
- [ ] Multiple vector databases tested
- [ ] Hybrid search implemented
- [ ] Reranking improving quality
- [ ] RAGAS evaluation framework created
- [ ] Production-ready RAG system built

---

## Module 3.6: AI Agents & Agentic Systems

**Duration:** Week 26 (10-12 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build AI agents using LangChain and LangGraph
- Implement multi-agent systems for complex tasks
- Design and implement tool-using agents
- Evaluate agent performance and reliability

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.6.1 | Create agents with custom tools | Apply |
| 3.6.2 | Design multi-agent architectures | Create |
| 3.6.3 | Implement stateful agent workflows with LangGraph | Apply |
| 3.6.4 | Evaluate agent performance and reliability | Evaluate |

### Topics

1. **Agent Fundamentals**
   - ReAct pattern (Reasoning + Acting)
   - Tool use and function calling
   - Memory and context management
   - Planning and decomposition

2. **LangChain Framework**
   - Chains and composition
   - Agents and tools
   - Memory systems
   - Callbacks and tracing

3. **LangGraph & Workflows**
   - Stateful agents
   - Graph-based orchestration
   - Human-in-the-loop
   - Error recovery and fallbacks

4. **Multi-Agent Systems**
   - Agent communication patterns
   - CrewAI framework
   - AutoGen for coding agents
   - Task decomposition strategies

5. **Agent Evaluation**
   - Task completion metrics
   - Tool use efficiency
   - Cost and latency tracking
   - Reliability and error rates

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 3.6.1 | **Custom Tools** | Create 5 custom tools (web search, calculator, code executor, file reader, API caller). Build agent that uses them. | 2 hours |
| 3.6.2 | **LangGraph Workflow** | Implement multi-step workflow with branching logic, error handling, human approval. | 2 hours |
| 3.6.3 | **Multi-Agent System** | Create 3-agent system (researcher, writer, reviewer) for content generation. | 2 hours |
| 3.6.4 | **Coding Agent** | Build agent that can write, run, and debug code. Test on programming tasks. | 2 hours |
| 3.6.5 | **Agent with RAG** | Combine RAG system with agent for document-grounded responses. | 2 hours |
| 3.6.6 | **Agent Benchmark** | Create evaluation framework. Test systematically. | 2 hours |

### Milestone Checklist

- [ ] 5 custom tools implemented and tested
- [ ] LangGraph workflow with branching
- [ ] Multi-agent content generation system
- [ ] Coding agent working
- [ ] Agent + RAG integration complete
- [ ] Agent evaluation framework created

---

# DOMAIN 4: PRODUCTION AI

## Module 4.1: Multimodal AI

**Duration:** Week 27 (8-10 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Work with vision-language models for image understanding
- Build multimodal pipelines combining vision and language
- Process documents with OCR and layout understanding
- Fine-tune multimodal models

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.1.1 | Use vision-language models for image analysis | Apply |
| 4.1.2 | Build multimodal RAG systems | Apply |
| 4.1.3 | Process documents with OCR and layout analysis | Apply |
| 4.1.4 | Implement audio transcription pipelines | Apply |

### Topics

1. **Vision-Language Models**
   - LLaVA architecture
   - CLIP and multimodal embeddings
   - Qwen2-VL, InternVL
   - Document understanding

2. **Multimodal Pipelines**
   - Document AI (OCR, layout)
   - Multimodal RAG
   - Video understanding
   - Audio integration (Whisper)

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 4.1.1 | **VLM Demo** | Deploy LLaVA or Qwen2-VL. Create demo for image understanding, VQA, OCR. | 2 hours |
| 4.1.2 | **Multimodal RAG** | Build RAG that indexes images and text. Query with natural language. | 2 hours |
| 4.1.3 | **Document AI Pipeline** | Create PDF processing pipeline: OCR, layout analysis, QA over documents. | 2 hours |
| 4.1.4 | **Audio Transcription** | Deploy Whisper, create transcription pipeline. Combine with LLM for audio Q&A. | 2 hours |

### Milestone Checklist

- [ ] Vision-language model running with demo
- [ ] Multimodal RAG indexing images and text
- [ ] Document AI pipeline working
- [ ] Audio transcription pipeline complete

---

## Module 4.2: AI Safety & Alignment

**Duration:** Weeks 28-29 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Implement guardrails and safety filters for LLM applications
- Perform red teaming and vulnerability assessment
- Evaluate models on safety benchmarks
- Apply responsible AI practices

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.2.1 | Implement NeMo Guardrails for LLM safety | Apply |
| 4.2.2 | Perform automated red teaming with DeepTeam/Promptfoo | Apply |
| 4.2.3 | Evaluate models on safety benchmarks (TruthfulQA, BBQ) | Apply |
| 4.2.4 | Create model cards documenting safety considerations | Create |

### Topics

1. **LLM Safety Challenges**
   - Prompt injection attacks
   - Jailbreaking techniques
   - Hallucination and factuality
   - Bias and fairness

2. **Guardrails Implementation**
   - NeMo Guardrails framework
   - Llama Guard classification (8B)
   - Input/output filtering
   - Topic guardrails

3. **Red Teaming**
   - Manual red teaming methodology
   - Automated red teaming (DeepTeam, Promptfoo, PyRIT)
   - Attack categories (OWASP LLM Top 10)
   - Vulnerability assessment

4. **Safety Benchmarks**
   - TruthfulQA for factuality
   - BBQ for bias
   - HELM Safety Suite
   - Custom safety evaluations

5. **Responsible AI Practices**
   - Model cards
   - EU AI Act compliance basics
   - NIST AI RMF overview
   - Documentation requirements

6. **Alignment Techniques Review**
   - Constitutional AI concepts
   - RLHF vs DPO for safety
   - Refusal training
   - Harmlessness vs helpfulness tradeoff

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 4.2.1 | **NeMo Guardrails Setup** | Implement guardrails for a chatbot. Configure topic restrictions, input validation, output filtering. | 3 hours |
| 4.2.2 | **Llama Guard Integration** | Deploy Llama Guard 3 8B on DGX Spark. Create classification pipeline for user inputs. | 2 hours |
| 4.2.3 | **Automated Red Teaming** | Use DeepTeam/Promptfoo to attack your models. Document vulnerabilities. | 3 hours |
| 4.2.4 | **Safety Benchmark Suite** | Run TruthfulQA, BBQ on your fine-tuned models. Compare safety metrics. | 2 hours |
| 4.2.5 | **Bias Evaluation** | Use Fairlearn to evaluate model outputs for demographic biases. | 2 hours |
| 4.2.6 | **Model Card Creation** | Create comprehensive model card for a fine-tuned model including safety evaluation. | 2 hours |

### Guidance

> **NeMo Guardrails Config:**
> ```yaml
> rails:
>   input:
>     flows:
>       - check_jailbreak
>       - check_topic_allowed
>   output:
>     flows:
>       - check_hallucination
>       - check_harmful_content
> ```

### Milestone Checklist

- [ ] NeMo Guardrails implemented for chatbot
- [ ] Llama Guard classification working
- [ ] Automated red teaming completed
- [ ] Safety benchmarks run and documented
- [ ] Bias evaluation performed
- [ ] Model card created with safety section

---

## Module 4.3: MLOps & Experiment Tracking 

**Duration:** Weeks 30-31 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Track experiments with MLflow and Weights & Biases
- Evaluate LLMs using standard benchmarks
- Detect model drift and performance degradation
- Create reproducible ML pipelines

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.3.1 | Set up and use MLflow/W&B for experiment tracking | Apply |
| 4.3.2 | Run standard LLM benchmarks (MMLU, HellaSwag) | Apply |
| 4.3.3 | Implement drift detection with Evidently AI | Apply |
| 4.3.4 | Version datasets and models systematically | Apply |

### Topics

1. **Experiment Tracking**
   - MLflow setup and concepts
   - Weights & Biases integration
   - Logging parameters, metrics, artifacts
   - Experiment comparison

2. **LLM Benchmarks**
   - MMLU, HellaSwag, ARC, WinoGrande
   - HumanEval for code
   - MT-Bench for chat
   - LM Evaluation Harness

3. **Custom Evaluation**
   - Task-specific metrics
   - LLM-as-judge evaluation
   - Human evaluation protocols

4. **Model Monitoring**
   - Concept drift detection
   - Data drift detection
   - Performance monitoring
   - Evidently AI integration

5. **Versioning and Reproducibility**
   - Model versioning (HF Hub, DVC)
   - Dataset versioning
   - Environment reproducibility
   - Random seed management

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 4.3.1 | **MLflow Setup** | Set up MLflow on DGX Spark. Log fine-tuning runs with parameters, metrics, checkpoints. | 2 hours |
| 4.3.2 | **W&B Integration** | Integrate Weights & Biases. Create training dashboards. | 2 hours |
| 4.3.3 | **Benchmark Suite** | Run lm-evaluation-harness on multiple models. Create comparison tables. | 2 hours |
| 4.3.4 | **Custom Evaluation** | Design evaluation for your use case with LLM-as-judge. | 2 hours |
| 4.3.5 | **Drift Detection** | Set up Evidently AI monitoring. Simulate drift, verify detection. | 2 hours |
| 4.3.6 | **Model Registry** | Create model versioning workflow. Track fine-tuned versions. | 2 hours |
| 4.3.7 | **Reproducibility Audit** | Document complete environment. Verify you can reproduce a training run. | 2 hours |

### Milestone Checklist

- [ ] MLflow tracking server running
- [ ] W&B integration complete
- [ ] Benchmark results collected
- [ ] Custom evaluation framework implemented
- [ ] Drift detection working
- [ ] Model registry workflow established
- [ ] Reproducibility verified

---

## Module 4.4: Containerization & Deployment 

**Duration:** Weeks 32-33 (12-15 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Containerize ML applications with Docker
- Deploy models to cloud platforms (AWS, GCP)
- Use Kubernetes basics for ML deployment
- Build demo applications with Gradio/Streamlit

### Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.4.1 | Create Docker images for ML applications | Apply |
| 4.4.2 | Deploy models to AWS SageMaker or GCP Vertex AI | Apply |
| 4.4.3 | Use basic Kubernetes for ML deployments | Apply |
| 4.4.4 | Build interactive demos with Gradio | Apply |

### Topics

1. **Docker for ML**
   - Dockerfile best practices for ML
   - Multi-stage builds
   - GPU container configuration
   - Docker Compose for ML stacks
   - NGC container customization

2. **Cloud Deployment**
   - AWS SageMaker endpoints
   - GCP Vertex AI deployment
   - Azure ML (overview)
   - Cost optimization strategies

3. **Kubernetes Basics**
   - Kubernetes concepts for ML
   - Deployments and Services
   - GPU scheduling
   - Horizontal scaling

4. **Demo Building**
   - **Gradio** for ML interfaces
   - **Streamlit** for data apps
   - Sharing and hosting
   - Integration with inference servers

5. **CI/CD for ML**
   - GitHub Actions for ML
   - Model validation in CI
   - Automated deployment
   - A/B testing frameworks

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 4.4.1 | **Docker ML Image** | Create optimized Dockerfile for inference server. Multi-stage build, GPU support. | 2 hours |
| 4.4.2 | **Docker Compose Stack** | Create compose file for: inference server, monitoring, vector DB. Test locally. | 2 hours |
| 4.4.3 | **AWS SageMaker Deployment** | Deploy model to SageMaker endpoint. Benchmark latency and cost. | 2 hours |
| 4.4.4 | **GCP Vertex AI** | Deploy same model to Vertex AI. Compare with SageMaker. | 2 hours |
| 4.4.5 | **Kubernetes Deployment** | Deploy inference server to local K8s (minikube/kind). Configure GPU. | 2 hours |
| 4.4.6 | **Gradio Demo** | Build interactive demo for your fine-tuned model. Deploy to Hugging Face Spaces. | 2 hours |
| 4.4.7 | **Streamlit Dashboard** | Create dashboard showing model metrics, comparisons, playground. | 2 hours |

### Guidance

> **Docker for NGC:**
> ```dockerfile
> FROM nvcr.io/nvidia/pytorch:25.11-py3
> 
> # Your customizations
> COPY requirements.txt .
> RUN pip install -r requirements.txt
> 
> COPY app/ /app/
> CMD ["python", "/app/serve.py"]
> ```

### Milestone Checklist

- [ ] Optimized Docker image created
- [ ] Docker Compose stack running
- [ ] AWS SageMaker deployment working
- [ ] GCP Vertex AI deployment working
- [ ] Kubernetes deployment complete
- [ ] Gradio demo on Hugging Face Spaces
- [ ] Streamlit dashboard created

---

## Module 4.5: Demo Building & Prototyping

**Duration:** Week 34 (6-8 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Build polished demo applications rapidly
- Create shareable prototypes for stakeholders
- Integrate multiple ML components into cohesive applications

### Topics

1. **Gradio Advanced**
   - Custom components
   - Blocks API for complex layouts
   - State management
   - Authentication

2. **Streamlit Advanced**
   - Multi-page apps
   - Session state
   - Caching strategies
   - Custom components

3. **Prototype Patterns**
   - Demo ≠ Production
   - Rapid iteration
   - User feedback integration

### Labs

| Lab | Title | Description & Deliverable | Est. Time |
|-----|-------|---------------------------|-----------|
| 4.5.1 | **Complete RAG Demo** | Build Gradio app: file upload → RAG indexing → Q&A → sources display. | 3 hours |
| 4.5.2 | **Agent Playground** | Create Streamlit app to interact with your agent, visualize tool calls, show reasoning. | 3 hours |
| 4.5.3 | **Portfolio Demo** | Create polished demo showcasing your capstone project. | 2 hours |

### Milestone Checklist

- [ ] RAG demo with full features
- [ ] Agent playground interactive
- [ ] Portfolio demo polished and deployed

---

## Module 4.6: Capstone Project

**Duration:** Weeks 35-40 (40-50 hours)

### Learning Outcomes
By the end of this module, you will be able to:
- Design and implement an end-to-end AI system
- Apply all techniques learned throughout the curriculum
- Document and present your work professionally
- Deploy a production-ready AI application

### Project Options

Choose ONE of the following capstone projects:

#### Option A: Domain-Specific AI Assistant
Build a complete AI assistant for a specific domain with safety guardrails.

**Requirements:**
- Fine-tuned base model (70B with QLoRA)
- RAG system with domain knowledge base
- Custom tools for domain operations
- NeMo Guardrails for safety
- Production API with streaming
- Comprehensive evaluation (benchmark + safety)
- Gradio demo

#### Option B: Multimodal Document Intelligence
Build a system that processes and understands complex documents.

**Requirements:**
- Document ingestion pipeline (PDF, images)
- Vision-language model integration
- Structured information extraction
- Multimodal RAG
- Export to structured formats
- Evaluation on document benchmarks
- Interactive demo

#### Option C: AI Agent Swarm with Safety
Build a multi-agent system with comprehensive safety measures.

**Requirements:**
- Minimum 4 specialized agents
- Agent communication protocol
- Task decomposition and planning
- Human-in-the-loop approval
- Guardrails on all agent outputs
- Error recovery and fallbacks
- Performance benchmarking

#### Option D: End-to-End MLOps Pipeline
Build a complete fine-tuning and deployment pipeline.

**Requirements:**
- Data collection and curation pipeline
- Automated fine-tuning (SFT + DPO)
- Experiment tracking with MLflow
- Safety evaluation in CI
- Model versioning and comparison
- Automated deployment to cloud
- Drift monitoring

### Project Phases

| Phase | Duration | Activities |
|-------|----------|------------|
| **Planning** | Week 35 | Requirements, architecture design, tech selection |
| **Foundation** | Week 36-37 | Core components, data preparation, initial models |
| **Integration** | Week 38 | Component integration, API development |
| **Safety & Eval** | Week 39 | Safety testing, benchmarking, optimization |
| **Documentation** | Week 40 | Documentation, demo, presentation |

### Deliverables

1. **Technical Report** (15-20 pages)
2. **Code Repository** (well-documented)
3. **Demo** (Gradio/Streamlit + video)
4. **Presentation** (15-20 slides)
5. **Model Card** (with safety evaluation)

### Milestone Checklist

- [ ] Project proposal approved
- [ ] Architecture design documented
- [ ] Core components implemented
- [ ] Safety measures integrated
- [ ] Evaluation results collected
- [ ] Documentation written
- [ ] Demo video created
- [ ] Model card completed
- [ ] Presentation ready

---

# OPTIONAL MODULES

These modules can be completed independently based on interest and time availability.

## Optional A: Learning Theory Deep Dive

**Duration:** 4-6 hours

### Topics
- VC dimension and model capacity
- Bias-variance tradeoff analysis
- PAC learning framework
- Generalization bounds
- Double descent phenomenon

### Lab
- Implement experiments demonstrating bias-variance tradeoff
- Visualize double descent on MNIST

---

## Optional B: Recommender Systems

**Duration:** 6-8 hours

### Topics
- Collaborative filtering
- Matrix factorization
- Neural collaborative filtering
- Two-tower architectures
- Evaluation metrics (NDCG, MAP)

### Lab
- Build movie recommender with matrix factorization
- Implement neural collaborative filtering

---

## Optional C: Mechanistic Interpretability

**Duration:** 6-8 hours

### Topics
- Activation patching
- Attention head analysis
- Circuit discovery
- Probing classifiers
- TransformerLens library

### Lab
- Analyze attention patterns in small transformer
- Identify circuits for specific behaviors

---

## Optional D: Reinforcement Learning Fundamentals

**Duration:** 8-10 hours

### Topics
- Markov Decision Processes
- Q-learning and DQN
- Policy gradient methods
- Connection to RLHF
- PPO algorithm

### Labs
- Implement Q-learning for GridWorld
- Train DQN on CartPole
- Understand PPO for language model fine-tuning

---

## Optional E: Graph Neural Networks

**Duration:** 6-8 hours

### Topics
- Graph data representations
- Message passing framework
- GCN, GraphSAGE, GAT
- Applications (molecules, social networks)
- PyTorch Geometric

### Lab
- Node classification with GCN
- Molecular property prediction

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

### Performance Benchmarks (2025 Models)
| Model | Precision | Prefill (tok/s) | Decode (tok/s) | Capabilities | Notes |
|-------|-----------|-----------------|----------------|--------------|-------|
| Nemotron-3-Nano | Q4_K_M | 15,000 | 55 | Think ✅ Tools ✅ | 1M context, 3.2B active |
| Qwen3-8B | NVFP4 | 12,000 | 42 | Think ✅ Tools ✅ | Hybrid /think mode |
| Qwen3-32B | Q4_K_M | 3,800 | 35 | Think ✅ Tools ✅ | Best BFCL (68.2) |
| QwQ-32B | Q4_K_M | 3,200 | 28 | Think ✅ Tools ⚠️ | 79.5% AIME |
| Magistral-Small | Q4_K_M | 3,500 | 32 | Think ✅ Tools ✅ Vision ✅ | 86% AIME, multimodal |
| Devstral-Small-2 | Q4_K_M | 3,800 | 34 | Think ✅ Tools ✅ Vision ✅ | 68% SWE-Bench, code |
| DeepSeek-R1-8B | Q4_K_M | 8,000 | 45 | Think ✅ Tools ❌ | Reasoning only |
| Qwen3-VL-8B | Q4_K_M | 4,500 | 38 | Think ✅ Tools ✅ Vision ✅ | 32-lang OCR |

> **⚠️ Tool Calling Warning:** DeepSeek-R1 does NOT support tool calling - use QwQ or Magistral for agents!

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

# Ollama (2025 Tier 1 Models)
ollama list
ollama run nemotron-3-nano     # General purpose (1M ctx, fastest)
ollama run qwen3:32b           # General purpose (best tools)
ollama run qwq:32b             # Reasoning (79.5% AIME)
ollama run magistral-small     # Reasoning + vision + tools
ollama run devstral-small-2    # Agentic coding (68% SWE-Bench)
ollama run qwen3-vl:8b         # Vision-language (GUI agents)
```

## Appendix B: NVIDIA Tools Compatibility

| Tool | ARM64 Support | Notes |
|------|--------------|-------|
| NeMo Framework | ✅ | Blackwell support confirmed |
| TensorRT-LLM | ⚠️ | Requires source build/NGC |
| Triton Server | ✅ | Official aarch64 wheels |
| RAPIDS (cuDF/cuML) | ✅ | Official since v22.04 |
| vLLM | ⚠️ | Use --enforce-eager flag |
| SGLang | ✅ | Blackwell support listed |
| llama.cpp | ✅ | Full support |

## Appendix C: Recommended Resources

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
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Safety
- [DeepTeam](https://github.com/confident-ai/deepteam) - Red teaming

## Appendix D: Progress Tracking

### Module Completion Tracker

| Module | Status | Start Date | End Date | Notes |
|--------|--------|------------|----------|-------|
| 1.1 DGX Spark Platform | ⬜ | | | |
| 1.2 Python for AI/ML | ⬜ | | | |
| 1.3 CUDA Python | ⬜ | | | |
| 1.4 Mathematics | ⬜ | | | |
| 1.5 Neural Networks | ⬜ | | | |
| 1.6 Classical ML | ⬜ | | | |
| 1.7 Capstone—MicroGrad+ | ⬜ | | | |
| 2.1 PyTorch | ⬜ | | | |
| 2.2 Computer Vision | ⬜ | | | |
| 2.3 NLP & Transformers | ⬜ | | | |
| 2.4 Efficient Architectures | ⬜ | | | |
| 2.5 Hugging Face | ⬜ | | | |
| 2.6 Diffusion Models | ⬜ | | | |
| 3.1 LLM Fine-Tuning | ⬜ | | | |
| 3.2 Quantization | ⬜ | | | |
| 3.3 Deployment & Inference | ⬜ | | | |
| 3.4 Test-Time Compute | ⬜ | | | |
| 3.5 RAG Systems | ⬜ | | | |
| 3.6 AI Agents | ⬜ | | | |
| 4.1 Multimodal AI | ⬜ | | | |
| 4.2 AI Safety | ⬜ | | | |
| 4.3 MLOps | ⬜ | | | |
| 4.4 Containerization | ⬜ | | | |
| 4.5 Demo Building | ⬜ | | | |
| 4.6 Capstone Project | ⬜ | | | |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01 | Initial curriculum release |
| 2.0 | 2025-12 | Major expansion based on curriculum research: Added CUDA Python, NVFP4/FP8 quantization, RAG systems, AI Safety, Docker/Cloud deployment, Mamba/MoE architectures, Diffusion models, Modern fine-tuning methods (DoRA, NEFTune, SimPO, ORPO), SGLang/speculative decoding, Test-time compute, Classical ML, Object detection/ViT, Kubernetes, Gradio/Streamlit, Optional modules for Learning Theory, Recommender Systems, Mechanistic Interpretability, RL, GNNs |

---

*This curriculum is optimized for NVIDIA DGX Spark with 128GB unified memory. Adjust batch sizes and model sizes if using different hardware.*