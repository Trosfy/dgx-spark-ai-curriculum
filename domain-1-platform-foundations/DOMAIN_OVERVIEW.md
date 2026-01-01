# Domain 1: Platform Foundations - Overview

## 🎯 Domain Purpose

Domain 1 establishes the essential foundation for AI/ML development on DGX Spark. You'll master the platform, programming fundamentals, mathematical concepts, and neural network building blocks needed for advanced deep learning work in later domains.

---

## 📊 Domain at a Glance

| Aspect | Details |
|--------|---------|
| **Modules** | 7 (1.1 through 1.7) |
| **Total Duration** | ~6-7 weeks |
| **Prerequisites** | Basic Python, high school math |
| **Capstone** | MicroGrad+ autograd library |

---

## 🗺️ Module Progression

```
Week 1          Week 2          Week 3          Week 4          Week 5          Week 6          Week 7
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  1.1    │    │  1.2    │    │  1.3    │    │  1.4    │    │  1.5    │    │  1.6    │    │  1.7    │
│ Platform│───►│ Python  │───►│  CUDA   │───►│  Math   │───►│ Neural  │───►│Classical│───►│Capstone │
│ Setup   │    │ NumPy   │    │ Python  │    │  Fndns  │    │Networks │    │   ML    │    │MicroGrad│
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │              │              │
     ▼              ▼              ▼              ▼              ▼              ▼              ▼
 Container      Vectorized     GPU kernels    Gradients     Forward/      XGBoost      Build your
 & GPU ready    operations     & memory       & chain       Backward      baselines    own autograd
                                              rule          passes
```

---

## 📚 Module Details

### Module 1.1: DGX Spark Platform
**Duration:** ~4-6 hours | **Priority:** P0 Critical

Understand your hardware and set up the development environment.

- GB10 Superchip specs (128GB unified memory, 6,144 CUDA cores)
- NGC container ecosystem
- Docker commands for GPU workloads
- Resource monitoring and troubleshooting

**Key Outcome:** Can launch NGC containers and verify GPU access

---

### Module 1.2: Python for AI/ML
**Duration:** ~6-8 hours | **Priority:** P0 Critical

Master NumPy vectorization and efficient data manipulation.

- NumPy array operations and broadcasting
- Einsum notation for tensor operations
- Pandas for data preprocessing
- Vectorization for 100x speedups

**Key Outcome:** Write vectorized code without Python loops

---

### Module 1.3: CUDA Python Introduction
**Duration:** ~8-10 hours | **Priority:** P1 High

Write GPU-accelerated code with Numba and CuPy.

- GPU architecture (SMs, warps, memory hierarchy)
- Numba CUDA kernels
- CuPy as drop-in NumPy replacement
- Memory coalescing and optimization

**Key Outcome:** Write custom GPU kernels that outperform CPU

---

### Module 1.4: Math Foundations for Deep Learning
**Duration:** ~6-8 hours | **Priority:** P0 Critical

Build the mathematical intuition for neural networks.

- Derivatives and gradients
- Chain rule and backpropagation
- Optimization (gradient descent, Adam)
- Linear algebra for deep learning

**Key Outcome:** Compute gradients by hand and verify with code

---

### Module 1.5: Neural Network Fundamentals
**Duration:** ~6-8 hours | **Priority:** P0 Critical

Implement neural networks from scratch.

- Layers, activations, and forward pass
- Backpropagation implementation
- Loss functions (MSE, Cross-Entropy)
- Training loops and batching

**Key Outcome:** Train a neural network from scratch on MNIST

---

### Module 1.6: Classical ML Foundations
**Duration:** ~6-8 hours | **Priority:** P2 Medium

Know when classical ML beats deep learning.

- XGBoost and gradient boosting
- RAPIDS cuML for GPU acceleration
- Hyperparameter tuning with Optuna
- When to use classical vs deep learning

**Key Outcome:** Create XGBoost baselines for comparison

---

### Module 1.7: Capstone — MicroGrad+
**Duration:** ~8-10 hours | **Priority:** P0 Critical

Build your own autograd library from scratch.

- Tensor class with automatic differentiation
- Layer, loss, and optimizer implementations
- Comprehensive testing with gradient checks
- MNIST example achieving >95% accuracy

**Key Outcome:** Working autograd library demonstrating Domain 1 mastery

---

## 🔗 How Modules Connect

```
Platform (1.1) ────► Required for everything else

Python (1.2) ──────► NumPy arrays used in all modules
                     ├── CUDA Python (1.3): Array operations on GPU
                     ├── Math (1.4): Gradient computations
                     ├── Neural Networks (1.5): Layer implementations
                     └── Classical ML (1.6): cuML DataFrames

CUDA Python (1.3) ─► Understanding GPU for:
                     ├── Neural Networks (1.5): Why GPU training matters
                     └── Capstone (1.7): Appreciate PyTorch's optimizations

Math (1.4) ────────► Foundation for:
                     ├── Neural Networks (1.5): Backpropagation
                     └── Capstone (1.7): Autograd implementation

Neural Networks (1.5) ► Implement components for Capstone (1.7)

Classical ML (1.6) ─► Baselines to compare against:
                     └── Capstone (1.7): MicroGrad+ vs XGBoost
```

---

## 📈 Skills Progression

| Skill | Module 1.1 | Module 1.2 | Module 1.3 | Module 1.4 | Module 1.5 | Module 1.6 | Module 1.7 |
|-------|------------|------------|------------|------------|------------|------------|------------|
| Docker/Containers | ●●●○ | ●●●○ | ●●●● | ●●●○ | ●●●○ | ●●●○ | ●●●○ |
| NumPy/Vectorization | ○○○○ | ●●●● | ●●●● | ●●●● | ●●●● | ●●●● | ●●●● |
| GPU Programming | ●○○○ | ○○○○ | ●●●● | ○○○○ | ●○○○ | ●●○○ | ●○○○ |
| Math/Calculus | ○○○○ | ●○○○ | ○○○○ | ●●●● | ●●●● | ●●○○ | ●●●● |
| Neural Networks | ○○○○ | ○○○○ | ○○○○ | ●●○○ | ●●●● | ●●●○ | ●●●● |
| Software Engineering | ●●○○ | ●●○○ | ●●●○ | ●●○○ | ●●●○ | ●●●○ | ●●●● |

Legend: ○ = Not covered, ● = Basic, ●● = Intermediate, ●●● = Advanced, ●●●● = Expert

---

## ✅ Domain Completion Checklist

### Module 1.1
- [ ] NGC container pulled and running
- [ ] GPU verified with nvidia-smi and torch.cuda
- [ ] Understand DGX Spark architecture

### Module 1.2
- [ ] NumPy broadcasting mastered
- [ ] Einsum notation comfortable
- [ ] Replaced loops with vectorization

### Module 1.3
- [ ] Custom CUDA kernel written
- [ ] Memory coalescing understood
- [ ] CuPy speedup demonstrated

### Module 1.4
- [ ] Gradients computed by hand
- [ ] Chain rule applied to multi-layer networks
- [ ] Numerical gradient verification working

### Module 1.5
- [ ] Neural network trained from scratch
- [ ] Backpropagation implemented
- [ ] MNIST accuracy >85%

### Module 1.6
- [ ] XGBoost model trained
- [ ] RAPIDS cuML speedup measured
- [ ] Know when to use classical vs deep learning

### Module 1.7
- [ ] Tensor autograd implemented
- [ ] All layers, losses, optimizers working
- [ ] Test coverage >80%
- [ ] MNIST accuracy >95%

---

## 🎓 What You'll Be Able to Do

After completing Domain 1, you will:

1. **Operate DGX Spark** — Launch containers, monitor resources, troubleshoot issues
2. **Write Efficient Code** — Use vectorization and GPU acceleration
3. **Understand Math** — Apply chain rule, compute gradients, optimize functions
4. **Build Neural Networks** — Implement from scratch or use frameworks effectively
5. **Choose Wisely** — Know when classical ML beats deep learning
6. **Debug Effectively** — Verify gradients numerically, test thoroughly

---

## 🚀 Preparing for Domain 2

Domain 1 builds the foundation that Domain 2 (Deep Learning Frameworks) depends on:

| Domain 1 Skill | Domain 2 Application |
|----------------|----------------------|
| Container management | PyTorch NGC containers |
| NumPy vectorization | Understanding tensor operations |
| GPU basics | Multi-GPU training |
| Math foundations | Understanding optimizer behavior |
| Manual backprop | Appreciating autograd |
| Classical ML baselines | Model comparison |

With Domain 1 complete, you'll understand what PyTorch does under the hood—making you a more effective deep learning practitioner.

---

## 📖 Study Resources

Each module includes:
- **QUICKSTART.md** — 5-minute hands-on introduction
- **STUDY_GUIDE.md** — Learning roadmap and objectives
- **QUICK_REFERENCE.md** — Commands and code patterns

Selected modules also include:
- **ELI5.md** — Jargon-free explanations for complex concepts
- **LAB_PREP.md** — Environment setup checklist
- **TROUBLESHOOTING.md** — Common issues and solutions
- **FAQ.md** — Frequently asked questions

---

## ⏭️ Next Domain

After completing Domain 1, proceed to:

**[Domain 2: Deep Learning Frameworks](../domain-2-deep-learning-frameworks/)**

Where you'll master PyTorch for production deep learning work.
