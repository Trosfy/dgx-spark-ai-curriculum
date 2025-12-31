# Module 2.1: Deep Learning with PyTorch - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Build complex neural networks** using PyTorch's `nn.Module` system
2. **Implement custom datasets** with efficient data loading pipelines
3. **Create custom autograd functions** for novel operations
4. **Use mixed precision training** (BF16) for speed and memory efficiency
5. **Profile and optimize** PyTorch training loops
6. **Implement robust checkpointing** with early stopping and resume capability

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-2.1.1-custom-module-lab.ipynb | nn.Module architecture | ~2 hr | ResNet-18 from scratch |
| 2 | lab-2.1.2-dataset-pipeline.ipynb | Data loading | ~2 hr | Optimized DataLoader |
| 3 | lab-2.1.3-autograd-deep-dive.ipynb | Custom autograd | ~2 hr | Custom activation function |
| 4 | lab-2.1.4-mixed-precision-training.ipynb | AMP training | ~2 hr | BF16 training comparison |
| 5 | lab-2.1.5-profiling-workshop.ipynb | Performance analysis | ~2 hr | Profiling report |
| 6 | lab-2.1.6-checkpointing-system.ipynb | Training robustness | ~2 hr | Checkpoint manager |

**Total time**: ~12-15 hours

---

## Core Concepts

This module introduces these fundamental ideas:

### nn.Module
**What**: Base class for all neural network modules in PyTorch
**Why it matters**: Everything you build will inherit from this - understanding it deeply prevents bugs and enables custom architectures
**First appears in**: Lab 2.1.1

### DataLoader
**What**: Utility for batching, shuffling, and loading data in parallel
**Why it matters**: Data loading is often the bottleneck - efficient pipelines can 2-3x training speed on DGX Spark
**First appears in**: Lab 2.1.2

### Autograd
**What**: PyTorch's automatic differentiation engine
**Why it matters**: Enables gradient computation for any computational graph; custom functions let you implement novel operations
**First appears in**: Lab 2.1.3

### Mixed Precision (AMP)
**What**: Training with lower precision (BF16) while maintaining accuracy
**Why it matters**: On DGX Spark's Blackwell GPU, BF16 is native - use it for 2x memory savings and faster training
**First appears in**: Lab 2.1.4

### torch.profiler
**What**: Built-in tool for profiling PyTorch operations
**Why it matters**: Can't optimize what you can't measure - profiling reveals CPU vs GPU bottlenecks
**First appears in**: Lab 2.1.5

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 1.7              ►   Module 2.1              ►  Module 2.2
MicroGrad                   PyTorch                    Computer Vision
(build from scratch)        (professional tools)       (apply to images)
```

**Builds on**:
- **Neural network fundamentals** from Module 1.5 (now use nn.Module instead of manual implementation)
- **MicroGrad experience** from Module 1.7 (understand what PyTorch automates for you)
- **GPU basics** from Module 1.3 (now use GPU professionally)

**Prepares for**:
- **Module 2.2** will use these skills to build CNN architectures
- **Module 2.3** will extend to transformers and attention
- **All subsequent modules** assume PyTorch fluency

---

## Recommended Approach

### Standard Path (12-15 hours)
1. **Start with Lab 2.1.1** - Establishes nn.Module patterns you'll use everywhere
2. **Work through 2.1.2** - Data loading is critical; optimize for DGX Spark
3. **Complete 2.1.3** - Deep autograd understanding (helps debugging)
4. **Master 2.1.4** - BF16 is essential for DGX Spark
5. **Apply 2.1.5** - Profile everything you build
6. **Finish 2.1.6** - Never lose training progress again

### Quick Path (6-8 hours, if experienced)
1. Skim Lab 2.1.1 - Focus on ResNet block implementation
2. Focus on Lab 2.1.2 - DGX Spark optimization tips
3. Skip to Lab 2.1.4 - BF16 is the key skill here
4. Complete Lab 2.1.5 - Profiling is universally useful

### Deep-Dive Path (15+ hours)
1. Complete all labs thoroughly
2. Implement additional architectures (VGG, DenseNet) in Lab 2.1.1
3. Create custom datasets for your own data in Lab 2.1.2
4. Implement multiple custom activations in Lab 2.1.3

---

## Lab-by-Lab Summary

### Lab 2.1.1: Custom Module Lab
**Goal**: Build ResNet-18 from scratch using nn.Module
**Key skills**:
- `nn.Module` subclassing with `__init__` and `forward`
- Proper parameter initialization (`nn.init`)
- Skip connections (residual blocks)
- State dict saving/loading

### Lab 2.1.2: Dataset Pipeline
**Goal**: Create efficient data loading optimized for DGX Spark
**Key skills**:
- Custom `Dataset` class implementation
- `DataLoader` with `num_workers` optimization
- `transforms.Compose` for augmentation
- Benchmarking data loading speed

### Lab 2.1.3: Autograd Deep Dive
**Goal**: Implement custom autograd function with verified gradients
**Key skills**:
- `torch.autograd.Function` with `forward` and `backward`
- Gradient verification with `torch.autograd.gradcheck`
- Hooks for gradient inspection
- Comparing with built-in implementations

### Lab 2.1.4: Mixed Precision Training
**Goal**: Use AMP for memory-efficient, fast training
**Key skills**:
- `torch.amp.autocast` with BF16
- Understanding when GradScaler is needed (not for BF16!)
- Memory comparison FP32 vs BF16
- Accuracy verification after mixed precision

### Lab 2.1.5: Profiling Workshop
**Goal**: Identify and fix training bottlenecks
**Key skills**:
- `torch.profiler` usage
- Chrome trace export and visualization
- Identifying CPU vs GPU bottlenecks
- Optimizing based on profiling data

### Lab 2.1.6: Checkpointing System
**Goal**: Build robust training that can recover from interruptions
**Key skills**:
- Saving complete training state (model + optimizer + scheduler)
- Best model tracking
- Early stopping implementation
- Resume from checkpoint

---

## Before You Start

- See [PREREQUISITES.md](./PREREQUISITES.md) for skill self-check
- See [LAB_PREP.md](./LAB_PREP.md) for environment setup
- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first success

---

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "My DataLoader is slow" | Increase `num_workers`, use `pin_memory=True` |
| "Out of memory during training" | Use mixed precision (BF16), reduce batch size |
| "Gradients are NaN" | Check learning rate, use gradient clipping |
| "Can't resume training correctly" | Make sure you save optimizer state too |
| "Model accuracy dropped with AMP" | Rare with BF16; check for numerical instability |

---

## Success Metrics

You've mastered this module when you can:

- [ ] Build a custom architecture from an architecture diagram
- [ ] Create a data pipeline that doesn't bottleneck training
- [ ] Implement a new operation with correct gradients
- [ ] Train models 2x faster with mixed precision
- [ ] Identify and fix training bottlenecks with profiling
- [ ] Resume training from any checkpoint
