# Module 1.5: Neural Network Fundamentals

**Domain:** 1 - Platform Foundations
**Duration:** Week 5 (12-15 hours)
**Prerequisites:** Module 1.4 (Mathematics)

---

## Overview

This module bridges theory and practice. You'll build neural networks from scratch using only NumPy, understanding every component before moving to frameworks. By the end, you'll have trained a model on real data and know how to diagnose common training issues.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Build neural networks from scratch using only NumPy
- ✅ Explain the purpose of each neural network component
- ✅ Train networks on real datasets and diagnose common issues
- ✅ Implement regularization and normalization techniques

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.5.1 | Implement fully-connected layers with forward and backward passes | Apply |
| 1.5.2 | Explain vanishing/exploding gradients and implement solutions | Understand |
| 1.5.3 | Apply regularization techniques (L2, dropout) to prevent overfitting | Apply |
| 1.5.4 | Diagnose training issues from loss curves and metrics | Analyze |

---

## Topics

### 1.5.1 Perceptron to MLP
- Single neuron and activation functions
- Multi-layer perceptrons
- Universal approximation theorem

### 1.5.2 Activation Functions
- Sigmoid, Tanh, ReLU, GELU, SiLU
- Vanishing gradient problem
- Choosing activations for different tasks

### 1.5.3 Loss Functions
- MSE for regression
- Cross-entropy for classification
- Custom losses

### 1.5.4 Regularization
- L1/L2 regularization
- Dropout
- Early stopping
- Data augmentation concepts

### 1.5.5 Normalization
- Batch normalization
- Layer normalization
- RMSNorm (used in modern transformers)

### 1.5.6 Weight Initialization
- Xavier/Glorot initialization
- He initialization
- Impact on training dynamics

---

## Labs

### Lab 1.5.1: NumPy Neural Network
**Time:** 4 hours

Build a complete MLP from scratch.

**Instructions:**
1. Implement `Linear` layer (forward + backward)
2. Implement `ReLU` activation (forward + backward)
3. Implement `Softmax` + `CrossEntropyLoss`
4. Implement `SGD` optimizer
5. Create training loop
6. Train on MNIST to >95% accuracy
7. Plot training curves

**Deliverable:** Working NumPy MLP with >95% MNIST accuracy

---

### Lab 1.5.2: Activation Function Study
**Time:** 2 hours

Compare activation functions empirically.

**Instructions:**
1. Implement 6 activations: Sigmoid, Tanh, ReLU, LeakyReLU, GELU, SiLU
2. Plot each function and its derivative
3. Train same network architecture with each
4. Compare training speed and final accuracy
5. Document which activations cause vanishing gradients

**Deliverable:** Activation comparison notebook with recommendations

---

### Lab 1.5.3: Regularization Experiments
**Time:** 2 hours

Understand overfitting and regularization.

**Instructions:**
1. Create a dataset where overfitting is easy (small train, large test)
2. Train without regularization → observe overfitting
3. Add L2 regularization with varying λ values
4. Add Dropout with varying rates
5. Create visualization: underfitting ↔ good fit ↔ overfitting
6. Find optimal regularization strength

**Deliverable:** Regularization analysis notebook

---

### Lab 1.5.4: Normalization Comparison
**Time:** 2 hours

Implement and compare normalization techniques.

**Instructions:**
1. Implement BatchNorm from scratch (forward + backward)
2. Implement LayerNorm from scratch
3. Train same network with: no norm, BatchNorm, LayerNorm
4. Compare training stability and convergence
5. Document when to use each

**Deliverable:** Normalization implementation and comparison

---

### Lab 1.5.5: Training Diagnostics Lab
**Time:** 2 hours

Learn to debug neural networks.

**Instructions:**
1. Deliberately create problems:
   - Learning rate too high (loss explodes)
   - Learning rate too low (barely moves)
   - Vanishing gradients (deep sigmoid network)
   - Overfitting (no regularization)
2. Document symptoms for each
3. Implement fixes
4. Create a "debugging checklist"

**Deliverable:** Training diagnostics guide with examples

---

### Lab 1.5.6: GPU Acceleration
**Time:** 2 hours

Experience the GPU speedup on DGX Spark.

**Instructions:**
1. Port your NumPy MLP to PyTorch
2. Run on CPU, measure training time
3. Run on GPU, measure training time
4. Calculate speedup factor
5. Test with varying batch sizes
6. Document optimal batch size for DGX Spark

**Deliverable:** CPU vs GPU comparison notebook

---

## Guidance

### Network Architecture for MNIST

```python
# Recommended architecture
Input: 784 (28x28 flattened)
Hidden 1: 256, ReLU
Hidden 2: 128, ReLU
Output: 10, Softmax

# Hyperparameters
learning_rate = 0.01
batch_size = 64
epochs = 10
```

### Debugging Checklist

1. **Can your network overfit a single batch?**
   - If no: bug in forward/backward pass
   - Test: train on 1 batch, loss should → 0

2. **Does loss decrease at all?**
   - If no: learning rate too low, or gradient issue
   - Test: increase LR by 10x

3. **Does loss explode?**
   - If yes: learning rate too high, or numerical instability
   - Test: decrease LR by 10x, add gradient clipping

4. **Does validation loss increase while training loss decreases?**
   - Overfitting: add regularization

### Initialization Formulas

```python
# Xavier/Glorot (for tanh, sigmoid)
std = np.sqrt(2.0 / (fan_in + fan_out))

# He (for ReLU)
std = np.sqrt(2.0 / fan_in)

# Weights
W = np.random.randn(fan_in, fan_out) * std
b = np.zeros(fan_out)
```

### BatchNorm vs LayerNorm

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalizes over | Batch dimension | Feature dimension |
| Best for | CNNs, large batches | Transformers, RNNs |
| Batch size dependency | Yes | No |
| Training vs inference | Different behavior | Same behavior |

---

## Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| `torch.cuda.is_available()` returns `False` | Not using NGC container or missing `--gpus all` | Use NGC container: `docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.11-py3` |
| Loss is `NaN` | Learning rate too high or numerical instability | Lower learning rate, add epsilon to divisions |
| Loss barely decreases | Learning rate too low or vanishing gradients | Increase LR by 10x, use ReLU, check initialization |
| Kernel crashes | Out of memory | Reduce batch size, clear memory with `gc.collect()` |
| `ModuleNotFoundError` for scripts | Scripts directory not in path | Verify `sys.path.insert(0, str(scripts_dir))` is running |

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions.

---

## Milestone Checklist

- [ ] NumPy MLP achieving >95% on MNIST
- [ ] All 6 activation functions implemented and compared
- [ ] Regularization experiments documented
- [ ] BatchNorm and LayerNorm implemented from scratch
- [ ] Training diagnostics guide created
- [ ] GPU vs CPU speedup measured on DGX Spark
- [ ] All notebooks documented with explanations

---

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Build a Linear + ReLU layer in 5 minutes |
| [PREREQUISITES.md](./PREREQUISITES.md) | Required skills and self-assessment checklist |
| [ELI5.md](./ELI5.md) | Intuitive explanations of activations, backprop, overfitting |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning roadmap and concept overview |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Layer implementations and debugging checklist |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup and data preparation |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Solutions to common problems |
| [FAQ.md](./FAQ.md) | Frequently asked questions |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save layer implementations to `scripts/`
3. ➡️ Proceed to [Module 1.6: Classical ML Foundations](../module-1.6-classical-ml/)

---

## Resources

- [Neural Networks and Deep Learning (Nielsen)](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [He Initialization Paper](https://arxiv.org/abs/1502.01852)
