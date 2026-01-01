# FAQ - Module 1.5: Neural Network Fundamentals

**Module:** 1.5 - Neural Network Fundamentals
**Domain:** 1 - Platform Foundations

---

## General Questions

### Q: Why do we implement neural networks from scratch in NumPy first?

**A:** Building neural networks from scratch gives you deep understanding of:

1. **Forward propagation**: How data flows through layers
2. **Backpropagation**: How gradients flow backward (it's just the chain rule!)
3. **Weight updates**: How optimizers modify parameters
4. **Numerical stability**: Why we need epsilon values and careful initialization

This knowledge is invaluable when debugging PyTorch/TensorFlow models. When something goes wrong, you'll understand what's happening under the hood.

---

### Q: Why can't I pip install PyTorch on DGX Spark?

**A:** DGX Spark uses ARM64 (aarch64) architecture, not x86. The standard PyTorch pip wheels are compiled for x86 and won't work.

**Solution:** Use the NGC container which has PyTorch pre-compiled for ARM64:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

---

### Q: What's the difference between He and Xavier initialization?

**A:** Both maintain stable gradients, but for different activations:

| Initialization | Formula | Best For |
|----------------|---------|----------|
| **He** | `W = randn() * sqrt(2/fan_in)` | ReLU, Leaky ReLU |
| **Xavier/Glorot** | `W = randn() * sqrt(2/(fan_in + fan_out))` | Tanh, Sigmoid |

**Why it matters:** Wrong initialization can cause vanishing or exploding gradients. He initialization accounts for ReLU zeroing out half the activations.

---

### Q: When should I use BatchNorm vs LayerNorm vs RMSNorm?

**A:**

| Method | Normalizes Over | Best For | Used In |
|--------|-----------------|----------|---------|
| **BatchNorm** | Batch dimension | CNNs, vision | ResNet, EfficientNet |
| **LayerNorm** | Feature dimension | Transformers, RNNs | BERT, GPT |
| **RMSNorm** | Feature (no mean) | Modern LLMs | LLaMA, Mistral |

**Key differences:**
- BatchNorm needs batch size > 1 (different behavior train vs inference)
- LayerNorm works with any batch size (same behavior always)
- RMSNorm is faster than LayerNorm (no mean computation)

---

### Q: What learning rate should I start with?

**A:** Common starting points:

| Optimizer | Starting LR |
|-----------|-------------|
| SGD | 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam | 0.001 |
| AdamW | 0.001 |

**Tuning strategy:**
1. Start with default
2. If loss explodes → decrease by 10x
3. If loss barely moves → increase by 10x
4. Use learning rate finder for fine-tuning

---

### Q: What's the "overfit one batch" trick?

**A:** Before training on full data, verify your model can memorize a single batch:

```python
# Take one small batch
X_batch = X_train[:32]
y_batch = y_train[:32]

# Train for many iterations on just this batch
for i in range(200):
    loss = train_step(model, X_batch, y_batch)

# Should achieve ~100% accuracy on this batch
```

**If it fails:** There's a bug in your forward or backward pass. This is the most important debugging technique!

---

## DGX Spark Specific Questions

### Q: Why does DGX Spark have unified memory?

**A:** Traditional systems have separate CPU RAM and GPU VRAM, requiring slow data transfers. DGX Spark's 128GB unified memory means:

1. **No transfers needed**: CPU and GPU share the same memory
2. **Larger models**: 70B+ parameter models fit entirely in memory
3. **Faster development**: No need to optimize memory transfers
4. **Simpler code**: Less memory management complexity

This is possible because of the NVIDIA Grace-Blackwell architecture.

---

### Q: What's special about the GB10 Blackwell chip?

**A:** Key features:

| Feature | Specification |
|---------|---------------|
| CUDA Cores | 6,144 |
| Tensor Cores | 192 (5th generation) |
| NVFP4 Performance | 1 PFLOP |
| FP8 Performance | ~209 TFLOPS |
| BF16 Performance | ~100 TFLOPS |

**Why it matters for AI:**
- Native FP8 and NVFP4 support for efficient inference
- Tensor Cores accelerate matrix operations
- 5th gen Tensor Cores are 2x faster than previous generation

---

### Q: What batch size should I use on DGX Spark?

**A:** DGX Spark's 128GB unified memory allows larger batch sizes than typical GPUs:

| Model Size | Recommended Batch Size |
|------------|----------------------|
| Small MLP | 256-2048 |
| Medium CNN | 128-512 |
| Large Transformer | 32-128 |
| LLM (7B) | 8-32 |
| LLM (70B) | 1-4 |

**Finding optimal:**
```python
# Start large and reduce if OOM
for batch_size in [2048, 1024, 512, 256, 128]:
    try:
        train_epoch(model, batch_size)
        print(f"Batch size {batch_size} works!")
        break
    except RuntimeError:
        print(f"Batch size {batch_size} OOM")
```

---

## Training Questions

### Q: My loss is stuck and not decreasing. What should I check?

**A:** Debugging checklist:

1. **Learning rate too low?** Try increasing by 10x
2. **Vanishing gradients?** Check gradient magnitudes, switch to ReLU
3. **Data issue?** Visualize samples, check for NaN values
4. **Labels shuffled?** Verify X and y correspond correctly
5. **Stuck in local minimum?** Add momentum or use Adam

---

### Q: What's the difference between train and validation loss behavior?

**A:**

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| Both high, not moving | Underfitting | Bigger model, more training |
| Train ↓, Val ↓ | Good training | Keep going! |
| Train ↓, Val ↑ | Overfitting | Add regularization, more data |
| Both oscillating | LR too high | Reduce learning rate |

---

### Q: Should I use dropout or L2 regularization?

**A:** Both prevent overfitting but work differently:

| Technique | How It Works | When to Use |
|-----------|--------------|-------------|
| **Dropout** | Randomly zeros neurons during training | Large models, dense layers |
| **L2 (Weight Decay)** | Penalizes large weights | Always reasonable default |
| **Both** | Complementary effects | Very large models |

**Typical values:**
- Dropout: 0.1-0.5 (start with 0.2)
- L2/Weight Decay: 0.01-0.0001 (start with 0.01)

---

## Architecture Questions

### Q: How many layers/neurons should my network have?

**A:** Rules of thumb for MLPs:

| Problem Complexity | Layers | Neurons per Layer |
|-------------------|--------|------------------|
| Simple (XOR) | 1-2 | 4-16 |
| Medium (MNIST) | 2-3 | 128-512 |
| Complex (ImageNet) | Use CNNs | - |

**Start simple and add complexity only if needed.** An overfit simple model is easier to regularize than an underfit complex one.

---

### Q: Why do modern LLMs use RMSNorm instead of LayerNorm?

**A:** RMSNorm is simpler and faster:

```python
# LayerNorm: center then scale
x_norm = (x - mean) / std * gamma + beta

# RMSNorm: just scale (no centering)
x_norm = x / rms * gamma
```

**Benefits:**
- ~10% faster (no mean computation)
- Empirically works just as well for transformers
- Used in LLaMA, Mistral, and most modern LLMs

---

## Next Steps

After completing this module:

1. **Module 1.6**: Classical ML Foundations
2. **Module 2.1**: Deep Learning with PyTorch
3. **Module 2.2**: Computer Vision (CNNs)
4. **Module 2.3**: NLP & Transformers

---

*FAQ for DGX Spark AI Curriculum v2.0*
