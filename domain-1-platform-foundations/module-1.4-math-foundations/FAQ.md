# Module 1.4: Mathematics for Deep Learning - FAQ

## Frequently Asked Questions

### General Questions

#### Q: Do I need to be good at math to complete this module?

**A:** You need basic calculus (derivatives, chain rule) and linear algebra (matrix multiplication). If you can compute d/dx[x²] = 2x and multiply two matrices, you have the prerequisites. The module builds intuition—you don't need to prove theorems.

---

#### Q: Why implement backpropagation manually when PyTorch does it automatically?

**A:** Three reasons:

1. **Debugging:** When your model doesn't learn, understanding backprop helps you diagnose why (vanishing gradients, wrong loss function, etc.)

2. **Custom layers:** If you need to implement a novel operation, you may need to define its gradient

3. **Intuition:** Knowing what `loss.backward()` actually does makes you a better practitioner

---

#### Q: How is this connected to actual LLM training?

**A:** Everything in this module scales directly:

| This Module | LLM Training |
|-------------|--------------|
| 3-layer MLP backprop | Transformer backprop (same chain rule) |
| Adam optimizer | AdamW (same algorithm + weight decay) |
| SVD for LoRA intuition | LoRA fine-tuning of 70B+ models |
| Cross-entropy loss | Next-token prediction loss |

---

### Backpropagation (Lab 1.4.1)

#### Q: Why do we multiply matrices in a specific order during backprop?

**A:** Matrix dimensions must align. For a layer z = Wx + b:

- Forward: W is (out_dim, in_dim), x is (in_dim, 1) → z is (out_dim, 1)
- Backward: To get dL/dW with shape (out_dim, in_dim), we compute:
  - dL/dz @ x.T = (out_dim, 1) @ (1, in_dim) = (out_dim, in_dim) ✓

---

#### Q: What's the difference between gradient and derivative?

**A:** They're the same concept in different dimensions:

- **Derivative:** Rate of change for a function of ONE variable (f: R → R)
- **Gradient:** Vector of partial derivatives for a function of MULTIPLE variables (f: Rⁿ → R)

In neural networks, we have many parameters, so we use "gradient" (the vector of all partial derivatives).

---

#### Q: Why does ReLU work when its derivative is 0 or 1?

**A:** The derivative being exactly 0 for negative inputs means those neurons don't contribute to learning for that input—but they might for other inputs where they're positive. This "dying ReLU" issue is why variants like LeakyReLU exist, but standard ReLU works remarkably well in practice.

---

### Optimizers (Lab 1.4.2)

#### Q: Why is Adam the default optimizer for most projects?

**A:** Adam combines two powerful ideas:

1. **Momentum:** Smooths out noisy gradients by averaging
2. **Adaptive learning rates:** Different learning rates per parameter

This means it works well "out of the box" without extensive learning rate tuning. For transformers, AdamW (Adam with decoupled weight decay) is even better.

---

#### Q: What learning rate should I use?

**A:** Rules of thumb:

| Optimizer | Starting Learning Rate |
|-----------|----------------------|
| SGD | 0.01 - 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam/AdamW | 0.001 (1e-3) |

For transformers specifically: 1e-4 to 3e-4 for pretraining, 1e-5 to 3e-5 for fine-tuning.

---

#### Q: What's the difference between Adam and AdamW?

**A:** Weight decay implementation:

- **Adam:** Weight decay is added to the gradient (L2 regularization)
- **AdamW:** Weight decay is applied directly to weights (decoupled)

AdamW is generally better for transformers. The difference matters most with adaptive learning rates.

---

### Loss Landscapes (Lab 1.4.3)

#### Q: Are real neural network loss landscapes actually this smooth?

**A:** No! Real landscapes are:

- Much higher dimensional (millions of parameters)
- More complex with many local minima
- Surprisingly, often still navigable due to "mode connectivity"

The 2D visualizations are projections that help build intuition but don't capture full complexity.

---

#### Q: Why do wider networks have smoother landscapes?

**A:** More parameters = more "directions to escape" local minima. Research shows:

1. Overparameterized networks have more paths to good solutions
2. Skip connections (ResNet) make landscapes dramatically smoother
3. This partly explains why bigger models are easier to train

---

### SVD and LoRA (Lab 1.4.4)

#### Q: How does LoRA achieve 96%+ parameter reduction?

**A:** The key insight: weight UPDATES during fine-tuning are low-rank.

- Original weight W: 768×768 = 590,000 parameters
- LoRA: B (768×16) + A (16×768) = 24,576 parameters
- We only train B and A, not W

The math: If the "direction" of adaptation is low-dimensional (rank 16 captures it), we don't need all 590K parameters.

---

#### Q: What rank should I use for LoRA?

**A:** Common choices:

| Task | Typical Rank |
|------|--------------|
| Simple adaptation | 4-8 |
| General fine-tuning | 16-32 |
| Complex tasks | 64-128 |

Start with rank=16 and adjust based on results. Higher rank = more capacity but more parameters.

---

#### Q: Why initialize B to zero in LoRA?

**A:** So the model starts identical to the pretrained version:

- W_new = W_original + B @ A
- If B = 0, then W_new = W_original
- Training gradually moves away from the pretrained initialization

This is crucial for stable fine-tuning—you don't want random changes at the start.

---

### Probability (Lab 1.4.5)

#### Q: Why is cross-entropy better than MSE for classification?

**A:** Gradient properties:

- **MSE gradient:** 2(p - y) — small when p is far from y and wrong!
- **Cross-entropy gradient:** (p - y) / (p(1-p)) — LARGE when p is wrong

Cross-entropy provides stronger learning signal when the model is very wrong, leading to faster initial learning.

---

#### Q: What's the connection between temperature and softmax?

**A:** Temperature scales logits before softmax:

```python
probs = softmax(logits / temperature)
```

- Temperature = 1: Normal softmax
- Temperature < 1: Sharper distribution (more confident)
- Temperature > 1: Flatter distribution (more random)

This is used in LLM sampling to control creativity vs. determinism.

---

#### Q: Why do we use KL divergence instead of just comparing probabilities?

**A:** KL divergence has information-theoretic meaning:

- It measures the "extra bits" needed to encode data from P using a code optimized for Q
- It's asymmetric: KL(P||Q) ≠ KL(Q||P), which captures that mistaking rare events for common is worse than vice versa
- It's always non-negative (≥ 0), with 0 only when P = Q

---

## Still Have Questions?

1. Check the [ELI5.md](./ELI5.md) for intuitive explanations
2. Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for formulas
3. See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common errors
4. Look at solution notebooks in `solutions/` directory
