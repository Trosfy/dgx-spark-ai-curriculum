# MicroGrad+ API Reference

Complete API documentation for the MicroGrad+ neural network library.

## Table of Contents

- [Tensor](#tensor)
- [Layers](#layers)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Training Utilities](#training-utilities)
- [Utility Functions](#utility-functions)

---

## Tensor

The `Tensor` class is the fundamental data structure in MicroGrad+. It wraps a numpy array and provides automatic differentiation.

### Constructor

```python
Tensor(data, requires_grad=False, dtype=np.float32)
```

**Parameters:**
- `data`: Array-like, numpy array, or scalar
- `requires_grad`: If True, gradients will be computed during backward pass
- `dtype`: Data type for the tensor (default: float32)

**Example:**
```python
from micrograd_plus import Tensor

# From list
t1 = Tensor([1, 2, 3])

# From numpy array
t2 = Tensor(np.array([[1, 2], [3, 4]]))

# With gradient tracking
t3 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
```

### Properties

| Property | Description |
|----------|-------------|
| `data` | The underlying numpy array |
| `grad` | Gradient array (None if requires_grad=False) |
| `shape` | Tuple of tensor dimensions |
| `dtype` | Data type of elements |
| `requires_grad` | Whether gradient computation is enabled |
| `T` | Transpose of the tensor |

### Arithmetic Operations

All operations support automatic differentiation:

| Operation | Description | Example |
|-----------|-------------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Element-wise multiplication | `a * b` |
| `/` | Element-wise division | `a / b` |
| `**` | Power | `a ** 2` |
| `@` | Matrix multiplication | `a @ b` |
| `-a` | Negation | `-a` |

### Methods

#### `backward(gradient=None)`
Compute gradients via reverse-mode automatic differentiation.

```python
x = Tensor([1, 2, 3], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2, 4, 6]
```

#### `zero_grad()`
Reset gradient to zeros.

#### `detach()`
Return a new tensor detached from the computation graph.

#### `clone()`
Return a copy that shares the computation graph.

#### `item()`
Extract scalar value from single-element tensor.

#### `numpy()`
Return a copy as numpy array.

### Reduction Operations

| Method | Description |
|--------|-------------|
| `sum(axis=None)` | Sum over elements |
| `mean(axis=None)` | Mean over elements |
| `max(axis=None)` | Maximum over elements |
| `min(axis=None)` | Minimum over elements |

### Shape Operations

| Method | Description |
|--------|-------------|
| `reshape(*shape)` | Reshape to new dimensions |
| `flatten()` | Flatten to 1D |
| `transpose(*axes)` | Transpose axes |
| `unsqueeze(axis)` | Add dimension |
| `squeeze(axis=None)` | Remove size-1 dimensions |

### Activation Functions

| Method | Description |
|--------|-------------|
| `relu()` | ReLU: max(0, x) |
| `sigmoid()` | Sigmoid: 1/(1+e^-x) |
| `tanh()` | Hyperbolic tangent |
| `softmax(axis=-1)` | Softmax normalization |
| `log_softmax(axis=-1)` | Log-softmax (numerically stable) |

### Math Functions

| Method | Description |
|--------|-------------|
| `exp()` | Exponential |
| `log()` | Natural logarithm |
| `sqrt()` | Square root |
| `abs()` | Absolute value |

---

## Layers

All layers inherit from `Module` base class.

### Module (Base Class)

```python
class Module:
    def forward(self, x: Tensor) -> Tensor
    def parameters(self) -> List[Tensor]
    def train(mode=True) -> Module
    def eval() -> Module
    def zero_grad()
```

### Linear

Fully connected layer: y = xW + b

```python
Linear(in_features, out_features, bias=True)
```

**Parameters:**
- `in_features`: Size of input
- `out_features`: Size of output
- `bias`: Include bias term

**Example:**
```python
layer = Linear(784, 256)
x = Tensor(np.random.randn(32, 784))
y = layer(x)  # (32, 256)
```

### ReLU

Rectified Linear Unit: max(0, x)

```python
relu = ReLU()
y = relu(x)
```

### Sigmoid

Sigmoid activation: 1/(1+e^-x)

```python
sigmoid = Sigmoid()
y = sigmoid(x)
```

### Tanh

Hyperbolic tangent activation.

```python
tanh = Tanh()
y = tanh(x)
```

### Softmax

Softmax normalization.

```python
softmax = Softmax(axis=-1)
y = softmax(x)  # Sums to 1 along axis
```

### Dropout

Dropout regularization.

```python
Dropout(p=0.5)
```

**Parameters:**
- `p`: Probability of dropping elements

**Note:** Behaves differently in train vs eval mode.

### BatchNorm

Batch normalization.

```python
BatchNorm(num_features, eps=1e-5, momentum=0.1)
```

### LayerNorm

Layer normalization.

```python
LayerNorm(normalized_shape, eps=1e-5)
```

### Embedding

Lookup table for embeddings.

```python
Embedding(num_embeddings, embedding_dim)
```

### Sequential

Container for stacking layers.

```python
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 10)
)

y = model(x)
params = model.parameters()
```

---

## Loss Functions

### MSELoss

Mean Squared Error for regression.

```python
MSELoss(reduction='mean')
```

**Parameters:**
- `reduction`: 'mean', 'sum', or 'none'

**Example:**
```python
loss_fn = MSELoss()
loss = loss_fn(predictions, targets)
```

### CrossEntropyLoss

Cross-entropy for classification.

```python
CrossEntropyLoss(reduction='mean')
```

**Note:** Expects raw logits, not softmax probabilities.

**Example:**
```python
loss_fn = CrossEntropyLoss()
loss = loss_fn(logits, class_indices)
```

### BCELoss

Binary Cross-Entropy (expects probabilities).

```python
BCELoss(reduction='mean')
```

### BCEWithLogitsLoss

Binary Cross-Entropy with built-in sigmoid.

```python
BCEWithLogitsLoss(reduction='mean')
```

### L1Loss

Mean Absolute Error.

```python
L1Loss(reduction='mean')
```

### HuberLoss

Smooth L1 Loss.

```python
HuberLoss(delta=1.0, reduction='mean')
```

---

## Optimizers

### SGD

Stochastic Gradient Descent.

```python
SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False)
```

**Parameters:**
- `params`: List of parameters to optimize
- `lr`: Learning rate
- `momentum`: Momentum factor
- `weight_decay`: L2 regularization
- `nesterov`: Use Nesterov momentum

**Methods:**
- `step()`: Update parameters
- `zero_grad()`: Clear gradients

### Adam

Adam optimizer.

```python
Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
```

### AdamW

Adam with decoupled weight decay.

```python
AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

### RMSprop

RMSprop optimizer.

```python
RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0)
```

### Learning Rate Schedulers

```python
StepLR(optimizer, step_size, gamma=0.1)
ExponentialLR(optimizer, gamma)
CosineAnnealingLR(optimizer, T_max, eta_min=0)
```

---

## Training Utilities

### DataLoader

Simple data loader for batching.

```python
DataLoader(X, y, batch_size=32, shuffle=True)
```

**Example:**
```python
loader = DataLoader(X_train, y_train, batch_size=64)
for batch_x, batch_y in loader:
    # Training step
```

### train_epoch

Train model for one epoch.

```python
train_epoch(model, dataloader, loss_fn, optimizer, verbose=False)
```

### evaluate

Evaluate model on dataset.

```python
evaluate(model, dataloader, loss_fn, metrics=None)
```

### accuracy

Compute classification accuracy.

```python
acc = accuracy(predictions, targets)
```

### count_parameters

Count trainable parameters.

```python
n_params = count_parameters(model)
```

### save_model / load_model

Save and load model parameters.

```python
save_model(model, "model.npz")
load_model(model, "model.npz")
```

---

## Utility Functions

### set_seed

Set random seed for reproducibility.

```python
set_seed(42)
```

### numerical_gradient

Compute numerical gradient via finite differences.

```python
grad = numerical_gradient(f, x, eps=1e-5)
```

### gradient_check

Verify analytical vs numerical gradients.

```python
passed, error = gradient_check(f, x)
```

### get_memory_usage

Get current memory usage.

```python
mem = get_memory_usage()
print(f"Memory: {mem['rss']:.1f} MB")
```

### one_hot

Convert class indices to one-hot encoding.

```python
one_hot_labels = one_hot(labels, num_classes)
```

### normalize

Normalize data to zero mean and unit variance.

```python
X_norm, mean, std = normalize(X)
```

### train_test_split

Split data into train and test sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## Quick Reference

### Complete Training Loop

```python
from micrograd_plus import (
    Tensor, Linear, ReLU, Sequential,
    CrossEntropyLoss, Adam
)

# Build model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Setup
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # Forward
        logits = model(Tensor(X_batch, requires_grad=True))
        loss = loss_fn(logits, Tensor(y_batch))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    val_loss, metrics = evaluate(model, val_loader, loss_fn)
    print(f"Epoch {epoch}: Loss={val_loss:.4f}")
```
