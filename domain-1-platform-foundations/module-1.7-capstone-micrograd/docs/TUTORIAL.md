# MicroGrad+ Getting Started Tutorial

Welcome to MicroGrad+! This tutorial will walk you through building and training your first neural network using our from-scratch deep learning library.

## Table of Contents

1. [Installation](#installation)
2. [Your First Tensor](#your-first-tensor)
3. [Understanding Autograd](#understanding-autograd)
4. [Building a Neural Network](#building-a-neural-network)
5. [Training on Real Data](#training-on-real-data)
6. [Next Steps](#next-steps)

---

## Installation

MicroGrad+ requires only NumPy:

```bash
# Navigate to the module directory
cd domain-1-platform-foundations/module-1.7-capstone-micrograd

# The package is ready to use - just import it
python -c "from micrograd_plus import Tensor; print('Success!')"
```

For visualization (optional):
```bash
pip install matplotlib
```

---

## Your First Tensor

Tensors are the building blocks of neural networks - multi-dimensional arrays that can track gradients.

```python
from micrograd_plus import Tensor
import numpy as np

# Create tensors from different sources
t1 = Tensor([1, 2, 3])                    # From list
t2 = Tensor(np.array([[1, 2], [3, 4]]))   # From numpy
t3 = Tensor(5.0)                           # Scalar

print(f"t1 shape: {t1.shape}")  # (3,)
print(f"t2 shape: {t2.shape}")  # (2, 2)
print(f"t3 value: {t3.item()}")  # 5.0
```

### Basic Operations

Tensors support familiar mathematical operations:

```python
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Arithmetic
print(f"a + b = {a + b}")     # [5, 7, 9]
print(f"a * b = {a * b}")     # [4, 10, 18]
print(f"a ** 2 = {a ** 2}")   # [1, 4, 9]

# Reductions
c = Tensor([[1, 2], [3, 4]])
print(f"sum = {c.sum()}")     # 10
print(f"mean = {c.mean()}")   # 2.5
```

---

## Understanding Autograd

The magic of MicroGrad+ is **automatic differentiation**. When you enable `requires_grad=True`, the tensor tracks all operations so it can compute gradients.

### Simple Example

```python
# Enable gradient tracking
x = Tensor([2.0], requires_grad=True)

# Build computation: y = x^2
y = x ** 2

# Compute gradients
y.backward()

# dy/dx = 2x = 4
print(f"x.grad = {x.grad}")  # [4.0]
```

### Chain Rule in Action

```python
x = Tensor([2.0], requires_grad=True)

# Complex computation: y = (3x + 2)^2
y = ((x * 3) + 2) ** 2

y.backward()

# By chain rule: dy/dx = 2(3x + 2) * 3 = 6(3*2 + 2) = 48
print(f"x.grad = {x.grad}")  # [48.0]
```

### Why This Matters

This is exactly how neural networks learn! During training:
1. Forward pass: compute predictions
2. Backward pass: compute gradients of loss w.r.t. parameters
3. Update: adjust parameters in direction that reduces loss

---

## Building a Neural Network

Let's build a simple neural network layer by layer.

### Step 1: Create Layers

```python
from micrograd_plus import Linear, ReLU, Sequential

# Individual layers
linear1 = Linear(4, 8)   # 4 inputs → 8 outputs
relu = ReLU()            # Activation function
linear2 = Linear(8, 2)   # 8 inputs → 2 outputs

# Forward pass manually
x = Tensor(np.random.randn(3, 4).astype(np.float32))  # 3 samples, 4 features
h = linear1(x)           # (3, 8)
h = relu(h)              # (3, 8)
out = linear2(h)         # (3, 2)

print(f"Input: {x.shape} → Output: {out.shape}")
```

### Step 2: Use Sequential Container

```python
# Much cleaner!
model = Sequential(
    Linear(4, 8),
    ReLU(),
    Linear(8, 2)
)

out = model(x)
print(f"Output: {out.shape}")

# Access parameters
params = model.parameters()
print(f"Total parameters: {sum(p.size for p in params)}")
```

### Step 3: Add Regularization

```python
from micrograd_plus import Dropout

model = Sequential(
    Linear(4, 8),
    ReLU(),
    Dropout(0.2),    # Drop 20% of neurons during training
    Linear(8, 2)
)

# Training mode (dropout active)
model.train()
out1 = model(x)

# Evaluation mode (dropout disabled)
model.eval()
out2 = model(x)
```

---

## Training on Real Data

Let's train a classifier on synthetic data.

### Generate Data

```python
from micrograd_plus import Tensor, CrossEntropyLoss, Adam
from micrograd_plus.utils import set_seed

set_seed(42)

# Create spiral dataset (3 classes)
def make_spiral(n_points=100, n_classes=3):
    X = np.zeros((n_points * n_classes, 2), dtype=np.float32)
    y = np.zeros(n_points * n_classes, dtype=np.int32)

    for c in range(n_classes):
        ix = range(n_points * c, n_points * (c + 1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(c * 4, (c + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c

    return X, y

X, y = make_spiral()
print(f"Data shape: {X.shape}, Labels: {np.unique(y)}")
```

### Build Model

```python
model = Sequential(
    Linear(2, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 3)    # 3 classes
)

loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)
```

### Training Loop

```python
# Training
num_epochs = 200

for epoch in range(num_epochs):
    # Forward pass
    X_tensor = Tensor(X, requires_grad=True)
    y_tensor = Tensor(y)

    logits = model(X_tensor)
    loss = loss_fn(logits, y_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update parameters
    optimizer.step()

    # Track progress
    if epoch % 20 == 0:
        predictions = np.argmax(logits.data, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy:.2%}")

print("\nTraining complete!")
```

### Evaluate

```python
# Final evaluation
model.eval()
logits = model(Tensor(X))
predictions = np.argmax(logits.data, axis=1)
final_accuracy = np.mean(predictions == y)
print(f"Final Accuracy: {final_accuracy:.2%}")
```

---

## Complete Example: MNIST Classifier

Here's a complete example training on MNIST handwritten digits:

```python
from micrograd_plus import (
    Tensor, Linear, ReLU, Dropout, Sequential,
    CrossEntropyLoss, Adam
)
from micrograd_plus.utils import set_seed, DataLoader
import numpy as np

# Set seed for reproducibility
set_seed(42)

# Load MNIST (assuming data is downloaded)
# X_train: (60000, 784), y_train: (60000,)
# X_test: (10000, 784), y_test: (10000,)

# Build model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)

# Setup training
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    model.train()

    # Shuffle data
    indices = np.random.permutation(len(X_train))
    total_loss = 0
    n_batches = 0

    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = Tensor(X_train[batch_idx], requires_grad=True)
        y_batch = Tensor(y_train[batch_idx])

        # Forward
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    # Evaluate
    model.eval()
    test_logits = model(Tensor(X_test[:1000]))
    test_preds = np.argmax(test_logits.data, axis=1)
    test_acc = np.mean(test_preds == y_test[:1000])

    print(f"Epoch {epoch+1}: Loss={total_loss/n_batches:.4f}, Test Acc={test_acc:.2%}")
```

---

## Tips and Best Practices

### 1. Always Zero Gradients

```python
# Before each backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 2. Use Train/Eval Modes

```python
model.train()  # Enable dropout, batch norm training mode
# ... training ...

model.eval()   # Disable dropout, use running stats
# ... evaluation ...
```

### 3. Verify Gradients

When implementing new operations, check against numerical gradients:

```python
from micrograd_plus.utils import gradient_check

x = Tensor([1, 2, 3], requires_grad=True)
passed, error = gradient_check(lambda t: (t ** 2).sum(), x)
print(f"Gradient check: {'PASS' if passed else 'FAIL'} (error={error:.2e})")
```

### 4. Monitor Training

```python
history = {'loss': [], 'accuracy': []}

for epoch in range(epochs):
    # ... training code ...
    history['loss'].append(loss.item())
    history['accuracy'].append(accuracy)

# Plot curves
import matplotlib.pyplot as plt
plt.plot(history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

---

## Next Steps

Now that you understand the basics:

1. **Explore the notebooks**: Work through Labs 1.7.1-1.7.6 for deeper understanding
2. **Read the code**: Check out `tensor.py`, `layers.py` to see how it all works
3. **Experiment**: Try different architectures, optimizers, learning rates
4. **Move to PyTorch**: Apply these concepts with a production framework

Congratulations on completing your journey through MicroGrad+! You now understand the fundamentals of how deep learning works under the hood. 🎉
