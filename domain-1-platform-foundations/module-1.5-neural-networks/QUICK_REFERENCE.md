# Module 1.5: Neural Network Fundamentals - Quick Reference

## 🧱 Layer Implementations

### Linear Layer
```python
class Linear:
    def __init__(self, in_features, out_features):
        # He initialization for ReLU
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T
```

### ReLU
```python
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
```

### Softmax + Cross-Entropy (Combined)
```python
class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        # Numerical stability
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        self.probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        self.y_true = y_true

        # Cross-entropy loss
        n = logits.shape[0]
        loss = -np.sum(y_true * np.log(self.probs + 1e-8)) / n
        return loss

    def backward(self):
        n = self.probs.shape[0]
        return (self.probs - self.y_true) / n
```

## 🎯 Activation Functions

### All Activations
```python
# ReLU
def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

# Leaky ReLU
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def leaky_relu_grad(x, alpha=0.01): return np.where(x > 0, 1, alpha)

# Sigmoid
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)

# Tanh
def tanh(x): return np.tanh(x)
def tanh_grad(x): return 1 - np.tanh(x)**2

# GELU (Gaussian Error Linear Unit)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# SiLU / Swish
def silu(x): return x * sigmoid(x)
```

## 🎚️ Normalization

### Batch Normalization
```python
class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        self.x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta
```

### Layer Normalization
```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta
```

## 🔧 Regularization

### L2 Regularization
```python
def l2_loss(params, lambda_):
    """Add to total loss."""
    return lambda_ * sum(np.sum(p**2) for p in params) / 2

def l2_grad(param, lambda_):
    """Add to parameter gradient."""
    return lambda_ * param
```

### Dropout
```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, dout):
        return dout * self.mask
```

## 🔢 Initialization

### Xavier (for Sigmoid/Tanh)
```python
W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / (fan_in + fan_out))
```

### He (for ReLU)
```python
W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
```

## 📉 Training Loop

### Basic Training Loop
```python
def train(model, X_train, y_train, epochs=10, lr=0.01, batch_size=64):
    n = len(X_train)
    losses = []

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            output = model.forward(X_batch)
            loss = model.compute_loss(output, y_batch)

            # Backward pass
            model.backward()

            # Update weights
            model.update(lr)

            epoch_loss += loss * len(X_batch)

        losses.append(epoch_loss / n)
        print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")

    return losses
```

## 🔍 Debugging Checklist

### Can't Overfit Single Batch?
```python
# Test: Should get loss near 0 on one batch
model.train_on_batch(X[:32], y[:32], epochs=100)
# If not: Bug in forward/backward pass
```

### Loss Doesn't Decrease?
```python
# Try 10x larger learning rate
lr = 0.1  # instead of 0.01
# If still stuck: gradient issue or bug
```

### Loss Explodes?
```python
# Reduce learning rate
lr = 0.001  # instead of 0.01

# Add gradient clipping
grad = np.clip(grad, -1, 1)
```

### Validation Loss Increases While Training Decreases?
```python
# Overfitting! Add regularization:
# - L2 regularization
# - Dropout
# - Early stopping
# - More data / data augmentation
```

## 📊 MNIST Architecture

```python
# Recommended architecture
model = Sequential([
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10),
])
loss_fn = SoftmaxCrossEntropy()
optimizer = Adam(model.parameters(), lr=0.001)
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgot to zero gradients | Reset grads before backward |
| Wrong axis in softmax | Use axis=1 for batch |
| Not using mask in dropout backward | Store mask from forward |
| BatchNorm in eval mode during training | Set training=True |
| Numerical overflow in softmax | Subtract max before exp |

## 🔗 Quick Links
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n Backprop](https://cs231n.github.io/optimization-2/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
