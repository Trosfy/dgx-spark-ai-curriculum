# Module 1.7: Capstone - MicroGrad+ - Quick Reference

## 🔢 Value Class (Scalar Autograd)

### Basic Structure
```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

### Operations with Backward
```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward
    return out

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out

def __pow__(self, n):
    out = Value(self.data ** n, (self,), f'**{n}')

    def _backward():
        self.grad += n * (self.data ** (n - 1)) * out.grad
    out._backward = _backward
    return out

def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out
```

## 🔄 Backward Pass (Topological Sort)

```python
def backward(self):
    # Build topological order
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)

    # Backpropagate
    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

## 📊 Tensor Class (NumPy-based)

### Basic Structure
```python
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
```

### Matrix Multiplication
```python
def __matmul__(self, other):
    out = Tensor(
        self.data @ other.data,
        requires_grad=self.requires_grad or other.requires_grad,
        _children=(self, other),
        _op='@'
    )

    def _backward():
        if self.requires_grad:
            self.grad += out.grad @ other.data.T
        if other.requires_grad:
            other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out
```

### Sum and Mean
```python
def sum(self, axis=None, keepdims=False):
    out = Tensor(
        np.sum(self.data, axis=axis, keepdims=keepdims),
        requires_grad=self.requires_grad,
        _children=(self,),
        _op='sum'
    )

    def _backward():
        if self.requires_grad:
            # Broadcast gradient back
            grad = np.ones_like(self.data) * out.grad
            self.grad += grad
    out._backward = _backward
    return out

def mean(self, axis=None, keepdims=False):
    n = self.data.size if axis is None else self.data.shape[axis]
    return self.sum(axis=axis, keepdims=keepdims) / n
```

## 🏗️ Layers

### Linear Layer
```python
class Linear:
    def __init__(self, in_features, out_features):
        # Xavier initialization
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
        )

    def __call__(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]
```

### ReLU Activation
```python
class ReLU:
    def __call__(self, x):
        out = Tensor(
            np.maximum(0, x.data),
            requires_grad=x.requires_grad,
            _children=(x,),
            _op='relu'
        )

        def _backward():
            if x.requires_grad:
                x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out

    def parameters(self):
        return []
```

### Softmax
```python
class Softmax:
    def __call__(self, x):
        # Numerically stable softmax
        exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op='softmax')

        def _backward():
            if x.requires_grad:
                # Jacobian-vector product
                s = out.data
                x.grad += out.grad * s - s * np.sum(out.grad * s, axis=-1, keepdims=True)
        out._backward = _backward
        return out

    def parameters(self):
        return []
```

### Dropout
```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def __call__(self, x):
        if not self.training:
            return x

        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        out = Tensor(
            x.data * mask,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op='dropout'
        )

        def _backward():
            if x.requires_grad:
                x.grad += out.grad * mask
        out._backward = _backward
        return out

    def parameters(self):
        return []
```

## 📉 Loss Functions

### MSE Loss
```python
class MSELoss:
    def __call__(self, pred, target):
        diff = pred.data - target.data
        loss = Tensor(
            np.mean(diff ** 2),
            requires_grad=pred.requires_grad,
            _children=(pred,),
            _op='mse'
        )

        def _backward():
            if pred.requires_grad:
                pred.grad += 2 * diff / diff.size * loss.grad
        loss._backward = _backward
        return loss
```

### Cross-Entropy Loss
```python
class CrossEntropyLoss:
    def __call__(self, logits, targets):
        # Stable softmax + NLL
        exp_logits = np.exp(logits.data - np.max(logits.data, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        n = logits.data.shape[0]
        log_probs = np.log(probs[np.arange(n), targets.data] + 1e-10)
        loss = Tensor(-np.mean(log_probs), requires_grad=True, _children=(logits,), _op='ce')

        def _backward():
            if logits.requires_grad:
                grad = probs.copy()
                grad[np.arange(n), targets.data] -= 1
                logits.grad += grad / n * loss.grad
        loss._backward = _backward
        return loss
```

## ⚡ Optimizers

### SGD with Momentum
```python
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad
            p.data += self.velocities[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
```

### Adam
```python
class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
```

## 🔧 Sequential Model

```python
class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
```

## ✅ Gradient Checking

```python
def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient for verification."""
    grad = np.zeros_like(x.data)

    for i in np.ndindex(x.data.shape):
        old_val = x.data[i]

        x.data[i] = old_val + eps
        f_plus = f().data

        x.data[i] = old_val - eps
        f_minus = f().data

        grad[i] = (f_plus - f_minus) / (2 * eps)
        x.data[i] = old_val

    return grad

def check_gradient(f, x, eps=1e-5, tol=1e-4):
    """Verify autograd gradient against numerical gradient."""
    # Compute autograd gradient
    out = f()
    out.backward()
    autograd_grad = x.grad.copy()

    # Compute numerical gradient
    x.grad = np.zeros_like(x.data)
    num_grad = numerical_gradient(f, x, eps)

    # Compare
    diff = np.abs(autograd_grad - num_grad)
    max_diff = np.max(diff)

    return max_diff < tol, max_diff
```

## 🎯 MNIST Training Loop

```python
def train_mnist(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        total_loss = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = Tensor(X_train[batch_idx], requires_grad=False)
            y_batch = Tensor(y_train[batch_idx])

            # Forward
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            total_loss += loss.data

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_logits = model(Tensor(X_val))
        val_pred = np.argmax(val_logits.data, axis=1)
        val_acc = np.mean(val_pred == y_val)

        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Val Acc={val_acc:.4f}")
```

## 🔗 Quick Links
- [Karpathy's micrograd](https://github.com/karpathy/micrograd)
- [PyTorch autograd docs](https://pytorch.org/docs/stable/autograd.html)
- [Backpropagation explained](https://colah.github.io/posts/2015-08-Backprop/)

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Gradient explosion | Use gradient clipping or lower learning rate |
| Gradients all zero | Check ReLU dying, verify backward implementation |
| Wrong gradient shape | Ensure broadcasting handled correctly |
| Numerical instability | Use log-sum-exp trick for softmax |
| Memory issues | Clear intermediate tensors, reduce batch size |
