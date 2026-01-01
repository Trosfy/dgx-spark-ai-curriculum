# Module 1.5: Neural Network Fundamentals - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
A working neural network layer with forward and backward passes.

## ✅ Before You Start
- [ ] Completed Module 1.4 (math foundations)
- [ ] NGC PyTorch container running

## 🚀 Let's Go!

### Step 1: Create a Linear Layer
```python
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)
        self.x = None  # Store for backward

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T
```

### Step 2: Create ReLU Activation
```python
class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
```

### Step 3: Test Forward Pass
```python
# Create layers
layer1 = Linear(4, 8)
relu = ReLU()
layer2 = Linear(8, 2)

# Forward pass
x = np.random.randn(3, 4)  # batch of 3, 4 features
h = layer1.forward(x)
h = relu.forward(h)
out = layer2.forward(h)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
```

**Expected output:**
```
Input shape: (3, 4)
Output shape: (3, 2)
```

### Step 4: Test Backward Pass
```python
# Fake gradient from loss (in real training, this comes from loss function)
dout = np.ones_like(out)

# Backward through network
dh = layer2.backward(dout)
dh = relu.backward(dh)
dx = layer1.backward(dh)

print(f"Gradient shapes: dW1={layer1.dW.shape}, dW2={layer2.dW.shape}")
print("✅ Backward pass complete!")
```

**Expected output:**
```
Gradient shapes: dW1=(4, 8), dW2=(8, 2)
✅ Backward pass complete!
```

## 🎉 You Did It!

You just:
- ✅ Built a Linear layer with forward/backward
- ✅ Implemented ReLU activation
- ✅ Ran a complete forward-backward pass
- ✅ Computed gradients for learning

In the full module, you'll:
- Train on MNIST to >95% accuracy
- Compare activation functions
- Implement BatchNorm and Dropout
- Debug common training issues

## ▶️ Next Steps
1. **Understand the concepts**: Read [ELI5.md](./ELI5.md)
2. **See implementation patterns**: Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
3. **Start Lab 1**: Open `labs/lab-1.5.1-numpy-neural-network.ipynb`
