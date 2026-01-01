# Module E: Graph Neural Networks - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build

A simple graph neural network that predicts paper topics in a citation network - understanding how "similar papers cite each other."

## ✅ Before You Start

- [ ] PyTorch installed
- [ ] `pip install torch-geometric` completed

## 🚀 Let's Go!

### Step 1: Load a Real Graph Dataset

```python
from torch_geometric.datasets import Planetoid

# Cora: A citation network of ML papers
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"📊 Cora Citation Network:")
print(f"   {data.num_nodes} papers (nodes)")
print(f"   {data.num_edges} citations (edges)")
print(f"   {data.num_features} word features per paper")
print(f"   {dataset.num_classes} topics to predict")
```

**Expected output:**
```
📊 Cora Citation Network:
   2708 papers (nodes)
   10556 citations (edges)
   1433 word features per paper
   7 topics to predict
```

### Step 2: Build a Simple GCN

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, dataset.num_classes)
print(f"Model: {model}")
```

### Step 3: Train!

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

**Expected output:**
```
Epoch 0: Loss = 1.9456
Epoch 50: Loss = 0.3234
Epoch 100: Loss = 0.1876
Epoch 150: Loss = 0.1234
```

### Step 4: Test Accuracy!

```python
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())
print(f"\n✅ Test Accuracy: {accuracy:.1%}")
```

**Expected output:**
```
✅ Test Accuracy: 81.2%
```

## 🎉 You Did It!

You just trained a Graph Neural Network that:
1. **Reads** paper features (word vectors)
2. **Aggregates** information from cited/citing papers
3. **Predicts** the research topic

The magic: Papers with similar topics cite each other, and GNNs exploit this structure!

## ▶️ Next Steps

1. **Understand message passing**: See ELI5.md and Notebook 01
2. **Implement GCN from scratch**: Notebook 02
3. **Add attention**: Build GAT in Notebook 03
4. **Graph classification**: Predict molecular properties in Notebook 04

---

## 💡 The Key Insight

> **GNNs learn by aggregating neighbor information.**
>
> To understand a node, look at its neighbors.
> To understand those neighbors, look at THEIR neighbors.
> Stack layers = look further into the network.
>
> This is why papers about "transformers" cluster together - they cite each other!
