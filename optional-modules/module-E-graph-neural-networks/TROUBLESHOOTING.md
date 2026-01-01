# Module E: Graph Neural Networks - Troubleshooting & FAQ

## 🔍 Quick Diagnostic

**Before diving into specific errors:**
1. Check PyG installation: `import torch_geometric; print(torch_geometric.__version__)`
2. Verify edge_index format: Should be `[2, num_edges]`, not `[num_edges, 2]`
3. Check for self-loops: `data.has_self_loops()`

---

## 🚨 Error Categories

### Installation Issues

#### Error: `ModuleNotFoundError: No module named 'torch_geometric'`

**Solution:**
```bash
pip install torch-geometric

# If issues persist, install with dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch-geometric
```

---

#### Error: `RuntimeError: Detected that PyTorch and torch_geometric were compiled with different CUDA versions`

**Solution:**
```bash
# Check your CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Install matching PyG version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# Replace with your torch and CUDA versions
```

---

### Edge Index Errors

#### Error: `IndexError: index out of range in self`

**Symptoms:**
```
IndexError: index out of range in self
```

**Cause:** Edge index contains node IDs larger than `num_nodes`.

**Solution:**
```python
# Check edge index range
print(f"Max node in edge_index: {edge_index.max()}")
print(f"Num nodes in features: {x.shape[0]}")

# Fix: Ensure consistency
assert edge_index.max() < x.shape[0], "Edge index out of range!"

# Or reindex nodes
from torch_geometric.utils import to_undirected
edge_index = to_undirected(edge_index)  # Also handles issues
```

---

#### Issue: Wrong edge_index format

**Symptoms:**
- Model runs but accuracy is ~random
- Error about wrong dimensions

**Cause:** Edge index should be `[2, num_edges]`, not `[num_edges, 2]`.

**Solution:**
```python
# Check format
print(f"Edge index shape: {edge_index.shape}")  # Should be [2, N]

# Fix if needed
if edge_index.shape[0] != 2:
    edge_index = edge_index.t()  # Transpose

# Verify
assert edge_index.shape[0] == 2, "Edge index should be [2, num_edges]"
```

---

### Training Issues

#### Issue: Accuracy stuck at ~14% (random for 7 classes)

**Symptoms:**
- Training loss decreases but accuracy doesn't improve
- Accuracy around 1/num_classes

**Causes and Solutions:**

```python
# 1. Forgot to add self-loops (important for GCN!)
from torch_geometric.utils import add_self_loops
edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

# 2. Using wrong mask for evaluation
# Bad: Evaluating on training nodes
acc = (pred == data.y).mean()

# Good: Evaluating on test nodes only
acc = (pred[data.test_mask] == data.y[data.test_mask]).mean()

# 3. Model in training mode during evaluation
model.eval()  # Don't forget!
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=1)

# 4. Labels not starting from 0
print(f"Label range: {data.y.min()} to {data.y.max()}")
# Should be 0 to num_classes-1
```

---

#### Issue: Oversmoothing (accuracy drops with more layers)

**Symptoms:**
- 2-layer GNN: 80% accuracy
- 4-layer GNN: 60% accuracy
- 8-layer GNN: 40% accuracy

**Cause:** Deep GNNs smooth node features until they become indistinguishable.

**Solutions:**
```python
# 1. Limit depth (2-3 layers is usually best)
class ShallowGNN(nn.Module):
    def __init__(self, ...):
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        # No more layers!

# 2. Add skip connections
class ResGCN(nn.Module):
    def forward(self, x, edge_index):
        identity = x
        x = self.conv1(x, edge_index)
        x = x + self.proj(identity)  # Skip connection
        return x

# 3. Use jumping knowledge
from torch_geometric.nn import JumpingKnowledge
self.jk = JumpingKnowledge(mode='cat')  # Or 'max', 'lstm'

# 4. Use normalization
from torch_geometric.nn import BatchNorm
x = self.bn(self.conv1(x, edge_index))
```

---

#### Issue: OOM on large graphs

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Use mini-batch sampling
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[10, 5],  # Sample fewer neighbors
    batch_size=256,
    input_nodes=data.train_mask
)

for batch in loader:
    out = model(batch.x, batch.edge_index)[:batch.batch_size]
    loss = criterion(out, batch.y[:batch.batch_size])

# 2. Reduce hidden dimension
hidden_dim = 32  # Was 256

# 3. Use CPU for very large graphs
data = data.to('cpu')
model = model.to('cpu')

# 4. Clear memory
torch.cuda.empty_cache()
gc.collect()
```

---

### Graph Batching Issues

#### Issue: Confusion about `data.batch` in graph classification

**Symptoms:**
- Graph classification giving wrong results
- Pooling returning wrong shape

**Explanation:**
```python
# In graph classification, DataLoader batches multiple graphs
# data.batch tells you which graph each node belongs to

# Example with 3 graphs (5, 3, 4 nodes):
# data.batch = [0,0,0,0,0, 1,1,1, 2,2,2,2]
#               graph 0     graph 1  graph 2

# Global pooling uses this:
from torch_geometric.nn import global_mean_pool
graph_emb = global_mean_pool(node_emb, data.batch)
# Shape: [num_graphs, hidden_dim], not [num_nodes, hidden_dim]

# If you forget data.batch in pooling, you pool ALL nodes into one vector!
```

---

#### Error: Graph batching gives wrong batch size

**Symptoms:**
- Expected 32 graphs, got different number

**Solution:**
```python
# DataLoader batch_size is number of GRAPHS, not nodes
loader = DataLoader(dataset, batch_size=32)  # 32 graphs per batch

for data in loader:
    print(f"Nodes: {data.num_nodes}, Graphs: {data.num_graphs}")
    # Nodes varies, Graphs ≈ 32 (last batch may be smaller)
```

---

## ❓ Frequently Asked Questions

### Conceptual Questions

#### Q: What's the difference between GCN, GAT, and GraphSAGE?

**A:**

| Model | Aggregation | Strengths | Weaknesses |
|-------|-------------|-----------|------------|
| **GCN** | Normalized mean | Simple, fast | Fixed weights |
| **GAT** | Learned attention | Adaptive weights | Slower, more params |
| **GraphSAGE** | Various (mean/pool/LSTM) | Inductive, scalable | May miss structure |

**When to use:**
- Start with GCN (simplest)
- Use GAT when edges have varying importance
- Use GraphSAGE for large graphs or inductive setting

---

#### Q: What's the difference between transductive and inductive learning?

**A:**

| Transductive | Inductive |
|--------------|-----------|
| Train and test on SAME graph | Train and test on DIFFERENT graphs |
| Test nodes known during training | New nodes at test time |
| Example: Cora citation network | Example: Protein function prediction |
| All nodes in one graph | Many small graphs |

**GCN is transductive by default. GraphSAGE was designed for inductive.**

---

#### Q: Why do GNNs typically have only 2-3 layers?

**A:** Oversmoothing!

```
Layer 1: Each node sees 1-hop neighbors
Layer 2: Each node sees 2-hop neighbors
Layer 3: Each node sees 3-hop neighbors
...
Layer 10: Every node sees every other node → all features become similar!
```

**Analogy:** It's like mixing paint colors. Mix 2-3 colors → interesting. Mix 20 colors → brown mush.

---

#### Q: How do GNNs differ from transformers?

**A:**

| GNNs | Transformers |
|------|--------------|
| Local attention (neighbors) | Global attention (all tokens) |
| Structure from edges | Structure from positions |
| Nodes can have any connectivity | Usually sequential/fixed |
| Naturally handles varying graphs | Fixed sequence length (usually) |

**Transformers are like GNNs on a complete graph!** (Every node connected to every other.)

---

### Practical Questions

#### Q: How do I handle node features?

**A:**

```python
# If you have node features
data.x = node_features  # [num_nodes, feature_dim]

# If nodes have no features, use degree or one-hot
from torch_geometric.transforms import OneHotDegree
transform = OneHotDegree(max_degree=10)
data = transform(data)

# Or just use ones
data.x = torch.ones(data.num_nodes, 1)
```

---

#### Q: How do I add edge features?

**A:**

```python
# Add edge attributes
data.edge_attr = edge_features  # [num_edges, edge_feature_dim]

# Use in GNN (not all layers support this!)
from torch_geometric.nn import NNConv  # Edge-conditioned convolution

class EdgeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        nn_func = nn.Sequential(nn.Linear(edge_dim, hidden * hidden))
        self.conv = NNConv(in_dim, hidden, nn_func)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)
```

---

#### Q: How do I visualize attention in GAT?

**A:**

```python
from torch_geometric.nn import GATConv

class ExplainableGAT(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.conv = GATConv(in_dim, out_dim, heads=1)

    def forward(self, x, edge_index, return_attention=False):
        out, (edge_index_attn, attention_weights) = self.conv(
            x, edge_index, return_attention_weights=True
        )
        if return_attention:
            return out, attention_weights
        return out

# Visualize
import matplotlib.pyplot as plt

out, attn = model(data.x, data.edge_index, return_attention=True)
# attn is [num_edges, heads] - attention weight per edge
plt.hist(attn.detach().numpy().flatten(), bins=50)
plt.xlabel("Attention weight")
plt.show()
```

---

### Beyond the Basics

#### Q: How do I apply GNNs to molecules?

**A:**

```python
from torch_geometric.datasets import MoleculeNet

# Load molecular dataset
dataset = MoleculeNet(root='/tmp/HIV', name='HIV')

# Molecules have:
# - Nodes = atoms (features: atom type, charge, etc.)
# - Edges = bonds (features: bond type, conjugation, etc.)

# For molecular property prediction:
# - Use graph classification (predict property of whole molecule)
# - Pool node embeddings to get molecule embedding
# - Train on biological/chemical labels
```

---

#### Q: How do GNNs connect to knowledge graphs?

**A:**

Knowledge graphs are natural for GNNs:
- Nodes = entities (people, places, concepts)
- Edges = relationships (knows, located_in, is_a)

```python
# For knowledge graph completion:
# - Encode entities with GNN
# - Predict missing edges (link prediction)
# - Use for question answering, recommendations

from torch_geometric.nn import RGCNConv  # Relational GCN
# Handles multiple edge types (relations)
```

---

## 🔄 Reset Procedures

### Clear PyG Cache

```python
# Clear downloaded datasets
import shutil
shutil.rmtree('/tmp/Cora', ignore_errors=True)

# Redownload
dataset = Planetoid(root='/tmp/Cora', name='Cora')
```

### Reset Model

```python
def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
```

---

## 📞 Still Stuck?

1. **Check PyG docs** - Good examples for each layer
2. **Stanford CS224W** - Excellent graph ML course
3. **Visualize your graph** - Often reveals data issues
4. **Start with Cora** - Well-studied benchmark
