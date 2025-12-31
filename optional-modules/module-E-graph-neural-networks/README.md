# Optional Module E: Graph Neural Networks

**Category:** Optional - Structured Data
**Duration:** 6-8 hours
**Prerequisites:** Module 1.5 (Neural Networks), Module 2.1 (PyTorch)

---

## Overview

Not all data fits into grids (images) or sequences (text). Molecules, social networks, citation graphs, and knowledge graphs are naturally represented as graphs - nodes connected by edges. Graph Neural Networks (GNNs) extend deep learning to this rich data structure, enabling predictions about nodes, edges, and entire graphs.

**Why This Matters:** GNNs power drug discovery (molecular property prediction), social network analysis (friend recommendations), fraud detection (transaction networks), and knowledge graph reasoning. Understanding GNNs opens doors to these high-impact applications.

### The Kitchen Table Explanation

Imagine you're predicting who will become friends on a social network. For each person, you could look at their individual profile (age, interests). But more useful is looking at *who they already know* - their network neighborhood. People with overlapping friends are likely to become friends themselves. GNNs formalize this intuition: to understand a node, aggregate information from its neighbors. Then repeat - aggregate from neighbors of neighbors. This creates learned representations that capture network structure.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Understand message passing in graph neural networks
- ✅ Implement GCN, GraphSAGE, and GAT architectures
- ✅ Train GNNs for node classification and graph classification
- ✅ Use PyTorch Geometric for real-world graph tasks

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| E.1 | Explain the message passing framework for GNNs | Understand |
| E.2 | Implement Graph Convolutional Networks from scratch | Apply |
| E.3 | Apply attention mechanisms in Graph Attention Networks | Apply |
| E.4 | Train graph-level classifiers using pooling operations | Apply |

---

## Topics

### E.1 Graph Fundamentals

- **Graph Representations**
  - Adjacency matrix vs edge list
  - Node features, edge features
  - Directed vs undirected graphs

- **Graph Tasks**
  - Node classification
  - Link prediction
  - Graph classification
  - Node embedding

- **Challenges with Graphs**
  - Irregular structure (not grids)
  - Permutation invariance
  - Variable size graphs

### E.2 Message Passing Neural Networks

- **The MPNN Framework**
  - Message: m_ij = M(h_i, h_j, e_ij)
  - Aggregate: m_i = ⊕{m_ij for j ∈ N(i)}
  - Update: h_i = U(h_i, m_i)

- **Graph Convolutional Networks (GCN)**
  - Spectral convolutions on graphs
  - Simplified GCN: mean aggregation
  - Layer-wise propagation

- **Neighborhood Sampling**
  - Mini-batch training for large graphs
  - GraphSAGE sampling strategy
  - Scalability considerations

### E.3 Attention in Graphs

- **Graph Attention Networks (GAT)**
  - Attention weights over neighbors
  - Multi-head attention
  - Learning importance of edges

- **Transformer Variants**
  - Graph Transformers
  - Positional encodings for graphs
  - Global attention vs local attention

### E.4 Graph Pooling and Classification

- **Readout Operations**
  - Mean/sum/max pooling
  - Set2Set
  - Attention-based pooling

- **Hierarchical Pooling**
  - DiffPool: differentiable pooling
  - Top-k pooling
  - Graph coarsening

- **Applications**
  - Molecular property prediction
  - Social network classification
  - Document classification (as graphs)

---

## Labs

### Lab E.1: PyTorch Geometric Setup
**Time:** 1 hour

Get familiar with PyG and explore graph datasets.

**Instructions:**
1. Install PyTorch Geometric (compatible with ARM64)
2. Load Cora citation network
3. Visualize graph structure with NetworkX
4. Explore node features and labels
5. Understand PyG Data objects
6. Create train/val/test masks

**Deliverable:** Notebook with dataset exploration and visualization

---

### Lab E.2: GCN from Scratch
**Time:** 2 hours

Implement Graph Convolutional Network and train on Cora.

**Instructions:**
1. Implement GCN layer with adjacency multiplication
2. Add self-loops and normalization
3. Stack 2-3 GCN layers with ReLU
4. Train for node classification on Cora
5. Visualize learned embeddings with t-SNE
6. Achieve >80% accuracy on test set

**Deliverable:** Working GCN with training visualization

---

### Lab E.3: Graph Attention Networks
**Time:** 2 hours

Implement GAT and compare to GCN.

**Instructions:**
1. Implement attention mechanism for graphs
2. Add multi-head attention
3. Train on Cora and compare to GCN
4. Visualize attention weights
5. Analyze which edges get high attention
6. Achieve >81% accuracy on Cora

**Deliverable:** GAT implementation with attention analysis

---

### Lab E.4: Graph Classification
**Time:** 2 hours

Build graph-level classifier for molecular property prediction.

**Instructions:**
1. Load MUTAG or PROTEINS dataset
2. Implement global pooling (mean, max)
3. Build GNN + pooling + MLP classifier
4. Train for graph classification
5. Implement attention-based pooling
6. Compare pooling strategies

**Deliverable:** Graph classifier with pooling comparison

---

## Guidance

### PyTorch Geometric Installation

```bash
# Install PyG (works on DGX Spark ARM64)
pip install torch-geometric

# Additional dependencies for visualization
pip install networkx matplotlib
```

### Understanding PyG Data Objects

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Node feature dimension: {data.num_features}")
print(f"Number of classes: {dataset.num_classes}")

# Data structure:
# - data.x: node features [num_nodes, num_features]
# - data.edge_index: edges [2, num_edges] - source/target node indices
# - data.y: node labels [num_nodes]

print(f"\nNode features shape: {data.x.shape}")
print(f"Edge index shape: {data.edge_index.shape}")
print(f"Labels shape: {data.y.shape}")

# Visualize a subgraph
G = to_networkx(data, to_undirected=True)
subgraph_nodes = list(range(50))  # First 50 nodes
subgraph = G.subgraph(subgraph_nodes)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(subgraph, pos, node_color=data.y[subgraph_nodes].numpy(),
        cmap=plt.cm.Set3, node_size=100, with_labels=False)
plt.title("Cora Subgraph (colored by class)")
plt.show()
```

### GCN Implementation from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer.

    h' = σ(D^(-1/2) Ã D^(-1/2) H W)

    Where:
    - Ã = A + I (adjacency with self-loops)
    - D = degree matrix
    - H = node features
    - W = learnable weights
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        # Add self-loops: A → A + I
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization: D^(-1/2)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transformation
        x = self.linear(x)

        # Message passing: aggregate normalized neighbors
        out = torch.zeros_like(x)
        for i, (src, dst) in enumerate(edge_index.t()):
            out[dst] += norm[i] * x[src]

        return out

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network.
    """

    def __init__(self, num_features, num_classes, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNLayer(num_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Training
model = GCN(dataset.num_features, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs

for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
```

### Graph Attention Networks (GAT)

```python
class GATLayer(nn.Module):
    """
    Graph Attention Layer.

    Computes attention weights α_ij for each edge,
    then aggregates: h'_i = σ(Σ_j α_ij W h_j)
    """

    def __init__(self, in_channels, out_channels, heads=8, concat=True, dropout=0.6):
        super().__init__()
        self.heads = heads
        self.concat = concat

        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.a)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.out_channels = out_channels

    def forward(self, x, edge_index):
        N = x.size(0)
        H, C = self.heads, self.out_channels

        # Linear transformation: [N, in] → [N, H*C]
        x = self.W(x).view(N, H, C)  # [N, H, C]

        # Compute attention coefficients
        row, col = edge_index  # source, target

        # α_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        alpha_l = (x[row] * self.a[:, :C]).sum(dim=-1)  # [E, H]
        alpha_r = (x[col] * self.a[:, C:]).sum(dim=-1)  # [E, H]
        alpha = self.leaky_relu(alpha_l + alpha_r)

        # Softmax over neighbors
        alpha = self.softmax_by_node(alpha, col, N)
        alpha = self.dropout(alpha)

        # Weighted aggregation
        out = torch.zeros(N, H, C, device=x.device)
        for i, (src, dst) in enumerate(edge_index.t()):
            out[dst] += alpha[i].unsqueeze(-1) * x[src]

        if self.concat:
            return out.view(N, H * C)
        else:
            return out.mean(dim=1)

    def softmax_by_node(self, alpha, index, num_nodes):
        """Softmax over incoming edges for each node."""
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(alpha),
                                   alpha, reduce='amax')
        alpha = alpha - alpha_max[index]
        alpha_exp = alpha.exp()

        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha), alpha_exp)

        return alpha_exp / (alpha_sum[index] + 1e-16)

class GAT(nn.Module):
    """Two-layer Graph Attention Network."""

    def __init__(self, num_features, num_classes, hidden_dim=8, heads=8):
        super().__init__()
        self.conv1 = GATLayer(num_features, hidden_dim, heads=heads)
        self.conv2 = GATLayer(hidden_dim * heads, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### Graph Classification with Pooling

```python
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class GraphClassifier(nn.Module):
    """
    GNN for graph-level classification.

    1. Apply GNN layers to get node embeddings
    2. Pool node embeddings to graph embedding
    3. Classify graph embedding
    """

    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Pooling: combine mean and max
        x_mean = global_mean_pool(x, batch)  # [batch_size, hidden]
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classify
        return self.classifier(x)

# Load molecular dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
print(f"Number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")

# DataLoader handles batching graphs
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = GraphClassifier(dataset.num_features, hidden_dim=64, num_classes=dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")
```

### Using PyG High-Level API

```python
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# PyG provides optimized implementations
class PyGGCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GraphSAGE for large graphs (sampling-based)
class GraphSAGE(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, 64)
        self.conv2 = SAGEConv(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# GAT with PyG
class PyGGAT(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GATConv(num_features, 8, heads=8)
        self.conv2 = GATConv(64, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

### DGX Spark for Large Graphs

> **DGX Spark Tip:** Large graphs (millions of nodes) require:
> - Mini-batch training with neighbor sampling (GraphSAGE style)
> - Efficient sparse matrix operations on GPU
> - 128GB unified memory allows loading large graphs entirely
> - Consider RAPIDS cuGraph for GPU-accelerated graph algorithms

---

## Milestone Checklist

- [ ] PyG installed and Cora explored
- [ ] GCN achieving >80% on Cora
- [ ] GAT implemented with attention visualization
- [ ] Graph classifier working on MUTAG
- [ ] Understand message passing framework
- [ ] Can explain when to use GCN vs GAT vs GraphSAGE

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory on large graphs | Use mini-batch sampling, reduce hidden dim |
| Oversmoothing with many layers | Use skip connections, limit to 2-3 layers |
| Attention weights all equal | Check attention initialization, try more heads |
| Graph batching confusion | PyG handles this - trust data.batch |

---

## Why This Module is Optional

GNNs are specialized for graph-structured data. Most AI practitioners work with text, images, or tabular data. However, GNN knowledge is valuable for:

1. **Drug Discovery** - Molecular property prediction is a killer app
2. **Knowledge Graphs** - Combine with RAG for structured retrieval
3. **Fraud Detection** - Transaction networks, social networks
4. **Research** - Active area with many open problems

---

## Next Steps

After completing this module:
1. Apply GNNs to your capstone if dealing with relational data
2. Explore knowledge graph embeddings for RAG enhancement
3. Consider molecular ML if interested in drug discovery

---

## Resources

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [CS224W: Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/) - Stanford course
- [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/) - Free online
- [GCN Paper](https://arxiv.org/abs/1609.02907) - Original GCN
- [GAT Paper](https://arxiv.org/abs/1710.10903) - Graph Attention Networks
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216) - Sampling-based GNN
- [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_overview/) - Benchmarks

