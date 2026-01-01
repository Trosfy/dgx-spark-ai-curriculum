# Module E: Graph Neural Networks - Quick Reference

## 🔧 PyTorch Geometric Essentials

### Installation

```bash
pip install torch-geometric

# Additional dependencies (usually auto-installed)
pip install torch-scatter torch-sparse
```

### Data Structure

```python
from torch_geometric.data import Data

# Graph data object
data = Data(
    x=node_features,       # [num_nodes, num_features]
    edge_index=edges,      # [2, num_edges] - COO format
    y=labels,              # [num_nodes] or [num_graphs]
    edge_attr=edge_features,  # [num_edges, edge_features] (optional)
)

# Key attributes
data.num_nodes           # Number of nodes
data.num_edges           # Number of edges
data.num_features        # Node feature dimension
data.is_directed()       # Check if directed
```

### Edge Index Format

```python
# Edge index is a [2, num_edges] tensor
# Row 0: source nodes
# Row 1: target nodes

edge_index = torch.tensor([
    [0, 1, 1, 2],  # Source nodes
    [1, 0, 2, 1]   # Target nodes
])
# Edges: 0→1, 1→0, 1→2, 2→1

# Add self-loops
from torch_geometric.utils import add_self_loops
edge_index, _ = add_self_loops(edge_index, num_nodes=3)
```

---

## 📊 Common Architectures

### GCN (Graph Convolutional Network)

```python
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### GAT (Graph Attention Network)

```python
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

### GraphSAGE

```python
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
```

### Graph Classification (with Pooling)

```python
from torch_geometric.nn import global_mean_pool, global_max_pool

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pool: combine mean and max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.fc(x)
```

---

## 🎯 Message Passing Formula

### General Framework

```
h_i^(l+1) = UPDATE(h_i^(l), AGGREGATE({h_j^(l) : j ∈ N(i)}))
```

| Component | GCN | GAT | GraphSAGE |
|-----------|-----|-----|-----------|
| AGGREGATE | Normalized sum | Attention-weighted sum | Mean/Max/LSTM |
| UPDATE | Linear + nonlinearity | Linear + nonlinearity | Concat + linear |

### GCN Layer Math

```
h_i^(l+1) = σ(Σ_j (1/√(d_i·d_j)) · W · h_j^(l))
```

Where:
- `d_i, d_j` = degrees of nodes i, j
- `W` = learnable weight matrix
- `σ` = activation (ReLU)

---

## 📦 Common Datasets

```python
from torch_geometric.datasets import Planetoid, TUDataset

# Node classification
cora = Planetoid(root='/tmp/Cora', name='Cora')        # 2,708 nodes, 7 classes
citeseer = Planetoid(root='/tmp/Citeseer', name='CiteSeer')
pubmed = Planetoid(root='/tmp/Pubmed', name='PubMed')  # 19,717 nodes

# Graph classification
mutag = TUDataset(root='/tmp/MUTAG', name='MUTAG')     # 188 molecules
proteins = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')  # 1,113 proteins
enzymes = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')     # 600 enzymes
```

### Dataset Statistics

| Dataset | Nodes | Edges | Features | Classes | Task |
|---------|-------|-------|----------|---------|------|
| Cora | 2,708 | 10,556 | 1,433 | 7 | Node classification |
| MUTAG | ~18/graph | ~40/graph | 7 | 2 | Graph classification |
| PROTEINS | ~39/graph | ~146/graph | 3 | 2 | Graph classification |

---

## 🔄 Training Patterns

### Node Classification

```python
# Transductive: train/test on same graph
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
```

### Graph Classification

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for data in loader:
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
```

### Mini-Batch for Large Graphs

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # Sample 25 first-hop, 10 second-hop
    batch_size=128,
    input_nodes=data.train_mask
)

for batch in loader:
    out = model(batch.x, batch.edge_index)[:batch.batch_size]
    loss = F.cross_entropy(out, batch.y[:batch.batch_size])
```

---

## 📐 Key Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Hidden dim | 16-256 | 64 is common |
| Num layers | 2-3 | More = oversmoothing |
| Dropout | 0.3-0.6 | Higher for GAT |
| Learning rate | 0.001-0.01 | Adam optimizer |
| Weight decay | 5e-4 | Regularization |
| GAT heads | 4-8 | Multi-head attention |

---

## ⚠️ Common Pitfalls

| Problem | Solution |
|---------|----------|
| Oversmoothing (deep GNNs) | Limit to 2-3 layers, add skip connections |
| OOM on large graphs | Use NeighborLoader, reduce hidden dim |
| Poor performance | Check edge index direction, add self-loops |
| Graph batching confusion | Use `data.batch` for pooling |

---

## 🔧 Utility Functions

### Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

def visualize_graph(data, num_nodes=100):
    G = to_networkx(data, to_undirected=True)
    G = G.subgraph(list(range(num_nodes)))

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color=data.y[:num_nodes].numpy(),
            cmap=plt.cm.Set3, node_size=50)
    plt.show()
```

### Check Graph Stats

```python
def graph_stats(data):
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Avg degree: {data.num_edges / data.num_nodes:.1f}")
    print(f"Features: {data.num_features}")
    print(f"Has isolated nodes: {data.has_isolated_nodes()}")
    print(f"Has self-loops: {data.has_self_loops()}")
```

---

## 🔗 Quick Links

- Lab 1: PyTorch Geometric Setup
- Lab 2: GCN from Scratch
- Lab 3: Graph Attention Networks
- Lab 4: Graph Classification
- [PyG Documentation](https://pytorch-geometric.readthedocs.io/)
- [GCN Paper](https://arxiv.org/abs/1609.02907)
- [GAT Paper](https://arxiv.org/abs/1710.10903)
