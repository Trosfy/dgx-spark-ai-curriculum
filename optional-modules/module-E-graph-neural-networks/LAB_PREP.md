# Module E: Graph Neural Networks - Lab Preparation Guide

## ⏱️ Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| Lab 1: PyTorch Geometric Setup | 15 min | 1 hr | 1.25 hr |
| Lab 2: GCN from Scratch | 10 min | 2 hr | 2.25 hr |
| Lab 3: Graph Attention Networks | 10 min | 2 hr | 2.25 hr |
| Lab 4: Graph Classification | 15 min | 2 hr | 2.25 hr |

**Total preparation time**: ~50 minutes

---

## 📦 Required Downloads

### Python Packages

```bash
# Core package
pip install torch-geometric

# Visualization
pip install networkx matplotlib

# For graph visualization (optional but helpful)
pip install pyvis

# Verify installation
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"
```

### Datasets (Auto-download)

Datasets download automatically on first use, but you can pre-download:

```python
from torch_geometric.datasets import Planetoid, TUDataset

# Pre-download for Labs 1-3 (node classification)
print("Downloading Cora...")
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(f"Cora: {len(dataset)} graph, {dataset[0].num_nodes} nodes")

# Pre-download for Lab 4 (graph classification)
print("Downloading MUTAG...")
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
print(f"MUTAG: {len(dataset)} graphs")

print("All datasets ready!")
```

**Dataset sizes:**
| Dataset | Download | Disk Usage |
|---------|----------|------------|
| Cora | ~5 MB | ~15 MB |
| CiteSeer | ~5 MB | ~15 MB |
| MUTAG | ~1 MB | ~3 MB |
| PROTEINS | ~5 MB | ~15 MB |

---

## 🔧 Environment Setup

### 1. Start Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Install Packages

```bash
pip install torch-geometric networkx matplotlib
```

### 3. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

**Expected output:**
```
CUDA available: True
Device: NVIDIA GH200 480GB
Memory: 128.0 GB
```

### 4. Test PyTorch Geometric

```python
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print(f"Dataset: {dataset.name}")
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Features: {data.num_features}, Classes: {dataset.num_classes}")

# Quick model test
conv = GCNConv(data.num_features, 16)
x = conv(data.x, data.edge_index)
print(f"GCN output shape: {x.shape}")
```

**Expected output:**
```
Dataset: Cora
Nodes: 2708, Edges: 10556
Features: 1433, Classes: 7
GCN output shape: torch.Size([2708, 16])
```

---

## ✅ Pre-Lab Checklists

### Lab 1: PyTorch Geometric Setup

**Prerequisites:**
- [ ] PyG installed successfully
- [ ] NetworkX and Matplotlib installed
- [ ] Cora dataset downloaded
- [ ] Basic PyTorch knowledge (Module 2.1)

**Concepts to review:**
- What is a graph (nodes, edges)
- Adjacency matrices
- Why graphs need special neural networks

**Quick test:**
```python
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx

data = Planetoid(root='/tmp/Cora', name='Cora')[0]
G = to_networkx(data, to_undirected=True)

print(f"NetworkX graph has {G.number_of_nodes()} nodes")
print(f"Average degree: {sum(d for n, d in G.degree()) / G.number_of_nodes():.2f}")
```

---

### Lab 2: GCN from Scratch

**Prerequisites:**
- [ ] Completed Lab 1
- [ ] Understanding of message passing (see ELI5.md)
- [ ] Clear on edge_index format

**Concepts to review:**
- Matrix multiplication interpretation of GCN
- Normalization in graph convolution
- Why we add self-loops

**Test your intuition:**
```python
# What does this do?
from torch_geometric.utils import add_self_loops, degree

edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
row, col = edge_index
deg = degree(row, data.num_nodes)

print(f"Degree range: {deg.min():.0f} to {deg.max():.0f}")
# Degree = number of neighbors (including self after self-loops)
```

---

### Lab 3: Graph Attention Networks

**Prerequisites:**
- [ ] Completed Labs 1-2
- [ ] GCN working correctly
- [ ] Understanding of attention mechanism

**Concepts to review:**
- Attention weights (from transformers, Module 2.3)
- Multi-head attention
- Why learned attention > fixed weights

**Memory requirement:** GAT uses more memory than GCN due to attention computation.

```python
# Memory check for GAT
import torch

# Estimate: GAT with 8 heads on Cora
# ~500 MB should be sufficient
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
# DGX Spark has 128 GB - plenty!
```

---

### Lab 4: Graph Classification

**Prerequisites:**
- [ ] Completed Labs 1-3
- [ ] Understanding of graph pooling (see ELI5.md)
- [ ] MUTAG dataset downloaded

**Additional prep:**
```python
# Download graph classification dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
print(f"Number of graphs: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Avg nodes/graph: {sum(g.num_nodes for g in dataset)/len(dataset):.1f}")

# Test DataLoader batching
loader = DataLoader(dataset[:10], batch_size=4, shuffle=False)
for batch in loader:
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} total nodes")
    break
```

---

## 🚫 Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Wrong edge_index shape | Silent incorrect results | Always verify `shape == [2, num_edges]` |
| Forgetting self-loops | Poor GCN performance | Use `add_self_loops()` |
| Not using GPU | Slow training | Always move data to CUDA |
| Wrong PyG version | Import errors | Use latest: `pip install -U torch-geometric` |
| Evaluating on train set | Inflated accuracy | Use `data.test_mask` |

---

## 📁 Expected File Structure

```
/workspace/
├── module-E-gnn/
│   ├── notebooks/
│   │   ├── 01-pytorch-geometric-setup.ipynb
│   │   ├── 02-gcn-from-scratch.ipynb
│   │   ├── 03-graph-attention-networks.ipynb
│   │   └── 04-graph-classification.ipynb
│   ├── outputs/
│   │   ├── visualizations/
│   │   └── model_checkpoints/
│   └── data/
│       └── (datasets auto-download here or /tmp)
```

---

## ⚡ Quick Setup Script

```bash
# Navigate to workspace
cd /workspace
mkdir -p module-E-gnn/{notebooks,outputs/{visualizations,model_checkpoints},data}
cd module-E-gnn

# Install packages
pip install torch-geometric networkx matplotlib

# Download datasets and verify
python -c "
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import GCNConv
import torch

print('Testing PyTorch Geometric...')

# Test node classification dataset
data = Planetoid(root='/tmp/Cora', name='Cora')[0]
print(f'✓ Cora loaded: {data.num_nodes} nodes, {data.num_edges} edges')

# Test graph classification dataset
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
print(f'✓ MUTAG loaded: {len(dataset)} graphs')

# Test GCN layer
conv = GCNConv(data.num_features, 16)
out = conv(data.x, data.edge_index)
print(f'✓ GCN layer works: output shape {out.shape}')

# Test GPU
data = data.to('cuda')
conv = conv.to('cuda')
out = conv(data.x, data.edge_index)
print(f'✓ GPU works: output on {out.device}')

print('\\n✓ All setup complete!')
"
```

---

## 🎯 Memory Guide for DGX Spark

DGX Spark's 128 GB memory is overkill for most GNN tasks, but here's a guide:

| Task | Memory Needed | DGX Spark? |
|------|---------------|------------|
| Cora (node classification) | ~1 GB | ✓✓✓ Easy |
| Large citation networks | ~5-10 GB | ✓✓ Fine |
| Molecular graph datasets | ~2-5 GB | ✓✓✓ Easy |
| OGB large graphs (100M+ edges) | ~50-100 GB | ✓ Possible! |

```python
# Check memory during training
import torch

def memory_status():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Call periodically during training
memory_status()
```

---

## 📊 Expected Baseline Results

| Lab | Target Metric | What to Achieve |
|-----|---------------|-----------------|
| Lab 1 | Setup success | All cells run without error |
| Lab 2 (GCN) | Test accuracy | >80% on Cora |
| Lab 3 (GAT) | Test accuracy | >81% on Cora |
| Lab 4 (Graph class) | Test accuracy | >80% on MUTAG |

If you're significantly below these, check:
1. Learning rate (try 0.01)
2. Hidden dimensions (try 16-64)
3. Number of layers (try 2)
4. Training epochs (try 200)

---

## 🎯 Ready to Start?

- [ ] All packages installed
- [ ] GPU accessible and tested
- [ ] Datasets pre-downloaded
- [ ] Understand message passing conceptually (ELI5.md)
- [ ] Clear on edge_index format

**Start with Lab 1: PyTorch Geometric Setup!**
