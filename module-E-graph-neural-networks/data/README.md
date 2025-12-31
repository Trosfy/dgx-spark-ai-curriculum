# Data Directory

This directory contains data files and documentation for Module E: Graph Neural Networks.

## Datasets Used

This module uses publicly available datasets that are automatically downloaded by PyTorch Geometric. No manual data download is required.

### Citation Networks (Node Classification)

| Dataset | Nodes | Edges | Features | Classes | Task |
|---------|-------|-------|----------|---------|------|
| **Cora** | 2,708 | 10,556 | 1,433 | 7 | Paper topic classification |
| **CiteSeer** | 3,327 | 9,104 | 3,703 | 6 | Paper topic classification |
| **PubMed** | 19,717 | 88,648 | 500 | 3 | Paper topic classification |

**Loading:**
```python
from torch_geometric.datasets import Planetoid

cora = Planetoid(root='/tmp/Cora', name='Cora')
citeseer = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
pubmed = Planetoid(root='/tmp/PubMed', name='PubMed')
```

### Molecular Datasets (Graph Classification)

| Dataset | Graphs | Avg Nodes | Classes | Task |
|---------|--------|-----------|---------|------|
| **MUTAG** | 188 | 17.9 | 2 | Mutagenicity prediction |
| **PROTEINS** | 1,113 | 39.1 | 2 | Enzyme vs Non-enzyme |
| **NCI1** | 4,110 | 29.8 | 2 | Anti-cancer activity |

**Loading:**
```python
from torch_geometric.datasets import TUDataset

mutag = TUDataset(root='/tmp/MUTAG', name='MUTAG')
proteins = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
nci1 = TUDataset(root='/tmp/NCI1', name='NCI1')
```

## Data Format

### PyG Data Object Structure

```python
Data(
    x=[num_nodes, num_features],      # Node features
    edge_index=[2, num_edges],        # Edge list (source, target)
    y=[num_nodes] or [1],             # Labels (node or graph level)
    train_mask=[num_nodes],           # Boolean mask for training
    val_mask=[num_nodes],             # Boolean mask for validation
    test_mask=[num_nodes],            # Boolean mask for testing
)
```

### Edge Index Format

Edges are stored as a `[2, num_edges]` tensor:
- Row 0: Source node indices
- Row 1: Target node indices

For undirected graphs, each edge appears twice (both directions).

```python
# Example: edges A→B and B→C
edge_index = torch.tensor([
    [0, 1, 1, 2],  # sources: A, B, B, C
    [1, 0, 2, 1],  # targets: B, A, C, B
])
```

## Memory Considerations

| Dataset | Approximate Memory |
|---------|-------------------|
| Cora | ~50 MB |
| CiteSeer | ~60 MB |
| PubMed | ~200 MB |
| MUTAG | ~5 MB |
| PROTEINS | ~20 MB |

All datasets fit easily in DGX Spark's 128GB unified memory!

## Custom Data

To use your own graph data:

```python
from torch_geometric.data import Data

# Create a single graph
data = Data(
    x=torch.randn(100, 16),           # 100 nodes, 16 features
    edge_index=torch.randint(0, 100, (2, 500)),  # 500 edges
    y=torch.randint(0, 3, (100,)),    # 3-class node labels
)

# For graph classification, create a list
graphs = [Data(...) for _ in range(1000)]
```

## References

- **Cora/CiteSeer/PubMed**: Sen et al., "Collective Classification in Network Data" (2008)
- **MUTAG**: Debnath et al., "Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds" (1991)
- **PROTEINS**: Borgwardt et al., "Protein function prediction via graph kernels" (2005)
- **TUDataset collection**: https://chrsmrrs.github.io/datasets/
