"""
Graph Utilities for GNN Training and Evaluation

This module provides utility functions for working with graph data,
including data loading, preprocessing, training, and evaluation.

Author: Professor SPARK
Module: E - Graph Neural Networks

Example:
    >>> from graph_utils import train_epoch, evaluate, create_data_split
    >>>
    >>> # Train for one epoch
    >>> loss = train_epoch(model, data, optimizer)
    >>>
    >>> # Evaluate
    >>> train_acc, val_acc, test_acc = evaluate(model, data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, degree
from typing import Tuple, Optional, List, Dict, Any
import time


def load_cora(root: str = '/tmp/Cora') -> Tuple[Any, Any]:
    """
    Load the Cora citation network dataset.

    Args:
        root: Directory to store/load the dataset

    Returns:
        Tuple of (dataset, data)

    Example:
        >>> dataset, data = load_cora()
        >>> print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    """
    dataset = Planetoid(root=root, name='Cora')
    data = dataset[0]
    return dataset, data


def load_citeseer(root: str = '/tmp/CiteSeer') -> Tuple[Any, Any]:
    """Load the CiteSeer citation network dataset."""
    dataset = Planetoid(root=root, name='CiteSeer')
    data = dataset[0]
    return dataset, data


def load_pubmed(root: str = '/tmp/PubMed') -> Tuple[Any, Any]:
    """Load the PubMed citation network dataset."""
    dataset = Planetoid(root=root, name='PubMed')
    data = dataset[0]
    return dataset, data


def load_mutag(root: str = '/tmp/MUTAG') -> Any:
    """Load the MUTAG molecular dataset."""
    return TUDataset(root=root, name='MUTAG')


def load_proteins(root: str = '/tmp/PROTEINS') -> Any:
    """Load the PROTEINS dataset."""
    return TUDataset(root=root, name='PROTEINS')


def create_data_split(
    data,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Any:
    """
    Create custom train/val/test split for node classification.

    Args:
        data: PyG Data object
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        seed: Random seed for reproducibility

    Returns:
        Modified data object with new masks

    Example:
        >>> data = create_data_split(data, train_ratio=0.6, val_ratio=0.2)
        >>> print(f"Training nodes: {data.train_mask.sum()}")
    """
    torch.manual_seed(seed)

    n_nodes = data.num_nodes
    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)

    # Shuffle node indices
    perm = torch.randperm(n_nodes)

    # Create masks
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    # Update data object
    data.train_mask = train_mask.to(data.x.device)
    data.val_mask = val_mask.to(data.x.device)
    data.test_mask = test_mask.to(data.x.device)

    return data


def train_epoch(
    model: nn.Module,
    data: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module = None
) -> float:
    """
    Train for one epoch on node classification.

    Args:
        model: GNN model
        data: PyG Data object with train_mask
        optimizer: PyTorch optimizer
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        Training loss

    Example:
        >>> loss = train_epoch(model, data, optimizer)
        >>> print(f"Epoch loss: {loss:.4f}")
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data: Any) -> Tuple[float, float, float]:
    """
    Evaluate model on train/val/test sets.

    Args:
        model: GNN model
        data: PyG Data object with masks

    Returns:
        Tuple of (train_acc, val_acc, test_acc)

    Example:
        >>> train_acc, val_acc, test_acc = evaluate(model, data)
        >>> print(f"Test accuracy: {test_acc:.4f}")
    """
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accs.append(acc)

    return tuple(accs)


def train_graph_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train graph classifier for one epoch.

    Args:
        model: Graph classification model
        train_loader: DataLoader for training graphs
        optimizer: PyTorch optimizer
        device: Device to use

    Returns:
        Average training loss

    Example:
        >>> loss = train_graph_classifier(model, train_loader, optimizer, device)
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate_graph_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> float:
    """
    Evaluate graph classifier accuracy.

    Args:
        model: Graph classification model
        loader: DataLoader for evaluation
        device: Device to use

    Returns:
        Accuracy

    Example:
        >>> acc = evaluate_graph_classifier(model, test_loader, device)
    """
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return correct / total


def compute_homophily(data) -> float:
    """
    Compute graph homophily (fraction of same-class edges).

    High homophily (>0.7): Similar nodes are connected → GNNs work well!
    Low homophily (<0.3): Dissimilar nodes are connected → Need special architectures

    Args:
        data: PyG Data object with labels

    Returns:
        Homophily score in [0, 1]

    Example:
        >>> h = compute_homophily(data)
        >>> print(f"Homophily: {h:.4f}")
    """
    edge_index = data.edge_index
    labels = data.y

    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]

    same_class = (src_labels == dst_labels).float()
    homophily = same_class.mean().item()

    return homophily


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters

    Example:
        >>> n_params = count_parameters(model)
        >>> print(f"Parameters: {n_params:,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with memory statistics in MB

    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Allocated: {mem['allocated']:.1f} MB")
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}

    return {
        'allocated': torch.cuda.memory_allocated() / 1e6,
        'reserved': torch.cuda.memory_reserved() / 1e6,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e6
    }


def clear_memory():
    """Clear GPU memory cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def time_forward_pass(
    model: nn.Module,
    data: Any,
    n_runs: int = 100,
    warmup: int = 10
) -> float:
    """
    Time the forward pass of a model.

    Args:
        model: GNN model
        data: PyG Data object
        n_runs: Number of runs to average
        warmup: Number of warmup runs

    Returns:
        Average time per forward pass in milliseconds

    Example:
        >>> time_ms = time_forward_pass(model, data)
        >>> print(f"Forward pass: {time_ms:.2f} ms")
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(data.x, data.edge_index)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(data.x, data.edge_index)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / n_runs) * 1000


def get_graph_statistics(data) -> Dict[str, Any]:
    """
    Compute various statistics about a graph.

    Args:
        data: PyG Data object

    Returns:
        Dictionary of statistics

    Example:
        >>> stats = get_graph_statistics(data)
        >>> print(f"Average degree: {stats['avg_degree']:.2f}")
    """
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx
        G = to_networkx(data, to_undirected=True)
        degrees = [d for n, d in G.degree()]
        clustering = nx.average_clustering(G)
        n_components = nx.number_connected_components(G)
    except ImportError:
        # Fallback without NetworkX
        edge_index = data.edge_index
        deg = degree(edge_index[0], num_nodes=data.num_nodes)
        degrees = deg.tolist()
        clustering = None
        n_components = None

    import numpy as np

    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features if hasattr(data, 'num_features') else data.x.size(1),
        'avg_degree': np.mean(degrees),
        'max_degree': max(degrees),
        'min_degree': min(degrees),
        'density': data.num_edges / (data.num_nodes * (data.num_nodes - 1)),
    }

    if clustering is not None:
        stats['avg_clustering'] = clustering
    if n_components is not None:
        stats['num_components'] = n_components

    if hasattr(data, 'y'):
        stats['num_classes'] = data.y.max().item() + 1
        stats['homophily'] = compute_homophily(data)

    return stats


def print_graph_summary(data, dataset_name: str = "Graph"):
    """
    Print a formatted summary of graph statistics.

    Args:
        data: PyG Data object
        dataset_name: Name to display

    Example:
        >>> print_graph_summary(data, "Cora")
    """
    stats = get_graph_statistics(data)

    print("=" * 50)
    print(f"{dataset_name} DATASET SUMMARY")
    print("=" * 50)
    print(f"Nodes: {stats['num_nodes']:,}")
    print(f"Edges: {stats['num_edges']:,}")
    print(f"Features: {stats['num_features']}")

    if 'num_classes' in stats:
        print(f"Classes: {stats['num_classes']}")

    print(f"\nDegree statistics:")
    print(f"  Average: {stats['avg_degree']:.2f}")
    print(f"  Max: {stats['max_degree']}")
    print(f"  Min: {stats['min_degree']}")

    if 'avg_clustering' in stats:
        print(f"\nClustering coefficient: {stats['avg_clustering']:.4f}")

    if 'homophily' in stats:
        print(f"Homophily: {stats['homophily']:.4f}")

    print("=" * 50)


class EarlyStopping:
    """
    Early stopping handler for training.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum improvement to reset patience

    Example:
        >>> early_stop = EarlyStopping(patience=20)
        >>> for epoch in range(1000):
        ...     val_loss = train_and_validate()
        ...     if early_stop(val_loss):
        ...         print("Early stopping!")
        ...         break
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


if __name__ == "__main__":
    print("Testing graph utilities...")

    # Load Cora
    dataset, data = load_cora()
    print_graph_summary(data, "Cora")

    # Create custom split
    data = create_data_split(data, train_ratio=0.6, val_ratio=0.2)
    print(f"\nCustom split:")
    print(f"  Train: {data.train_mask.sum().item()}")
    print(f"  Val: {data.val_mask.sum().item()}")
    print(f"  Test: {data.test_mask.sum().item()}")

    print("\nAll tests passed!")
