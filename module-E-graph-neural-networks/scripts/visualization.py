"""
Graph Visualization Utilities

This module provides utilities for visualizing graphs, embeddings,
attention weights, and training progress.

Author: Professor SPARK
Module: E - Graph Neural Networks

Example:
    >>> from visualization import plot_graph, plot_embeddings, plot_attention
    >>>
    >>> # Visualize a graph
    >>> plot_graph(data, title="My Graph")
    >>>
    >>> # Visualize embeddings with t-SNE
    >>> plot_embeddings(embeddings, labels)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple, Union
import warnings

try:
    import networkx as nx
    from torch_geometric.utils import to_networkx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not installed. Some visualization functions will not work.")

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed. Embedding visualization will not work.")


def plot_graph(
    data,
    node_colors: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Graph Visualization",
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 300,
    with_labels: bool = False,
    cmap: str = 'Set3',
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a PyG graph using NetworkX.

    Args:
        data: PyTorch Geometric Data object
        node_colors: Colors for each node (class labels or continuous values)
        title: Plot title
        figsize: Figure size (width, height)
        node_size: Size of nodes
        with_labels: Whether to show node indices
        cmap: Matplotlib colormap name
        save_path: If provided, save figure to this path

    Example:
        >>> from torch_geometric.datasets import Planetoid
        >>> data = Planetoid(root='/tmp/Cora', name='Cora')[0]
        >>> plot_graph(data, node_colors=data.y, title="Cora Citation Network")
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for graph visualization")

    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)

    # Handle node colors
    if node_colors is None:
        if hasattr(data, 'y'):
            node_colors = data.y.cpu().numpy() if torch.is_tensor(data.y) else data.y
        else:
            node_colors = 'lightblue'
    elif torch.is_tensor(node_colors):
        node_colors = node_colors.cpu().numpy()

    # Create figure
    plt.figure(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(G.number_of_nodes()))

    # Draw
    nx.draw(
        G, pos,
        node_color=node_colors,
        cmap=plt.cm.get_cmap(cmap),
        node_size=node_size,
        with_labels=with_labels,
        edge_color='lightgray',
        alpha=0.8,
        width=0.5
    )

    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()


def plot_subgraph(
    data,
    node_indices: List[int],
    node_colors: Optional[np.ndarray] = None,
    title: str = "Subgraph",
    figsize: Tuple[int, int] = (10, 8),
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a subgraph of the full graph.

    Args:
        data: PyTorch Geometric Data object
        node_indices: List of node indices to include
        node_colors: Colors for nodes (will be subsetted)
        title: Plot title
        figsize: Figure size
        class_names: Names for each class (for legend)
        save_path: If provided, save figure to this path

    Example:
        >>> plot_subgraph(data, list(range(100)), node_colors=data.y[:100])
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for graph visualization")

    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    subgraph = G.subgraph(node_indices)

    # Handle colors
    if node_colors is None and hasattr(data, 'y'):
        node_colors = data.y[node_indices].cpu().numpy()
    elif torch.is_tensor(node_colors):
        node_colors = node_colors.cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.spring_layout(subgraph, seed=42, k=2)

    # Draw
    cmap = plt.cm.Set3
    nx.draw(
        subgraph, pos, ax=ax,
        node_color=node_colors,
        cmap=cmap,
        node_size=200,
        with_labels=False,
        edge_color='lightgray',
        alpha=0.8,
        width=0.5
    )

    # Add legend if class names provided
    if class_names is not None and node_colors is not None:
        unique_classes = np.unique(node_colors)
        handles = [
            mpatches.Patch(color=cmap(c / max(unique_classes)), label=class_names[int(c)])
            for c in unique_classes
        ]
        ax.legend(handles=handles, loc='upper left', fontsize=10)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_embeddings(
    embeddings: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    title: str = "Node Embeddings (t-SNE)",
    figsize: Tuple[int, int] = (12, 10),
    perplexity: int = 30,
    cmap: str = 'tab10',
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize high-dimensional embeddings using t-SNE.

    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        labels: Optional class labels for coloring
        class_names: Names for each class (for legend)
        title: Plot title
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        cmap: Matplotlib colormap
        save_path: If provided, save figure to this path

    Returns:
        2D t-SNE embeddings

    Example:
        >>> embeddings = model.get_embeddings(data.x, data.edge_index)
        >>> plot_embeddings(embeddings, labels=data.y, class_names=['A', 'B', 'C'])
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for t-SNE visualization")

    # Convert to numpy
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    if labels is not None and torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    print(f"Running t-SNE on {embeddings.shape[0]} samples...")

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    plt.figure(figsize=figsize)

    # Scatter plot
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels if labels is not None else 'steelblue',
        cmap=cmap,
        alpha=0.7,
        s=20
    )

    # Add legend
    if labels is not None and class_names is not None:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, loc='upper right', title='Class')

    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return embeddings_2d


def plot_attention(
    data,
    attention_weights: torch.Tensor,
    edge_index: torch.Tensor,
    node_id: int,
    title: str = "Attention Visualization",
    figsize: Tuple[int, int] = (10, 8),
    top_k: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights for a specific node.

    Args:
        data: PyTorch Geometric Data object
        attention_weights: Attention weights for edges
        edge_index: Edge indices [2, num_edges]
        node_id: The target node to visualize
        title: Plot title
        figsize: Figure size
        top_k: Number of top neighbors to show
        save_path: If provided, save figure to this path

    Example:
        >>> out, (edge_idx, attn) = model(x, edge_index, return_attention=True)
        >>> plot_attention(data, attn, edge_idx, node_id=0)
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for attention visualization")

    # Convert to numpy
    edge_index_np = edge_index.cpu().numpy()
    attn_np = attention_weights.cpu().numpy()

    # Find edges where node_id is the destination
    mask = edge_index_np[1] == node_id
    neighbors = edge_index_np[0][mask]
    neighbor_attentions = attn_np[mask]

    # Sort by attention
    sorted_idx = np.argsort(neighbor_attentions)[::-1]

    # Print top-k
    print(f"\nTop-{top_k} attended neighbors for node {node_id}:")
    print("-" * 40)
    labels = data.y.cpu().numpy() if hasattr(data, 'y') else None
    for i, idx in enumerate(sorted_idx[:top_k]):
        neighbor = neighbors[idx]
        attn = neighbor_attentions[idx]
        label_str = f" (class {labels[neighbor]})" if labels is not None else ""
        print(f"  {i+1}. Node {neighbor}: α={attn:.4f}{label_str}")

    # Create subgraph for visualization
    all_nodes = [node_id] + list(neighbors[:top_k])
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)

    for i, (neighbor, attn) in enumerate(zip(neighbors, neighbor_attentions)):
        if neighbor in all_nodes:
            G.add_edge(neighbor, node_id, weight=attn)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node colors (target node highlighted)
    node_colors = ['red' if n == node_id else 'lightblue' for n in G.nodes()]
    node_sizes = [800 if n == node_id else 400 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    # Edge widths proportional to attention
    edges = G.edges()
    weights = [G[u][v]['weight'] * 5 for u, v in edges]

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=weights,
                          alpha=0.7, edge_color='gray', arrows=True, arrowsize=15)

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

    ax.set_title(f"{title}\n(Node {node_id}, edge width = attention)", fontsize=12)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_training_curves(
    history: dict,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dictionary with 'loss', 'train_acc', 'val_acc', 'test_acc' keys
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Example:
        >>> history = {'loss': [...], 'train_acc': [...], 'val_acc': [...], 'test_acc': [...]}
        >>> plot_training_curves(history)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss curve
    if 'loss' in history:
        axes[0].plot(history['loss'], color='steelblue', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Validation', linewidth=2)
    if 'test_acc' in history:
        axes[1].plot(history['test_acc'], label='Test', linewidth=2)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_degree_distribution(
    data,
    title: str = "Degree Distribution",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot the degree distribution of a graph.

    Args:
        data: PyTorch Geometric Data object
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Example:
        >>> plot_degree_distribution(data)
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for degree distribution")

    G = to_networkx(data, to_undirected=True)
    degrees = [d for n, d in G.degree()]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Linear scale
    axes[0].hist(degrees, bins=50, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Degree Distribution')

    # Log scale
    axes[1].hist(degrees, bins=50, color='steelblue', edgecolor='white', log=True)
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Count (log scale)')
    axes[1].set_title('Degree Distribution (Log Scale)')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    # Print statistics
    print(f"\nDegree Statistics:")
    print(f"  Min: {min(degrees)}")
    print(f"  Max: {max(degrees)}")
    print(f"  Mean: {np.mean(degrees):.2f}")
    print(f"  Median: {np.median(degrees):.1f}")


def plot_pooling_comparison(
    results: dict,
    title: str = "Pooling Strategy Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of different pooling strategies.

    Args:
        results: Dictionary mapping pooling name to (mean_acc, std_acc)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Example:
        >>> results = {'mean': (0.85, 0.02), 'max': (0.82, 0.03)}
        >>> plot_pooling_comparison(results)
    """
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]

    colors = ['steelblue', 'coral', 'seagreen', 'gold', 'orchid']

    plt.figure(figsize=figsize)
    bars = plt.bar(names, means, yerr=stds, capsize=5,
                   color=colors[:len(names)], edgecolor='black')

    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.ylim(0.5, 1.0)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                 f'{mean:.3f}', ha='center', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_graph(data, node_colors=None)")
    print("  - plot_subgraph(data, node_indices)")
    print("  - plot_embeddings(embeddings, labels=None)")
    print("  - plot_attention(data, attention_weights, edge_index, node_id)")
    print("  - plot_training_curves(history)")
    print("  - plot_degree_distribution(data)")
    print("  - plot_pooling_comparison(results)")
