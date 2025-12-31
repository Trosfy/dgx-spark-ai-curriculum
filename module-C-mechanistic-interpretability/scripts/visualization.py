"""
Visualization Utilities for Mechanistic Interpretability
========================================================

High-quality visualization functions for interpretability research.

This module provides:
- Attention pattern heatmaps
- Activation patching visualizations
- Residual stream analysis plots
- Interactive circuit diagrams

Author: Professor SPARK
Hardware: Optimized for DGX Spark (128GB unified memory)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Dict, Any, Tuple, Union
import warnings

# Try to import optional dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Some visualizations will use matplotlib instead.")


# Color schemes for interpretability visualizations
ATTENTION_COLORSCALE = "Blues"
PATCHING_COLORSCALE = "RdBu_r"  # Red = negative effect, Blue = positive
RESIDUAL_COLORSCALE = "Viridis"


def plot_attention_pattern(
    attention_pattern: Union[torch.Tensor, np.ndarray],
    tokens: Optional[List[str]] = None,
    title: str = "Attention Pattern",
    figsize: Tuple[int, int] = (10, 8),
    use_plotly: bool = True,
    return_fig: bool = False
) -> Optional[Any]:
    """Visualize attention pattern as a heatmap.

    Args:
        attention_pattern: Attention weights [seq_len, seq_len] or [batch, heads, seq, seq]
        tokens: Optional list of token strings for labels
        title: Plot title
        figsize: Figure size for matplotlib
        use_plotly: Whether to use plotly (interactive) or matplotlib
        return_fig: Whether to return the figure object

    Returns:
        Figure object if return_fig=True

    Example:
        >>> pattern = cache["pattern", 5][0, 3]  # Layer 5, Head 3
        >>> tokens = model.to_str_tokens(input_tokens)
        >>> plot_attention_pattern(pattern, tokens, "Layer 5, Head 3")
    """
    if isinstance(attention_pattern, torch.Tensor):
        attention_pattern = attention_pattern.detach().cpu().numpy()

    # Handle different input shapes
    if len(attention_pattern.shape) == 4:
        attention_pattern = attention_pattern[0, 0]  # Take first batch, first head
    elif len(attention_pattern.shape) == 3:
        attention_pattern = attention_pattern[0]  # Take first batch/head

    seq_len = attention_pattern.shape[0]

    if tokens is None:
        tokens = [str(i) for i in range(seq_len)]
    elif len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = tokens + [f"[{i}]" for i in range(len(tokens), seq_len)]

    if use_plotly and PLOTLY_AVAILABLE:
        fig = px.imshow(
            attention_pattern,
            labels=dict(x="Key (Attended To)", y="Query (Attending From)", color="Weight"),
            x=tokens,
            y=tokens,
            color_continuous_scale=ATTENTION_COLORSCALE,
            title=title
        )
        fig.update_layout(
            width=figsize[0] * 80,
            height=figsize[1] * 80,
            xaxis_tickangle=45
        )
        if return_fig:
            return fig
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attention_pattern, cmap='Blues', aspect='auto')

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)

        ax.set_xlabel("Key (Attended To)")
        ax.set_ylabel("Query (Attending From)")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()


def plot_attention_heads_grid(
    cache: Any,
    layer: int,
    tokens: List[str],
    heads_per_row: int = 4,
    figsize: Tuple[int, int] = (16, 12),
    title: Optional[str] = None
) -> Optional[Any]:
    """Plot all attention heads in a layer as a grid.

    Args:
        cache: Activation cache from model.run_with_cache
        layer: Layer index to visualize
        tokens: List of token strings
        heads_per_row: Number of heads per row
        figsize: Figure size
        title: Optional title

    Returns:
        Figure object

    Example:
        >>> _, cache = model.run_with_cache(tokens)
        >>> plot_attention_heads_grid(cache, layer=5, tokens=token_strs)
    """
    pattern = cache["pattern", layer][0].detach().cpu().numpy()  # [heads, seq, seq]
    n_heads = pattern.shape[0]
    n_rows = (n_heads + heads_per_row - 1) // heads_per_row

    fig, axes = plt.subplots(n_rows, heads_per_row, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes] if heads_per_row == 1 else axes

    for head in range(n_heads):
        ax = axes[head]
        im = ax.imshow(pattern[head], cmap='Blues', aspect='auto')
        ax.set_title(f"Head {head}", fontsize=10)

        if len(tokens) <= 10:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide empty subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title or f"Layer {layer} Attention Heads", fontsize=14)
    plt.tight_layout()

    return fig


def plot_patching_heatmap(
    effects: np.ndarray,
    title: str = "Activation Patching Effects",
    xlabel: str = "Head",
    ylabel: str = "Layer",
    figsize: Tuple[int, int] = (12, 8),
    use_plotly: bool = True,
    return_fig: bool = False
) -> Optional[Any]:
    """Plot activation patching results as a heatmap.

    Args:
        effects: Array of patching effects [layers, heads] or [layers, components]
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        use_plotly: Whether to use plotly
        return_fig: Whether to return figure

    Returns:
        Figure if return_fig=True

    Example:
        >>> effects = run_head_patching(model, clean, corrupted, answer)
        >>> plot_patching_heatmap(effects, "IOI Head Patching")
    """
    if use_plotly and PLOTLY_AVAILABLE:
        fig = px.imshow(
            effects,
            labels=dict(x=xlabel, y=ylabel, color="Effect"),
            color_continuous_scale=PATCHING_COLORSCALE,
            color_continuous_midpoint=0,
            title=title
        )
        fig.update_layout(width=figsize[0] * 80, height=figsize[1] * 80)
        if return_fig:
            return fig
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=figsize)

        # Use diverging colormap centered at 0
        vmax = max(abs(effects.min()), abs(effects.max()))
        im = ax.imshow(effects, cmap='RdBu_r', aspect='auto',
                       vmin=-vmax, vmax=vmax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Effect (0=none, 1=full)")
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()


def plot_residual_stream_norms(
    cache: Any,
    positions: Optional[List[int]] = None,
    token_labels: Optional[List[str]] = None,
    title: str = "Residual Stream Norms Across Layers",
    figsize: Tuple[int, int] = (12, 6)
) -> Any:
    """Plot how residual stream norms change across layers.

    Args:
        cache: Activation cache
        positions: Which sequence positions to plot (default: last 5)
        token_labels: Labels for positions
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object

    Example:
        >>> _, cache = model.run_with_cache(tokens)
        >>> plot_residual_stream_norms(cache, positions=[-3, -2, -1])
    """
    # Determine number of layers
    n_layers = 0
    while True:
        try:
            _ = cache["resid_post", n_layers]
            n_layers += 1
        except KeyError:
            break

    seq_len = cache["resid_post", 0].shape[1]

    if positions is None:
        positions = list(range(max(0, seq_len - 5), seq_len))

    if token_labels is None:
        token_labels = [f"pos {p}" for p in positions]

    fig, ax = plt.subplots(figsize=figsize)

    for pos, label in zip(positions, token_labels):
        norms = []
        for layer in range(n_layers):
            resid = cache["resid_post", layer][0, pos, :]
            norms.append(resid.norm().item())
        ax.plot(range(n_layers), norms, marker='o', label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_logit_lens(
    model: Any,
    cache: Any,
    position: int = -1,
    top_k: int = 10,
    figsize: Tuple[int, int] = (14, 8)
) -> Any:
    """Create logit lens visualization showing predictions at each layer.

    The "logit lens" passes intermediate residual streams through
    the unembedding matrix to see what the model would predict
    at each layer.

    Args:
        model: HookedTransformer model
        cache: Activation cache
        position: Sequence position to analyze
        top_k: Number of top tokens to show
        figsize: Figure size

    Returns:
        Figure object

    Example:
        >>> _, cache = model.run_with_cache(tokens)
        >>> plot_logit_lens(model, cache, position=-1, top_k=5)
    """
    n_layers = model.cfg.n_layers

    # Get predictions at each layer
    layer_preds = []
    layer_probs = []

    W_U = model.W_U  # Unembedding matrix

    for layer in range(n_layers + 1):
        if layer == 0:
            resid = cache["resid_pre", 0][0, position, :]
        else:
            resid = cache["resid_post", layer - 1][0, position, :]

        # Apply final layer norm if it exists
        if hasattr(model, 'ln_final'):
            resid = model.ln_final(resid)

        # Get logits
        logits = resid @ W_U

        # Get top predictions
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)

        tokens = [model.tokenizer.decode(idx.item()) for idx in top_indices]
        layer_preds.append(tokens)
        layer_probs.append(top_probs.detach().cpu().numpy())

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap data
    prob_matrix = np.zeros((n_layers + 1, top_k))
    for layer in range(n_layers + 1):
        prob_matrix[layer, :] = layer_probs[layer]

    im = ax.imshow(prob_matrix.T, aspect='auto', cmap='YlOrRd')

    # Labels
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-k Rank")
    ax.set_title("Logit Lens: Top Predictions at Each Layer")

    # Add token labels
    for layer in range(n_layers + 1):
        for rank in range(min(5, top_k)):  # Show top 5 labels
            token = layer_preds[layer][rank]
            prob = layer_probs[layer][rank]
            if len(token) > 8:
                token = token[:6] + ".."
            ax.text(layer, rank, f"{token}\n{prob:.1%}",
                   ha='center', va='center', fontsize=7)

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()

    return fig


def plot_component_attributions(
    results: List[Any],
    component_filter: Optional[str] = None,
    title: str = "Component Attributions",
    figsize: Tuple[int, int] = (14, 6)
) -> Any:
    """Plot attribution results from activation patching.

    Args:
        results: List of PatchingResult objects
        component_filter: Filter to specific component type
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object

    Example:
        >>> results = run_activation_patching(model, clean, corrupted, answer)
        >>> plot_component_attributions(results, component_filter="resid_post")
    """
    if component_filter:
        results = [r for r in results if r.component == component_filter]

    layers = [r.layer for r in results]
    effects = [r.effect for r in results]
    components = [r.component for r in results]

    fig, ax = plt.subplots(figsize=figsize)

    # Group by component type
    unique_components = list(set(components))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_components)))

    for comp, color in zip(unique_components, colors):
        comp_results = [(r.layer, r.effect) for r in results if r.component == comp]
        comp_layers, comp_effects = zip(*comp_results)
        ax.bar([l + unique_components.index(comp) * 0.25 for l in comp_layers],
               comp_effects, width=0.25, label=comp, color=color, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Patching Effect")
    ax.set_title(title)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_induction_scores(
    scores: np.ndarray,
    threshold: float = 0.1,
    title: str = "Induction Head Scores",
    figsize: Tuple[int, int] = (12, 8)
) -> Any:
    """Plot induction head scores across the model.

    Args:
        scores: Array of shape [n_layers, n_heads] with induction scores
        threshold: Highlight heads above this threshold
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object

    Example:
        >>> scores = compute_induction_scores(model)
        >>> plot_induction_scores(scores, threshold=0.2)
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(scores, cmap='YlOrRd', aspect='auto')

    # Mark heads above threshold
    above_threshold = np.argwhere(scores > threshold)
    for layer, head in above_threshold:
        ax.add_patch(plt.Rectangle((head - 0.5, layer - 0.5), 1, 1,
                                    fill=False, edgecolor='blue', linewidth=2))

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title} (boxes = score > {threshold})")

    plt.colorbar(im, ax=ax, label="Induction Score")
    plt.tight_layout()

    return fig


def create_circuit_diagram(
    components: List[Dict[str, Any]],
    connections: List[Tuple[str, str, float]],
    title: str = "Circuit Diagram",
    figsize: Tuple[int, int] = (14, 10)
) -> Any:
    """Create a circuit diagram showing information flow.

    Args:
        components: List of dicts with 'name', 'layer', 'type', 'importance'
        connections: List of (source, target, strength) tuples
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object

    Example:
        >>> components = [
        ...     {"name": "L0H3", "layer": 0, "type": "attention", "importance": 0.8},
        ...     {"name": "L5H9", "layer": 5, "type": "attention", "importance": 0.9},
        ... ]
        >>> connections = [("L0H3", "L5H9", 0.7)]
        >>> create_circuit_diagram(components, connections)
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx required for circuit diagrams. Install with: pip install networkx")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    G = nx.DiGraph()

    # Add nodes
    for comp in components:
        G.add_node(comp['name'],
                  layer=comp['layer'],
                  node_type=comp['type'],
                  importance=comp['importance'])

    # Add edges
    for source, target, strength in connections:
        G.add_edge(source, target, weight=strength)

    # Position nodes by layer
    pos = {}
    layers = {}
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)

    for layer, nodes in layers.items():
        for i, node in enumerate(nodes):
            pos[node] = (layer, i - len(nodes) / 2)

    # Draw
    node_colors = ['lightblue' if G.nodes[n]['node_type'] == 'attention' else 'lightgreen'
                   for n in G.nodes()]
    node_sizes = [G.nodes[n]['importance'] * 2000 for n in G.nodes()]

    edge_weights = [G.edges[e]['weight'] * 3 for e in G.edges()]

    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            node_size=node_sizes, font_size=8, font_weight='bold',
            edge_color='gray', width=edge_weights, arrows=True,
            arrowsize=20, connectionstyle="arc3,rad=0.1")

    ax.set_title(title)
    plt.tight_layout()

    return fig


def visualize_token_importance(
    tokens: List[str],
    importance_scores: np.ndarray,
    title: str = "Token Importance",
    figsize: Tuple[int, int] = (14, 3)
) -> Any:
    """Visualize importance of each token as a colored bar.

    Args:
        tokens: List of token strings
        importance_scores: Array of importance values
        title: Plot title
        figsize: Figure size

    Returns:
        Figure object

    Example:
        >>> tokens = model.to_str_tokens(input)
        >>> importance = compute_token_importance(model, input)
        >>> visualize_token_importance(tokens, importance)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize scores
    scores = (importance_scores - importance_scores.min()) / \
             (importance_scores.max() - importance_scores.min() + 1e-10)

    # Create colored boxes for each token
    cmap = plt.cm.YlOrRd
    for i, (token, score) in enumerate(zip(tokens, scores)):
        color = cmap(score)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black'))
        ax.text(i + 0.5, 0.5, token, ha='center', va='center',
               fontsize=8, rotation=45)

    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='horizontal', label='Importance', pad=0.2)

    plt.tight_layout()
    return fig


def save_figure(fig: Any, filename: str, dpi: int = 150) -> None:
    """Save figure to file.

    Args:
        fig: Figure object (matplotlib or plotly)
        filename: Output filename
        dpi: Resolution for raster formats

    Example:
        >>> fig = plot_attention_pattern(pattern, tokens, return_fig=True)
        >>> save_figure(fig, "attention.png")
    """
    if hasattr(fig, 'write_image'):
        # Plotly figure
        fig.write_image(filename)
    else:
        # Matplotlib figure
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filename}")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
    print(f"Plotly available: {PLOTLY_AVAILABLE}")
