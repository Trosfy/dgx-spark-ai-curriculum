"""
Graph Neural Network Layers - Reusable Implementations

This module provides production-quality implementations of common GNN layers
for use in the DGX Spark AI curriculum.

Author: Professor SPARK
Module: E - Graph Neural Networks

Example:
    >>> from gnn_layers import GCNLayer, GATLayer, GraphClassifier
    >>>
    >>> # Create a GCN layer
    >>> layer = GCNLayer(in_channels=1433, out_channels=64)
    >>> out = layer(x, edge_index)
    >>>
    >>> # Create a full classifier
    >>> model = GraphClassifier(num_features=7, hidden_dim=64, num_classes=2)
    >>> logits = model(x, edge_index, batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Optional, Tuple, List


class GCNLayer(nn.Module):
    """
    Graph Convolutional Layer.

    Implements the GCN propagation rule:
        H' = σ(D^(-1/2) Ã D^(-1/2) H W + b)

    Where:
        - Ã = A + I (adjacency with self-loops)
        - D = degree matrix of Ã
        - H = node features
        - W = learnable weight matrix
        - b = learnable bias

    Args:
        in_channels: Number of input features per node
        out_channels: Number of output features per node
        bias: If True, add learnable bias (default: True)

    Example:
        >>> layer = GCNLayer(1433, 64)
        >>> x = torch.randn(2708, 1433)
        >>> edge_index = torch.randint(0, 2708, (2, 10556))
        >>> out = layer(x, edge_index)
        >>> out.shape
        torch.Size([2708, 64])
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        num_nodes = x.size(0)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Compute normalization
        row, col = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Linear transformation
        x = x @ self.weight

        # Message passing with scatter_add
        out = torch.zeros_like(x)
        src_features = x[row] * norm.view(-1, 1)
        out.scatter_add_(0, col.view(-1, 1).expand_as(src_features), src_features)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'


class GATLayer(nn.Module):
    """
    Graph Attention Layer.

    Implements attention-based aggregation:
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        α_ij = softmax_j(e_ij)
        h'_i = Σ_j α_ij Wh_j

    Args:
        in_channels: Number of input features per node
        out_channels: Number of output features per node
        heads: Number of attention heads (default: 1)
        concat: If True, concatenate heads; if False, average (default: True)
        dropout: Dropout probability for attention weights (default: 0.6)
        negative_slope: LeakyReLU negative slope (default: 0.2)

    Example:
        >>> layer = GATLayer(1433, 8, heads=8)
        >>> out = layer(x, edge_index)
        >>> out.shape  # 8 heads * 8 dims = 64
        torch.Size([2708, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.6,
        negative_slope: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear transformation for each head
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.a_left = nn.Parameter(torch.Tensor(heads, out_channels))
        self.a_right = nn.Parameter(torch.Tensor(heads, out_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            return_attention: If True, also return attention weights

        Returns:
            Updated features [num_nodes, heads * out_channels] or [num_nodes, out_channels]
            (Optional) Attention weights [num_edges, heads]
        """
        num_nodes = x.size(0)
        H, C = self.heads, self.out_channels

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Linear transformation: [N, in] -> [N, H, C]
        Wh = self.W(x).view(num_nodes, H, C)

        # Compute attention scores
        src, dst = edge_index

        # e_ij = a_left^T Wh_i + a_right^T Wh_j
        e_left = (Wh * self.a_left).sum(dim=-1)  # [N, H]
        e_right = (Wh * self.a_right).sum(dim=-1)  # [N, H]

        e = e_left[src] + e_right[dst]  # [E, H]
        e = self.leaky_relu(e)

        # Softmax over neighbors
        alpha = softmax(e, dst, num_nodes=num_nodes)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted aggregation
        out = torch.zeros(num_nodes, H, C, device=x.device, dtype=x.dtype)
        src_features = Wh[src] * alpha.unsqueeze(-1)  # [E, H, C]

        # Scatter add for each head
        idx = dst.view(-1, 1, 1).expand(-1, H, C)
        out.scatter_add_(0, idx, src_features)

        # Combine heads
        if self.concat:
            out = out.view(num_nodes, H * C)
        else:
            out = out.mean(dim=1)

        if return_attention:
            return out, (edge_index, alpha)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network for node classification.

    Args:
        num_features: Number of input node features
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension (default: 64)
        num_layers: Number of GCN layers (default: 2)
        dropout: Dropout probability (default: 0.5)

    Example:
        >>> model = GCN(num_features=1433, num_classes=7)
        >>> out = model(x, edge_index)
        >>> out.shape
        torch.Size([2708, 7])
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.dropout = dropout

        # Build layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
        self.convs.append(GCNLayer(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before final classification layer."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class GAT(nn.Module):
    """
    Graph Attention Network for node classification.

    Args:
        num_features: Number of input node features
        num_classes: Number of output classes
        hidden_dim: Hidden dimension per head (default: 8)
        heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.6)

    Example:
        >>> model = GAT(num_features=1433, num_classes=7)
        >>> out = model(x, edge_index)
        >>> out.shape
        torch.Size([2708, 7])
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 8,
        heads: int = 8,
        dropout: float = 0.6
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATLayer(num_features, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATLayer(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Forward pass returning class logits."""
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attn1 = self.conv1(x, edge_index, return_attention=True)
        else:
            x = self.conv1(x, edge_index)

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attn2 = self.conv2(x, edge_index, return_attention=True)
            return x, (attn1, attn2)

        x = self.conv2(x, edge_index)
        return x


class GraphClassifier(nn.Module):
    """
    Graph Neural Network for graph-level classification.

    Architecture: GCN layers -> Pooling -> MLP classifier

    Args:
        num_features: Number of input node features
        hidden_dim: Hidden layer dimension (default: 64)
        num_classes: Number of output classes
        num_layers: Number of GCN layers (default: 3)
        pooling: Pooling method - 'mean', 'max', 'sum', 'mean_max' (default: 'mean_max')
        dropout: Dropout probability (default: 0.5)

    Example:
        >>> model = GraphClassifier(num_features=7, hidden_dim=64, num_classes=2)
        >>> batch = next(iter(loader))
        >>> out = model(batch.x, batch.edge_index, batch.batch)
        >>> out.shape
        torch.Size([32, 2])  # batch_size=32
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        pooling: str = 'mean_max',
        dropout: float = 0.5
    ):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout

        # GNN layers
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Classifier
        pool_dim = hidden_dim * 2 if pooling == 'mean_max' else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Class logits [batch_size, num_classes]
        """
        # GNN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # Pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'mean_max':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        return self.classifier(x)

    def get_graph_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Get graph embedding before classification."""
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return global_mean_pool(x, batch)


class AttentionPooling(nn.Module):
    """
    Attention-based graph pooling.

    Learns to weight nodes by importance for graph-level representation.

    Args:
        hidden_dim: Node embedding dimension

    Example:
        >>> pool = AttentionPooling(64)
        >>> graph_emb, attn_weights = pool(node_emb, batch)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node embeddings [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embeddings [batch_size, hidden_dim]
            Attention weights [num_nodes]
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)

        # Softmax within each graph
        attn_weights = softmax(attn_scores, batch)

        # Weighted sum
        weighted = x * attn_weights.unsqueeze(-1)
        out = global_add_pool(weighted, batch)

        return out, attn_weights


if __name__ == "__main__":
    # Quick test
    print("Testing GNN layers...")

    # Create dummy data
    num_nodes = 100
    num_edges = 300
    in_channels = 16
    out_channels = 8

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    batch[50:] = 1  # Two graphs

    # Test GCN layer
    gcn = GCNLayer(in_channels, out_channels)
    out = gcn(x, edge_index)
    print(f"GCN layer: {x.shape} -> {out.shape}")

    # Test GAT layer
    gat = GATLayer(in_channels, out_channels, heads=4)
    out = gat(x, edge_index)
    print(f"GAT layer: {x.shape} -> {out.shape}")

    # Test Graph Classifier
    clf = GraphClassifier(in_channels, hidden_dim=32, num_classes=2)
    out = clf(x, edge_index, batch)
    print(f"Graph Classifier: batch of 2 -> {out.shape}")

    print("\nAll tests passed!")
