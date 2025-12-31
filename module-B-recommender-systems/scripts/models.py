"""
Recommender System Models for Module B.

This module provides PyTorch implementations of various recommender system
architectures, from classic matrix factorization to modern neural approaches.

Professor SPARK's Note:
    "Start simple, then add complexity. Matrix Factorization still beats
    neural methods on many datasets - always have a strong baseline!"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


# =============================================================================
# Matrix Factorization Models
# =============================================================================

class MatrixFactorization(nn.Module):
    """
    Classic Matrix Factorization for collaborative filtering.

    Rating = User_embedding · Item_embedding + user_bias + item_bias + global_bias

    This is the model that won the Netflix Prize! Simple but effective.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of latent factors (higher = more expressive but may overfit)

    Example:
        >>> model = MatrixFactorization(num_users=1000, num_items=5000, embedding_dim=64)
        >>> user_ids = torch.LongTensor([0, 1, 2])
        >>> item_ids = torch.LongTensor([10, 20, 30])
        >>> predictions = model(user_ids, item_ids)
        >>> print(predictions.shape)
        torch.Size([3])
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict ratings for user-item pairs.

        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
            item_ids: Tensor of item IDs, shape (batch_size,)

        Returns:
            Predicted ratings, shape (batch_size,)
        """
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)  # (batch, dim)
        item_emb = self.item_embeddings(item_ids)  # (batch, dim)

        # Dot product interaction
        interaction = (user_emb * item_emb).sum(dim=1)  # (batch,)

        # Add biases
        prediction = (
            interaction +
            self.user_bias(user_ids).squeeze() +
            self.item_bias(item_ids).squeeze() +
            self.global_bias
        )

        return prediction

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding vector for a user."""
        with torch.no_grad():
            return self.user_embeddings(torch.LongTensor([user_id])).numpy().squeeze()

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get embedding vector for an item."""
        with torch.no_grad():
            return self.item_embeddings(torch.LongTensor([item_id])).numpy().squeeze()


class BPRMatrixFactorization(nn.Module):
    """
    Matrix Factorization with Bayesian Personalized Ranking loss.

    BPR is better for implicit feedback (clicks, views) where we only
    have positive signals. It learns to rank positive items above negatives.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of latent factors

    Example:
        >>> model = BPRMatrixFactorization(1000, 5000, 64)
        >>> user = torch.LongTensor([0])
        >>> pos_item = torch.LongTensor([10])
        >>> neg_item = torch.LongTensor([20])
        >>> pos_score, neg_score = model(user, pos_item, neg_item)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64
    ):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores for positive and negative items.

        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        user_emb = self.user_embeddings(user_ids)
        pos_emb = self.item_embeddings(pos_item_ids)
        neg_emb = self.item_embeddings(neg_item_ids)

        pos_score = (user_emb * pos_emb).sum(dim=-1)
        neg_score = (user_emb * neg_emb).sum(dim=-1)

        return pos_score, neg_score

    @staticmethod
    def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        """
        BPR loss: maximize difference between positive and negative scores.

        loss = -log(sigmoid(pos_score - neg_score))
        """
        return -F.logsigmoid(pos_score - neg_score).mean()


# =============================================================================
# Neural Collaborative Filtering
# =============================================================================

class GMF(nn.Module):
    """
    Generalized Matrix Factorization.

    Like classic MF, but with a learnable output layer instead of fixed dot product.
    This allows learning more complex interactions between user and item embeddings.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of embeddings
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64
    ):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.output = nn.Linear(embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Element-wise product
        element_product = user_emb * item_emb

        # Learned combination
        output = torch.sigmoid(self.output(element_product).squeeze())

        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron path for NCF.

    Concatenates user and item embeddings, then passes through MLP layers.
    Can learn non-linear interactions between users and items.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Dimension of embeddings
        hidden_layers: List of hidden layer sizes
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Build MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Concatenated embeddings

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Concatenate
        concat = torch.cat([user_emb, item_emb], dim=1)

        # Pass through MLP
        hidden = self.mlp(concat)
        output = torch.sigmoid(self.output(hidden).squeeze())

        return output


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) - combines GMF and MLP.

    The key insight: GMF captures linear interactions, MLP captures non-linear.
    Combining them gives the best of both worlds!

    Paper: "Neural Collaborative Filtering" (He et al., WWW 2017)

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        gmf_dim: Embedding dimension for GMF path
        mlp_dim: Embedding dimension for MLP path
        mlp_layers: Hidden layer sizes for MLP
        dropout: Dropout probability

    Example:
        >>> model = NeuMF(num_users=1000, num_items=5000)
        >>> users = torch.LongTensor([0, 1, 2])
        >>> items = torch.LongTensor([10, 20, 30])
        >>> predictions = model(users, items)  # Probabilities [0, 1]
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_dim: int = 32,
        mlp_dim: int = 64,
        mlp_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

        # GMF pathway - separate embeddings
        self.gmf_user_emb = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_dim)

        # MLP pathway - separate embeddings
        self.mlp_user_emb = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_dim)

        # MLP layers
        layers = []
        input_dim = mlp_dim * 2

        for hidden_dim in mlp_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Final prediction layer: combines GMF + MLP outputs
        self.output = nn.Linear(gmf_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small random weights."""
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.xavier_uniform_(self.output.weight)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict interaction probability.

        Args:
            user_ids: User IDs, shape (batch_size,)
            item_ids: Item IDs, shape (batch_size,)

        Returns:
            Interaction probabilities, shape (batch_size,)
        """
        # GMF pathway: element-wise product
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP pathway: concatenate then MLP
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = torch.sigmoid(self.output(combined).squeeze(-1))

        return prediction

    def predict_all_items(self, user_id: int) -> torch.Tensor:
        """
        Predict scores for all items for a given user.

        Useful for generating top-K recommendations.

        Args:
            user_id: Single user ID

        Returns:
            Scores for all items, shape (num_items,)
        """
        user_ids = torch.LongTensor([user_id] * self.num_items)
        item_ids = torch.arange(self.num_items)

        with torch.no_grad():
            return self(user_ids, item_ids)


# =============================================================================
# Two-Tower Architecture
# =============================================================================

class QueryTower(nn.Module):
    """
    Query (User) tower for two-tower retrieval.

    Encodes user features into a dense embedding for similarity search.

    Args:
        input_dim: Dimension of user feature vector
        embedding_dim: Output embedding dimension
        hidden_dims: Hidden layer sizes
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features to embedding."""
        embedding = self.network(user_features)
        # L2 normalize for cosine similarity
        return F.normalize(embedding, p=2, dim=-1)


class ItemTower(nn.Module):
    """
    Item tower for two-tower retrieval.

    Encodes item features into a dense embedding for similarity search.

    Args:
        input_dim: Dimension of item feature vector
        embedding_dim: Output embedding dimension
        hidden_dims: Hidden layer sizes
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, embedding_dim))
        layers.append(nn.LayerNorm(embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features to embedding."""
        embedding = self.network(item_features)
        return F.normalize(embedding, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower Model for large-scale retrieval.

    This architecture is used at Google, YouTube, and many other companies
    for candidate retrieval from billions of items.

    The key insight: encode queries and items into the same embedding space,
    then use fast approximate nearest neighbor search for retrieval.

    Args:
        user_feature_dim: Dimension of user features
        item_feature_dim: Dimension of item features
        embedding_dim: Shared embedding dimension
        hidden_dims: Hidden layer sizes for both towers

    Example:
        >>> model = TwoTowerModel(user_feature_dim=64, item_feature_dim=128)
        >>> user_features = torch.randn(32, 64)  # Batch of 32 users
        >>> item_features = torch.randn(32, 128)  # Batch of 32 items
        >>> logits = model(user_features, item_features)  # (32, 32) similarity matrix
    """

    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        temperature: float = 0.07
    ):
        super().__init__()

        self.query_tower = QueryTower(user_feature_dim, embedding_dim, hidden_dims)
        self.item_tower = ItemTower(item_feature_dim, embedding_dim, hidden_dims)

        # Learnable temperature for scaling
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def encode_query(self, user_features: torch.Tensor) -> torch.Tensor:
        """Encode user features to query embedding."""
        return self.query_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode item features to item embedding."""
        return self.item_tower(item_features)

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between all user-item pairs in batch.

        Args:
            user_features: (batch_size, user_feature_dim)
            item_features: (batch_size, item_feature_dim)

        Returns:
            Similarity logits: (batch_size, batch_size)
            Diagonal contains positive pairs, off-diagonal are in-batch negatives
        """
        query_emb = self.encode_query(user_features)
        item_emb = self.encode_item(item_features)

        # Compute all pairwise similarities
        logits = torch.matmul(query_emb, item_emb.T) / self.temperature

        return logits

    def compute_similarity(
        self,
        query_emb: torch.Tensor,
        item_emb: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between query and item embeddings."""
        return torch.matmul(query_emb, item_emb.T) / self.temperature


class TwoTowerWithIDs(nn.Module):
    """
    Two-Tower Model using ID embeddings instead of feature vectors.

    Simpler variant when you don't have rich features - just learn
    embeddings from user/item IDs like matrix factorization.

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_dim: Embedding dimension
        hidden_dims: Hidden layer sizes
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [256],
        temperature: float = 0.07
    ):
        super().__init__()

        # ID embeddings
        self.user_embeddings = nn.Embedding(num_users, hidden_dims[0])
        self.item_embeddings = nn.Embedding(num_items, hidden_dims[0])

        # Towers process the ID embeddings
        self.query_tower = nn.Sequential(
            nn.Linear(hidden_dims[0], embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.item_tower = nn.Sequential(
            nn.Linear(hidden_dims[0], embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.temperature = nn.Parameter(torch.tensor(temperature))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def encode_query(self, user_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        query_emb = self.query_tower(user_emb)
        return F.normalize(query_emb, p=2, dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embeddings(item_ids)
        item_emb = self.item_tower(item_emb)
        return F.normalize(item_emb, p=2, dim=-1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        query_emb = self.encode_query(user_ids)
        item_emb = self.encode_item(item_ids)

        logits = torch.matmul(query_emb, item_emb.T) / self.temperature
        return logits


# =============================================================================
# Loss Functions
# =============================================================================

def in_batch_negative_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    In-batch negative loss for two-tower training.

    The key insight: positives are on the diagonal, everything else is negative.
    This is very efficient because negatives come "for free" from the batch!

    Args:
        logits: Similarity matrix (batch_size, batch_size)
        labels: Optional labels (defaults to diagonal)

    Returns:
        Cross-entropy loss

    Example:
        >>> logits = model(user_features, item_features)  # (32, 32)
        >>> loss = in_batch_negative_loss(logits)
    """
    batch_size = logits.shape[0]

    if labels is None:
        # Positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=logits.device)

    return F.cross_entropy(logits, labels)


def bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.

    Maximizes the difference between positive and negative scores.

    Args:
        pos_score: Scores for positive items
        neg_score: Scores for negative items

    Returns:
        BPR loss value
    """
    return -F.logsigmoid(pos_score - neg_score).mean()


def weighted_bce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float = 1.0
) -> torch.Tensor:
    """
    Weighted binary cross-entropy for imbalanced implicit feedback.

    Args:
        predictions: Model predictions
        targets: Binary labels (1 for positive, 0 for negative)
        pos_weight: Weight for positive examples

    Returns:
        Weighted BCE loss
    """
    weights = torch.where(targets == 1, pos_weight, 1.0)
    bce = F.binary_cross_entropy(predictions, targets, reduction='none')
    return (bce * weights).mean()


# =============================================================================
# Utility Functions
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print a summary of the model architecture."""
    print(f"\n{'='*50}")
    print(f"{model_name} Summary")
    print(f"{'='*50}")
    print(model)
    print(f"{'─'*50}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Quick tests
    print("Testing recommender models...")

    # Test Matrix Factorization
    mf = MatrixFactorization(1000, 5000, 64)
    users = torch.LongTensor([0, 1, 2])
    items = torch.LongTensor([10, 20, 30])
    preds = mf(users, items)
    print(f"✅ MatrixFactorization: {preds.shape}")

    # Test NeuMF
    neumf = NeuMF(1000, 5000)
    preds = neumf(users, items)
    print(f"✅ NeuMF: {preds.shape}")

    # Test Two-Tower
    tt = TwoTowerModel(64, 128, 64)
    user_feats = torch.randn(4, 64)
    item_feats = torch.randn(4, 128)
    logits = tt(user_feats, item_feats)
    print(f"✅ TwoTowerModel: {logits.shape}")

    # Test loss
    loss = in_batch_negative_loss(logits)
    print(f"✅ In-batch negative loss: {loss.item():.4f}")

    print("\n✅ All model tests passed!")
