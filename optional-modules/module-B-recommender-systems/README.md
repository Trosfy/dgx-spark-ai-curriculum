# Optional Module B: Recommender Systems

**Category:** Optional - Applied ML
**Duration:** 6-8 hours
**Prerequisites:** Module 2.1 (PyTorch), Module 2.3 (NLP & Transformers)
**Priority:** P3 (Optional - Industry Applications)

---

## Overview

Recommender systems power the personalized experiences you encounter daily - Netflix suggestions, Amazon products, Spotify playlists, TikTok feeds. This module teaches you to build these systems from classical collaborative filtering to modern neural approaches used at scale.

**Why This Matters:** Recommendations drive engagement and revenue for most tech companies. Understanding these systems opens doors to high-impact roles in industry and gives you tools applicable to any personalization problem.

### The Kitchen Table Explanation

Imagine you're recommending restaurants to friends. For some friends, you know their taste well - "Sarah loves sushi, so she'll like this new omakase place." That's *content-based filtering*. For others, you think "Tom and Sarah have similar taste, and Tom loved it, so Sarah probably will too." That's *collaborative filtering*. Modern systems combine both approaches with neural networks that learn what "similar taste" really means.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Implement collaborative filtering from scratch
- ✅ Build neural recommendation models with embeddings
- ✅ Design two-tower architectures for large-scale retrieval
- ✅ Evaluate recommenders with appropriate metrics

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| B.1 | Implement matrix factorization for collaborative filtering | Apply |
| B.2 | Build neural collaborative filtering with PyTorch | Apply |
| B.3 | Design and train two-tower retrieval models | Create |
| B.4 | Evaluate recommenders with ranking metrics (NDCG, MAP) | Analyze |

---

## Topics

### B.1 Collaborative Filtering Fundamentals

- **User-Item Matrix**
  - Explicit vs implicit feedback
  - Sparsity challenges
  - Cold start problem

- **Memory-Based Methods**
  - User-based collaborative filtering
  - Item-based collaborative filtering
  - Similarity metrics (cosine, Pearson)

- **Matrix Factorization**
  - SVD and truncated SVD
  - Alternating Least Squares (ALS)
  - Regularization for generalization

### B.2 Neural Collaborative Filtering

- **Embedding Layers**
  - Learning latent representations
  - User and item embeddings
  - Handling new users/items

- **NCF Architecture**
  - Generalized Matrix Factorization (GMF)
  - Multi-Layer Perceptron (MLP) path
  - NeuMF: GMF + MLP fusion

- **Training Techniques**
  - Negative sampling strategies
  - Implicit feedback loss functions
  - Batch construction for recommendations

### B.3 Two-Tower Architecture

- **Retrieval vs Ranking**
  - Candidate generation at scale
  - Why two-tower works for billions of items
  - Approximate nearest neighbor search

- **Architecture Design**
  - Query tower (user features)
  - Item tower (item features)
  - Dot product or cosine similarity

- **Training at Scale**
  - In-batch negatives
  - Hard negative mining
  - Temperature scaling

### B.4 Evaluation and Deployment

- **Ranking Metrics**
  - Precision@K, Recall@K
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG)

- **Offline vs Online Evaluation**
  - A/B testing challenges
  - Counterfactual evaluation
  - Engagement vs diversity tradeoffs

- **Production Considerations**
  - Feature stores
  - Real-time vs batch inference
  - Feedback loops and filter bubbles

---

## Labs

### Lab B.1: Matrix Factorization from Scratch
**Time:** 2 hours

Implement collaborative filtering with matrix factorization on MovieLens.

**Instructions:**
1. Load MovieLens 100K dataset
2. Implement SVD-based matrix factorization
3. Train with ALS optimization
4. Evaluate with RMSE on held-out ratings
5. Visualize learned item embeddings with t-SNE
6. Analyze which movies cluster together

**Deliverable:** Notebook with MF implementation achieving RMSE < 0.95

---

### Lab B.2: Neural Collaborative Filtering
**Time:** 2 hours

Build the NeuMF model from the influential 2017 paper.

**Instructions:**
1. Implement GMF (dot product of embeddings)
2. Implement MLP tower (concatenated embeddings through MLPs)
3. Combine into NeuMF architecture
4. Train with binary cross-entropy (implicit feedback)
5. Compare GMF, MLP, and NeuMF performance
6. Tune embedding dimensions and MLP depth

**Deliverable:** NeuMF model with HR@10 > 0.65 on MovieLens

---

### Lab B.3: Two-Tower Retrieval System
**Time:** 2.5 hours

Build a scalable two-tower model for candidate retrieval.

**Instructions:**
1. Design query tower with user features
2. Design item tower with item features (title embeddings, genres)
3. Train with in-batch negative sampling
4. Export item embeddings for ANN index
5. Build FAISS index for approximate nearest neighbor search
6. Implement real-time retrieval pipeline
7. Measure retrieval quality and latency

**Deliverable:** Working two-tower retrieval system with sub-10ms latency

---

### Lab B.4: Evaluation and Analysis
**Time:** 1.5 hours

Comprehensively evaluate your recommender systems.

**Instructions:**
1. Implement NDCG@K and MAP@K metrics
2. Compare all models on ranking metrics
3. Analyze cold start performance (new users)
4. Examine diversity of recommendations
5. Visualize user-item embedding space
6. Write recommendations for production deployment

**Deliverable:** Evaluation report comparing all models

---

## Guidance

### Matrix Factorization with PyTorch

```python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """
    Classic matrix factorization for collaborative filtering.

    Rating = User_embedding · Item_embedding + user_bias + item_bias + global_bias
    """

    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)

        # Dot product
        interaction = (user_emb * item_emb).sum(dim=1)

        # Add biases
        prediction = (
            interaction +
            self.user_bias(user_ids).squeeze() +
            self.item_bias(item_ids).squeeze() +
            self.global_bias
        )
        return prediction

# Training loop
model = MatrixFactorization(num_users=1000, num_items=5000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

for epoch in range(10):
    for user_ids, item_ids, ratings in dataloader:
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Neural Collaborative Filtering (NeuMF)

```python
class NeuMF(nn.Module):
    """
    Neural Collaborative Filtering combining GMF and MLP.

    Paper: "Neural Collaborative Filtering" (He et al., 2017)
    """

    def __init__(self, num_users, num_items, gmf_dim=32, mlp_dim=64, mlp_layers=[128, 64, 32]):
        super().__init__()

        # GMF pathway
        self.gmf_user_emb = nn.Embedding(num_users, gmf_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_dim)

        # MLP pathway
        self.mlp_user_emb = nn.Embedding(num_users, mlp_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_dim)

        # MLP layers
        mlp_input_dim = mlp_dim * 2
        layers = []
        for hidden_dim in mlp_layers:
            layers.extend([
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            mlp_input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

        # Final prediction layer
        self.output = nn.Linear(gmf_dim + mlp_layers[-1], 1)

    def forward(self, user_ids, item_ids):
        # GMF pathway: element-wise product
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP pathway: concatenate and pass through MLP
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp(mlp_input)

        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = torch.sigmoid(self.output(combined)).squeeze()

        return prediction

# For implicit feedback, use binary cross-entropy
criterion = nn.BCELoss()
```

### Two-Tower Architecture

```python
class TwoTowerModel(nn.Module):
    """
    Two-tower model for large-scale retrieval.

    Query tower: processes user features
    Item tower: processes item features
    Similarity: dot product in embedding space
    """

    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim=128):
        super().__init__()

        # Query (user) tower
        self.query_tower = nn.Sequential(
            nn.Linear(user_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def encode_query(self, user_features):
        return nn.functional.normalize(self.query_tower(user_features), dim=-1)

    def encode_item(self, item_features):
        return nn.functional.normalize(self.item_tower(item_features), dim=-1)

    def forward(self, user_features, item_features):
        query_emb = self.encode_query(user_features)
        item_emb = self.encode_item(item_features)

        # Scaled dot product similarity
        logits = torch.matmul(query_emb, item_emb.T) / self.temperature
        return logits

def in_batch_negative_loss(logits, labels=None):
    """
    In-batch negatives: positives on diagonal, rest are negatives.
    This is efficient because negatives come "for free" from the batch.
    """
    batch_size = logits.shape[0]
    if labels is None:
        labels = torch.arange(batch_size, device=logits.device)
    return nn.functional.cross_entropy(logits, labels)
```

### Evaluation Metrics

```python
import numpy as np

def ndcg_at_k(relevances, k):
    """
    Compute Normalized Discounted Cumulative Gain @ K.

    Args:
        relevances: List of relevance scores in ranked order
        k: Cutoff position
    """
    relevances = np.array(relevances)[:k]

    # DCG
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    dcg = np.sum(relevances / discounts)

    # Ideal DCG (perfect ranking)
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = np.sum(ideal_relevances / discounts)

    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(predictions, ground_truth, k):
    """
    Hit Rate @ K: Was the true item in top K predictions?

    Common metric for implicit feedback evaluation.
    """
    top_k = predictions[:k]
    return 1.0 if ground_truth in top_k else 0.0

def mean_average_precision(predictions_list, ground_truth_list, k):
    """
    Mean Average Precision @ K across all users.
    """
    aps = []
    for preds, gt in zip(predictions_list, ground_truth_list):
        # Compute precision at each position where relevant item appears
        hits = 0
        sum_precisions = 0
        for i, pred in enumerate(preds[:k]):
            if pred in gt:
                hits += 1
                sum_precisions += hits / (i + 1)
        ap = sum_precisions / min(len(gt), k) if gt else 0
        aps.append(ap)
    return np.mean(aps)
```

### FAISS for Fast Retrieval

```python
import faiss
import numpy as np

def build_faiss_index(embeddings, use_gpu=True):
    """
    Build FAISS index for fast approximate nearest neighbor search.

    On DGX Spark with 128GB unified memory, you can index
    hundreds of millions of items efficiently.
    """
    dim = embeddings.shape[1]
    num_items = embeddings.shape[0]

    # For small datasets: exact search
    if num_items < 100_000:
        index = faiss.IndexFlatIP(dim)  # Inner product (for normalized vectors)
    else:
        # For large datasets: IVF with PQ compression
        nlist = int(np.sqrt(num_items))  # Number of clusters
        m = 32  # Number of subvectors for PQ
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
        index.train(embeddings)

    if use_gpu:
        # Move to GPU for faster search
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    return index

def retrieve_candidates(index, query_embedding, k=100):
    """
    Retrieve top-k candidates for a query.
    """
    scores, indices = index.search(query_embedding.reshape(1, -1), k)
    return indices[0], scores[0]

# Example usage:
# item_embeddings = model.encode_item(all_items)  # Shape: [num_items, 128]
# index = build_faiss_index(item_embeddings.numpy())
# candidates, scores = retrieve_candidates(index, user_embedding.numpy(), k=100)
```

### DGX Spark Optimization

> **DGX Spark Tip:** With 128GB unified memory, you can:
> - Store millions of user/item embeddings in GPU memory
> - Use larger embedding dimensions (256-512) for richer representations
> - Train on full user interaction history without sampling
> - Run FAISS GPU index for real-time retrieval

---

## 📖 Study Materials

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](./QUICKSTART.md) | Build your first recommender in 10 minutes |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | MF, NeuMF, Two-Tower patterns at a glance |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Debug embedding collapse, cold start, and more |
| [WORKFLOWS.md](./WORKFLOWS.md) | Step-by-step workflows for common tasks |

---

## Milestone Checklist

- [ ] Matrix factorization achieving RMSE < 0.95
- [ ] NeuMF model with HR@10 > 0.65
- [ ] Two-tower model with working retrieval
- [ ] FAISS index with sub-10ms queries
- [ ] Comprehensive evaluation with NDCG and MAP
- [ ] Analysis of cold start and diversity

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Embeddings collapse to same vector | Add regularization, check learning rate |
| Cold start poor performance | Use content features in two-tower |
| Training too slow | Use in-batch negatives, reduce negative samples |
| Recommendations lack diversity | Add diversity penalty or MMR reranking |

---

## Why This Module is Optional

Recommender systems are highly specialized - most AI practitioners won't build them directly. However, the techniques transfer broadly:

1. **Embedding layers** - Foundational for all deep learning
2. **Two-tower architecture** - Used in semantic search, RAG retrieval
3. **Negative sampling** - Critical for contrastive learning
4. **Ranking metrics** - Applicable to any retrieval system

---

## Next Steps

After completing this module:
1. Apply two-tower retrieval to your RAG system (Module 3.5)
2. Consider adding personalization to your capstone
3. Explore sequence models for recommendations (transformers on user history)

---

## Resources

- [Neural Collaborative Filtering Paper](https://arxiv.org/abs/1708.05031)
- [YouTube Recommendations Paper](https://research.google/pubs/pub45530/) - Two-tower at scale
- [Microsoft Recommenders Library](https://github.com/microsoft/recommenders)
- [FAISS Documentation](https://faiss.ai/)
- [RecBole Framework](https://recbole.io/) - Unified recommender systems library

