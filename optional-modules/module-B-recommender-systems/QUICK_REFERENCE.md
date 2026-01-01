# Module B: Recommender Systems - Quick Reference

## 🚀 Essential Patterns

### Matrix Factorization (PyTorch)

```python
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return (u * i).sum(1) + \
               self.user_bias(user_ids).squeeze() + \
               self.item_bias(item_ids).squeeze() + \
               self.global_bias
```

### Neural Collaborative Filtering (NeuMF)

```python
class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_dim=32, mlp_dim=64):
        super().__init__()
        # GMF pathway
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)

        # MLP pathway
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Combine
        self.output = nn.Linear(gmf_dim + 32, 1)

    def forward(self, user_ids, item_ids):
        gmf = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_input = torch.cat([self.mlp_user(user_ids),
                               self.mlp_item(item_ids)], dim=1)
        mlp = self.mlp(mlp_input)
        return torch.sigmoid(self.output(torch.cat([gmf, mlp], dim=1))).squeeze()
```

### Two-Tower Architecture

```python
class TwoTower(nn.Module):
    def __init__(self, user_feat_dim, item_feat_dim, emb_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        self.temp = nn.Parameter(torch.ones(1) * 0.07)

    def encode_user(self, user_feats):
        return nn.functional.normalize(self.user_tower(user_feats), dim=-1)

    def encode_item(self, item_feats):
        return nn.functional.normalize(self.item_tower(item_feats), dim=-1)

    def forward(self, user_feats, item_feats):
        u = self.encode_user(user_feats)
        i = self.encode_item(item_feats)
        return torch.matmul(u, i.T) / self.temp
```

---

## 📊 Evaluation Metrics

### NDCG@K (Normalized Discounted Cumulative Gain)

```python
import numpy as np

def ndcg_at_k(relevances, k):
    """
    NDCG: Rewards putting relevant items at top of list.
    Perfect ranking = 1.0, random ranking = ~0.3-0.5
    """
    rel = np.array(relevances)[:k]
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sum(np.sort(rel)[::-1] / np.log2(np.arange(2, len(rel) + 2)))
    return dcg / ideal if ideal > 0 else 0.0

# Example
relevances = [3, 2, 0, 0, 1]  # Relevance scores in ranked order
print(f"NDCG@5: {ndcg_at_k(relevances, 5):.3f}")
```

### Hit Rate@K

```python
def hit_rate_at_k(predictions, ground_truth, k):
    """Was the ground truth item in top K?"""
    return 1.0 if ground_truth in predictions[:k] else 0.0
```

### Mean Average Precision (MAP)

```python
def map_at_k(predictions_list, ground_truth_list, k):
    """Average precision across all users."""
    aps = []
    for preds, gt in zip(predictions_list, ground_truth_list):
        hits, prec_sum = 0, 0
        for i, p in enumerate(preds[:k]):
            if p in gt:
                hits += 1
                prec_sum += hits / (i + 1)
        aps.append(prec_sum / min(len(gt), k) if gt else 0)
    return np.mean(aps)
```

---

## 🔧 Training Patterns

### In-Batch Negatives

```python
def in_batch_loss(user_emb, item_emb, temperature=0.07):
    """
    Efficient: negatives come from other items in batch.
    Positives are on diagonal.
    """
    logits = torch.matmul(user_emb, item_emb.T) / temperature
    labels = torch.arange(len(user_emb), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)
```

### Negative Sampling for Implicit Feedback

```python
def sample_negatives(user_ids, n_items, pos_items, n_neg=4):
    """Sample items user hasn't interacted with."""
    neg_items = []
    for user, pos in zip(user_ids, pos_items):
        user_negs = []
        while len(user_negs) < n_neg:
            neg = np.random.randint(0, n_items)
            if neg not in pos:  # pos is set of user's positive items
                user_negs.append(neg)
        neg_items.append(user_negs)
    return torch.tensor(neg_items)
```

---

## 📦 Data Loading

### MovieLens Dataset

```python
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MovieLensDataset(Dataset):
    def __init__(self, ratings_path):
        df = pd.read_csv(ratings_path, sep='\t',
                         names=['user', 'item', 'rating', 'timestamp'])
        self.users = torch.tensor(df['user'].values)
        self.items = torch.tensor(df['item'].values)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Usage
# dataset = MovieLensDataset('ml-100k/u.data')
# loader = DataLoader(dataset, batch_size=256, shuffle=True)
```

---

## ⚡ FAISS for Fast Retrieval

```python
import faiss
import numpy as np

def build_index(embeddings, use_gpu=True):
    """Build FAISS index for approximate nearest neighbor search."""
    dim = embeddings.shape[1]
    n_items = embeddings.shape[0]

    if n_items < 100000:
        index = faiss.IndexFlatIP(dim)  # Exact inner product
    else:
        # IVF for large datasets
        nlist = int(np.sqrt(n_items))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    return index

def retrieve(index, query_emb, k=100):
    """Get top-k similar items."""
    scores, indices = index.search(query_emb.reshape(1, -1), k)
    return indices[0], scores[0]
```

---

## 📊 Key Values

### Embedding Dimensions

| Use Case | User Dim | Item Dim |
|----------|----------|----------|
| Small (< 100K items) | 32-64 | 32-64 |
| Medium (100K-1M items) | 64-128 | 64-128 |
| Large (> 1M items) | 128-256 | 128-256 |
| DGX Spark (with memory) | 256-512 | 256-512 |

### Hyperparameter Ranges

| Parameter | Typical Range | Start With |
|-----------|---------------|------------|
| Learning rate | 1e-4 to 1e-2 | 1e-3 |
| Batch size | 256-4096 | 1024 |
| Embedding dim | 32-256 | 64 |
| Temperature | 0.05-0.2 | 0.07 |
| Weight decay | 1e-6 to 1e-4 | 1e-5 |
| Negative samples | 4-20 | 4 |

### Expected Performance (MovieLens 100K)

| Model | RMSE | HR@10 |
|-------|------|-------|
| Matrix Factorization | < 0.95 | - |
| NeuMF (implicit) | - | > 0.65 |
| Two-Tower | - | > 0.60 |

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Embedding collapse | Add L2 regularization, check lr |
| Cold start failure | Use content features in two-tower |
| Slow training | Use in-batch negatives |
| Poor diversity | Add MMR reranking |
| Data leakage | Use timestamp for train/test split |

---

## 🔗 Quick Links

- Notebook 01: Collaborative Filtering Fundamentals
- Notebook 02: Neural Collaborative Filtering
- Notebook 03: Two-Tower Retrieval
- Notebook 04: Evaluation and Analysis
- [FAISS Documentation](https://faiss.ai/)
- [NCF Paper](https://arxiv.org/abs/1708.05031)
