# Module B: Recommender Systems - Troubleshooting & FAQ

## 🔍 Quick Diagnostic

**Before diving into specific errors, check:**
1. GPU memory: `nvidia-smi` - embeddings can be memory-hungry
2. Data shapes: `print(tensor.shape)` at each step
3. Value ranges: Are predictions in expected range (1-5 for ratings)?

---

## 🚨 Error Categories

### Embedding Issues

#### Error: Embeddings Collapse to Same Vector

**Symptoms:**
- All users/items have nearly identical embeddings
- Predictions are all the same value
- Loss stops decreasing early

**Causes:**
1. Learning rate too high
2. No regularization
3. Initialization problem

**Solutions:**
```python
# Solution 1: Add L2 regularization (weight decay)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Solution 2: Proper initialization
nn.init.normal_(self.user_emb.weight, std=0.01)  # Not std=1.0!
nn.init.normal_(self.item_emb.weight, std=0.01)

# Solution 3: Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Was 1e-2

# Check for collapse:
def check_embedding_collapse(model):
    user_emb = model.user_emb.weight.detach()
    similarity = torch.nn.functional.cosine_similarity(
        user_emb[0:1], user_emb[1:], dim=1
    )
    print(f"Mean cosine similarity: {similarity.mean():.4f}")
    if similarity.mean() > 0.9:
        print("⚠️ WARNING: Embeddings may be collapsing!")
```

---

#### Error: `IndexError: index out of range in self`

**Symptoms:**
```
IndexError: index out of range in self
```

**Cause:** User or item ID exceeds embedding table size.

**Solutions:**
```python
# Check your ID ranges
print(f"Max user ID: {user_ids.max()}, Embedding size: {model.user_emb.num_embeddings}")
print(f"Max item ID: {item_ids.max()}, Embedding size: {model.item_emb.num_embeddings}")

# Fix: Reindex IDs to be consecutive starting from 0
from sklearn.preprocessing import LabelEncoder

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

user_ids = user_encoder.fit_transform(raw_user_ids)
item_ids = item_encoder.fit_transform(raw_item_ids)

n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)
```

---

### Training Issues

#### Issue: Loss Not Decreasing

**Symptoms:**
- Loss stays flat or oscillates
- Validation metrics don't improve

**Diagnostic and Solutions:**

```python
# 1. Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")
    else:
        print(f"{name}: NO GRADIENT!")

# 2. Try different learning rates
for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
    # Train for a few epochs, check loss
    ...

# 3. Check data quality
print(f"Rating range: {ratings.min()} to {ratings.max()}")
print(f"Rating distribution:\n{pd.Series(ratings.numpy()).value_counts()}")

# 4. Simpler model first
# If NeuMF doesn't work, try basic MF first
```

---

#### Issue: Training Very Slow

**Symptoms:**
- Each epoch takes minutes
- GPU utilization low

**Solutions:**
```python
# 1. Use larger batch size (DGX Spark has memory!)
loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=4)

# 2. Use in-batch negatives instead of sampling
# This is much faster than explicit negative sampling
def in_batch_loss(user_emb, item_emb):
    logits = user_emb @ item_emb.T
    labels = torch.arange(len(user_emb), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)

# 3. Move data to GPU once
user_ids = user_ids.cuda()
item_ids = item_ids.cuda()
ratings = ratings.cuda()

# 4. Pin memory for faster transfer
loader = DataLoader(dataset, batch_size=4096, pin_memory=True)
```

---

### Memory Issues

#### Error: `CUDA out of memory`

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Solutions:**
```python
# 1. Reduce embedding dimension
model = MatrixFactorization(n_users, n_items, dim=32)  # Was 128

# 2. Reduce batch size
loader = DataLoader(dataset, batch_size=512)  # Was 4096

# 3. Clear cache between operations
import gc
torch.cuda.empty_cache()
gc.collect()

# 4. For large item catalogs, use chunked inference
def predict_in_chunks(model, user_ids, all_items, chunk_size=10000):
    all_scores = []
    for i in range(0, len(all_items), chunk_size):
        chunk = all_items[i:i+chunk_size]
        with torch.no_grad():
            scores = model(user_ids.expand(len(chunk)), chunk)
        all_scores.append(scores.cpu())
        torch.cuda.empty_cache()
    return torch.cat(all_scores)
```

---

### Evaluation Issues

#### Issue: HR@10 is 0.0 or Very Low

**Symptoms:**
- Hit rate at K is near zero
- Model seems to predict random items

**Causes and Solutions:**

```python
# 1. Check that test items exist in training
train_items = set(train_data['item_id'].unique())
test_items = set(test_data['item_id'].unique())
print(f"Test items not in training: {len(test_items - train_items)}")
# If high, you have cold start issues

# 2. Check prediction format
predictions = get_top_k(model, user_id, k=10)
print(f"Predictions: {predictions}")
print(f"Ground truth: {ground_truth}")
# Ensure they're in same format (item IDs, not indices)

# 3. Check for data leakage
# Make sure test interactions aren't in training!
overlap = set(train_interactions) & set(test_interactions)
assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping interactions"

# 4. Use proper temporal split
# Sort by timestamp, use last N% as test
df = df.sort_values('timestamp')
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
test_df = df[split_idx:]
```

---

#### Issue: NDCG Calculation Gives NaN

**Symptoms:**
```
RuntimeWarning: invalid value encountered in double_scalars
```

**Cause:** Division by zero when ideal DCG is 0.

**Solution:**
```python
def ndcg_at_k(relevances, k):
    rel = np.array(relevances)[:k]
    if len(rel) == 0:
        return 0.0

    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))

    # Handle edge case
    if idcg == 0:
        return 0.0

    return dcg / idcg
```

---

### FAISS Issues

#### Error: FAISS GPU Not Working

**Symptoms:**
```
RuntimeError: Error in faiss::gpu
```

**Solutions:**
```python
# 1. Check FAISS installation
import faiss
print(f"FAISS version: {faiss.__version__}")
print(f"GPU support: {faiss.get_num_gpus()}")

# 2. Fall back to CPU if GPU fails
try:
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
except Exception as e:
    print(f"GPU failed, using CPU: {e}")
    # index stays on CPU

# 3. For ARM64 (DGX Spark), ensure faiss-gpu is installed correctly
# pip install faiss-gpu  # May need specific version
```

---

## ❓ Frequently Asked Questions

### Conceptual Questions

#### Q: What's the difference between collaborative filtering and content-based filtering?

**A:**

| Collaborative Filtering | Content-Based Filtering |
|------------------------|-------------------------|
| Uses user-item interactions | Uses item features |
| "Users like you liked this" | "Items similar to what you liked" |
| Suffers from cold start | Works for new items with features |
| Learns implicit similarity | Uses explicit similarity |

**Best practice:** Use both (hybrid systems)!

---

#### Q: When should I use matrix factorization vs neural models?

**A:**

| Matrix Factorization | Neural Models (NCF, Two-Tower) |
|----------------------|-------------------------------|
| Fewer parameters | More parameters |
| Faster training | Slower training |
| Good for explicit ratings | Better for implicit feedback |
| Limited expressiveness | Can model complex interactions |
| Simpler to debug | Harder to interpret |

**Start with MF**, graduate to neural when you need more expressiveness.

---

#### Q: What's the cold start problem and how do I solve it?

**A:** Cold start = can't recommend for new users or new items with no history.

**Solutions:**

| For New Users | For New Items |
|---------------|---------------|
| Ask for preferences (onboarding) | Use item features (two-tower) |
| Use demographic features | Use item metadata/descriptions |
| Show popular items initially | Similar to existing items |
| Hybrid with content features | Use category/genre defaults |

```python
# Two-tower handles cold start via features
class TwoTowerWithFeatures(nn.Module):
    def __init__(self, user_feat_dim, item_feat_dim, ...):
        # Uses features, not just IDs
        self.user_tower = nn.Linear(user_feat_dim, emb_dim)
        self.item_tower = nn.Linear(item_feat_dim, emb_dim)
        # New items with features can be embedded immediately!
```

---

#### Q: Why use two-tower for retrieval instead of scoring all items?

**A:** Scale and speed!

| Approach | Items | Time |
|----------|-------|------|
| Score all | 1M | ~seconds |
| Two-tower + ANN | 1M | ~milliseconds |

```python
# Two-tower enables pre-computation
item_embeddings = model.encode_item(all_items)  # Compute once
faiss_index = build_index(item_embeddings)       # Index once

# At serving time: encode user, search index
user_emb = model.encode_user(user_features)
candidates = faiss_index.search(user_emb, k=100)  # Fast!
```

---

### Practical Questions

#### Q: How do I handle implicit feedback (clicks, not ratings)?

**A:**
```python
# 1. Binary labels: clicked = 1, not clicked = 0
# 2. Use BCE loss, not MSE
criterion = nn.BCEWithLogitsLoss()

# 3. Sample negatives (items not clicked)
# 4. Consider weighting by engagement (time spent, purchases > clicks)
```

---

#### Q: My recommendations lack diversity. How do I fix this?

**A:** Add diversity through post-processing:

```python
def mmr_reranking(candidates, scores, embeddings, lambda_=0.5, k=10):
    """
    Maximal Marginal Relevance: balance relevance with diversity.
    """
    selected = []
    remaining = list(range(len(candidates)))

    while len(selected) < k and remaining:
        if not selected:
            # First item: highest score
            best = max(remaining, key=lambda i: scores[i])
        else:
            # Balance relevance and diversity
            def mmr_score(i):
                relevance = scores[i]
                max_sim = max(
                    cosine_similarity(embeddings[i], embeddings[j])
                    for j in selected
                )
                return lambda_ * relevance - (1 - lambda_) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]
```

---

#### Q: How do I evaluate during training without being too slow?

**A:**
```python
# 1. Sample a subset of users for validation
val_users = np.random.choice(all_users, size=1000, replace=False)

# 2. Use hit rate (faster than NDCG)
def quick_evaluate(model, val_users, val_items, k=10):
    hits = 0
    for user, item in zip(val_users, val_items):
        top_k = get_top_k(model, user, k)
        if item in top_k:
            hits += 1
    return hits / len(val_users)

# 3. Evaluate every N epochs, not every epoch
if epoch % 10 == 0:
    val_hr = quick_evaluate(model, val_users, val_items)
```

---

### Beyond the Basics

#### Q: How do production recommenders differ from this module?

**A:**

| This Module | Production |
|-------------|------------|
| Single model | Multiple stages (retrieval → ranking) |
| Batch training | Online learning |
| Simple features | Rich feature engineering |
| Accuracy focus | Multi-objective (CTR, revenue, diversity) |
| Offline eval | A/B testing |

---

#### Q: What about sequence-based recommendations?

**A:** Great next step! Use transformers on user history:

```python
# Instead of single embedding per user,
# encode their interaction sequence
class SequenceRecommender(nn.Module):
    def __init__(self, n_items, d_model=64, nhead=4, n_layers=2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            n_layers
        )

    def forward(self, item_sequence):
        x = self.item_emb(item_sequence)
        return self.transformer(x)[:, -1]  # Last position = prediction
```

---

## 🔄 Reset Procedures

### Clear Memory and Restart

```python
import torch
import gc

# Clear CUDA memory
torch.cuda.empty_cache()
gc.collect()

# Verify
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Reinitialize Model

```python
# If model is in bad state
def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
```

---

## 📞 Still Stuck?

1. **Check data quality** - Are ratings in expected range? IDs consecutive?
2. **Try simpler model** - If NeuMF fails, does basic MF work?
3. **Verify shapes** - Print tensor shapes at each step
4. **Check MovieLens baseline** - Known dataset with known results
