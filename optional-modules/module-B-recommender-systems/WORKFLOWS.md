# Module B: Recommender Systems - Workflow Cheatsheets

## Workflow 1: Building a Matrix Factorization Recommender

### 📋 When to Use
When you have explicit ratings (1-5 stars) and want a simple, interpretable baseline.

### 🔄 Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Load and Prepare Data                               │
├─────────────────────────────────────────────────────────────┤
│ □ Load ratings data (user_id, item_id, rating)              │
│ □ Reindex IDs to be consecutive (0 to N-1)                  │
│ □ Split into train/validation/test                          │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ from sklearn.preprocessing import LabelEncoder              │
│                                                             │
│ user_enc = LabelEncoder()                                   │
│ item_enc = LabelEncoder()                                   │
│ df['user_id'] = user_enc.fit_transform(df['user_id'])      │
│ df['item_id'] = item_enc.fit_transform(df['item_id'])      │
│                                                             │
│ # Temporal split (recommended)                              │
│ df = df.sort_values('timestamp')                           │
│ train = df[:int(0.8*len(df))]                              │
│ val = df[int(0.8*len(df)):int(0.9*len(df))]               │
│ test = df[int(0.9*len(df)):]                               │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: print(f"Users: {n_users}, Items: {n_items}") │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Create Model                                        │
├─────────────────────────────────────────────────────────────┤
│ □ Define embedding dimensions                               │
│ □ Create user and item embeddings                           │
│ □ Add bias terms                                            │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ class MF(nn.Module):                                        │
│     def __init__(self, n_users, n_items, dim=64):          │
│         super().__init__()                                  │
│         self.user_emb = nn.Embedding(n_users, dim)         │
│         self.item_emb = nn.Embedding(n_items, dim)         │
│         self.user_bias = nn.Embedding(n_users, 1)          │
│         self.item_bias = nn.Embedding(n_items, 1)          │
│         nn.init.normal_(self.user_emb.weight, std=0.01)    │
│         nn.init.normal_(self.item_emb.weight, std=0.01)    │
│                                                             │
│     def forward(self, u, i):                               │
│         return (self.user_emb(u) * self.item_emb(i)).sum(1)│
│                + self.user_bias(u).squeeze()               │
│                + self.item_bias(i).squeeze()               │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: model summary shows expected parameters       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Training Loop                                       │
├─────────────────────────────────────────────────────────────┤
│ □ Set optimizer with weight decay                           │
│ □ Use MSE loss for ratings                                  │
│ □ Track validation RMSE                                     │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ optimizer = torch.optim.Adam(model.parameters(),            │
│                              lr=1e-3, weight_decay=1e-5)    │
│ criterion = nn.MSELoss()                                    │
│                                                             │
│ for epoch in range(50):                                     │
│     model.train()                                           │
│     for users, items, ratings in train_loader:              │
│         pred = model(users.cuda(), items.cuda())           │
│         loss = criterion(pred, ratings.cuda())              │
│         optimizer.zero_grad()                               │
│         loss.backward()                                     │
│         optimizer.step()                                    │
│                                                             │
│     # Validate                                              │
│     val_rmse = evaluate(model, val_loader)                 │
│     print(f"Epoch {epoch}: Val RMSE = {val_rmse:.4f}")     │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: RMSE decreasing, < 1.0 on MovieLens          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Evaluate and Visualize                              │
├─────────────────────────────────────────────────────────────┤
│ □ Compute test RMSE                                         │
│ □ Visualize embeddings with t-SNE                          │
│ □ Find similar items                                        │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ # Test RMSE                                                 │
│ test_rmse = evaluate(model, test_loader)                   │
│                                                             │
│ # Visualize item embeddings                                 │
│ from sklearn.manifold import TSNE                          │
│ emb = model.item_emb.weight.detach().cpu().numpy()         │
│ tsne = TSNE(n_components=2)                                │
│ emb_2d = tsne.fit_transform(emb)                           │
│ plt.scatter(emb_2d[:,0], emb_2d[:,1], c=item_genres)       │
│                                                             │
│ # Find similar items                                        │
│ def similar_items(item_id, k=5):                           │
│     item_vec = model.item_emb.weight[item_id]              │
│     sims = (model.item_emb.weight @ item_vec).argsort()    │
│     return sims[-k-1:-1].flip(0)                           │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Embeddings cluster by genre                   │
└─────────────────────────────────────────────────────────────┘
```

### ⚠️ Common Pitfalls

| At Step | Watch Out For |
|---------|---------------|
| 1 | Non-consecutive IDs cause IndexError |
| 2 | Large embedding dim = more memory, risk of overfitting |
| 3 | No weight decay = embeddings may collapse |
| 4 | t-SNE perplexity too high = meaningless visualization |

### ✅ Success Criteria

- [ ] RMSE < 0.95 on MovieLens 100K
- [ ] Embeddings show genre clustering
- [ ] Similar item lookups make sense

---

## Workflow 2: Building a Two-Tower Retrieval System

### 📋 When to Use
When you have millions of items and need real-time retrieval (< 100ms).

### 🔄 Step-by-Step

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Prepare Features                                    │
├─────────────────────────────────────────────────────────────┤
│ □ Extract user features (history stats, demographics)       │
│ □ Extract item features (title embeddings, metadata)        │
│ □ Normalize features                                        │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ # User features: aggregated history                         │
│ user_features = df.groupby('user_id').agg({                │
│     'rating': ['mean', 'count'],                           │
│     'genre_action': 'mean',                                │
│     'genre_comedy': 'mean',                                │
│     ...                                                     │
│ })                                                          │
│                                                             │
│ # Item features: text embeddings + metadata                 │
│ from sentence_transformers import SentenceTransformer      │
│ encoder = SentenceTransformer('all-MiniLM-L6-v2')          │
│ title_emb = encoder.encode(item_titles)                    │
│ item_features = np.hstack([title_emb, genre_onehot])       │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Feature matrices have expected shapes         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Build Two-Tower Model                               │
├─────────────────────────────────────────────────────────────┤
│ □ Create user tower (features → embedding)                  │
│ □ Create item tower (features → embedding)                  │
│ □ Use L2 normalization for cosine similarity               │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ class TwoTower(nn.Module):                                  │
│     def __init__(self, user_dim, item_dim, emb_dim=128):   │
│         super().__init__()                                  │
│         self.user_tower = nn.Sequential(                   │
│             nn.Linear(user_dim, 256),                      │
│             nn.ReLU(),                                      │
│             nn.Linear(256, emb_dim),                       │
│             nn.LayerNorm(emb_dim)                          │
│         )                                                   │
│         self.item_tower = nn.Sequential(                   │
│             nn.Linear(item_dim, 256),                      │
│             nn.ReLU(),                                      │
│             nn.Linear(256, emb_dim),                       │
│             nn.LayerNorm(emb_dim)                          │
│         )                                                   │
│         self.temp = nn.Parameter(torch.tensor(0.07))       │
│                                                             │
│     def forward(self, user_feat, item_feat):               │
│         u = F.normalize(self.user_tower(user_feat), dim=-1)│
│         i = F.normalize(self.item_tower(item_feat), dim=-1)│
│         return (u @ i.T) / self.temp                       │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Output is [batch, batch] similarity matrix   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Train with In-Batch Negatives                       │
├─────────────────────────────────────────────────────────────┤
│ □ Batch has (user, positive_item) pairs                     │
│ □ Other items in batch are negatives                        │
│ □ Use cross-entropy loss (softmax over similarities)        │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ def train_step(model, user_batch, item_batch):              │
│     logits = model(user_batch, item_batch)                 │
│     # Positives are on diagonal                             │
│     labels = torch.arange(len(user_batch), device=device)  │
│     loss = F.cross_entropy(logits, labels)                 │
│     return loss                                             │
│                                                             │
│ # Large batches = more negatives = better training         │
│ batch_size = 2048  # Use DGX Spark's memory!               │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Loss decreasing, validation HR improving      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Build FAISS Index                                   │
├─────────────────────────────────────────────────────────────┤
│ □ Encode all items with item tower                          │
│ □ Build FAISS index                                         │
│ □ Optimize for your catalog size                            │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ import faiss                                                │
│                                                             │
│ # Encode all items                                          │
│ model.eval()                                                │
│ with torch.no_grad():                                       │
│     item_emb = model.item_tower(all_item_features)         │
│     item_emb = F.normalize(item_emb, dim=-1)               │
│ item_emb = item_emb.cpu().numpy()                          │
│                                                             │
│ # Build index                                               │
│ dim = item_emb.shape[1]                                    │
│ index = faiss.IndexFlatIP(dim)  # Inner product            │
│ index.add(item_emb)                                         │
│                                                             │
│ # GPU acceleration                                          │
│ res = faiss.StandardGpuResources()                         │
│ index_gpu = faiss.index_cpu_to_gpu(res, 0, index)          │
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Index has correct number of items             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Implement Retrieval Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│ □ Encode user at serving time                               │
│ □ Search FAISS index                                        │
│ □ Return top-K candidates                                   │
│                                                             │
│ Code:                                                       │
│ ```python                                                   │
│ def retrieve(user_features, k=100):                        │
│     # Encode user                                           │
│     with torch.no_grad():                                   │
│         user_emb = model.user_tower(user_features)         │
│         user_emb = F.normalize(user_emb, dim=-1)           │
│     user_emb = user_emb.cpu().numpy()                      │
│                                                             │
│     # Search                                                │
│     scores, indices = index_gpu.search(user_emb, k)        │
│     return indices[0], scores[0]                           │
│                                                             │
│ # Measure latency                                           │
│ import time                                                 │
│ start = time.time()                                         │
│ for _ in range(100):                                        │
│     retrieve(test_user_features[0:1])                      │
│ print(f"Avg latency: {(time.time()-start)/100*1000:.1f}ms")│
│ ```                                                         │
│                                                             │
│ ✓ Checkpoint: Latency < 10ms, candidates make sense        │
└─────────────────────────────────────────────────────────────┘
```

### ⚠️ Common Pitfalls

| At Step | Watch Out For |
|---------|---------------|
| 1 | Missing normalization = unstable training |
| 2 | No temperature = gradients vanish |
| 3 | Small batch size = weak negatives |
| 4 | CPU index = slow, use GPU |
| 5 | Forgetting normalization at serving time |

### ✅ Success Criteria

- [ ] Retrieval latency < 10ms
- [ ] HR@100 > 0.5 (relevant item in top 100 candidates)
- [ ] Index fits in GPU memory

---

## 🔀 Decision Flowchart: Which Approach?

```
                    Start
                      │
                      ▼
              ┌───────────────┐
              │ Have explicit │
              │   ratings?    │
              └───────┬───────┘
                      │
            ┌─────────┴─────────┐
            │ Yes               │ No (clicks, views)
            ▼                   ▼
    ┌───────────────┐   ┌───────────────┐
    │  Start with   │   │  Start with   │
    │ Matrix Factor │   │ NeuMF/NCF     │
    └───────┬───────┘   └───────┬───────┘
            │                   │
            ▼                   ▼
    ┌───────────────┐   ┌───────────────┐
    │ Need < 10ms   │   │ Need < 10ms   │
    │  retrieval?   │   │  retrieval?   │
    └───────┬───────┘   └───────┬───────┘
            │                   │
      ┌─────┴─────┐       ┌─────┴─────┐
      │ No        │ Yes   │ No        │ Yes
      ▼           ▼       ▼           ▼
  [Keep MF]  [Two-Tower] [Keep NCF] [Two-Tower]
```

---

## Workflow 3: Comprehensive Evaluation

### 📋 When to Use
After training any recommender, to properly measure its quality.

### 🔄 Step-by-Step

```
Step 1: Implement Metrics
────────────────────────
□ NDCG@K - position-aware ranking quality
□ HR@K - did we include the relevant item?
□ MAP@K - precision at each position

Step 2: Proper Train/Test Split
───────────────────────────────
□ Use temporal split (not random!)
□ Leave-one-out or leave-k-out
□ Ensure no data leakage

Step 3: Compute Metrics
───────────────────────
□ For each test user:
  - Get model's top-K predictions
  - Compare against ground truth
  - Compute per-user metrics
□ Average across users

Step 4: Compare Models
──────────────────────
□ Same test set for all models
□ Statistical significance tests
□ Consider diversity and coverage
```

### ✅ Success Criteria

- [ ] Using temporal split (not random)
- [ ] Reporting multiple metrics (NDCG, HR, MAP)
- [ ] Comparing against a baseline
- [ ] Checking for diversity issues
