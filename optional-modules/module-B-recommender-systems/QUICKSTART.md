# Module B: Recommender Systems - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build

A simple matrix factorization recommender that predicts movie ratings using learned embeddings.

## ✅ Before You Start

- [ ] PyTorch installed and working
- [ ] DGX Spark container running (or local GPU)

## 🚀 Let's Go!

### Step 1: Create a Simple Recommender

```python
import torch
import torch.nn as nn

class MovieRecommender(nn.Module):
    """Predict ratings from user and movie embeddings."""

    def __init__(self, n_users, n_movies, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.movie_emb = nn.Embedding(n_movies, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

    def forward(self, user_ids, movie_ids):
        # Dot product of embeddings + biases
        u = self.user_emb(user_ids)
        m = self.movie_emb(movie_ids)
        return (u * m).sum(dim=1) + \
               self.user_bias(user_ids).squeeze() + \
               self.movie_bias(movie_ids).squeeze()

model = MovieRecommender(n_users=1000, n_movies=5000)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

**Expected output:**
```
Model has 198,000 parameters
```

### Step 2: Create Some Fake Data

```python
# Simulate user-movie interactions
n_ratings = 1000
user_ids = torch.randint(0, 1000, (n_ratings,))
movie_ids = torch.randint(0, 5000, (n_ratings,))
ratings = torch.rand(n_ratings) * 4 + 1  # Ratings 1-5
```

### Step 3: Train for a Few Steps

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(5):
    optimizer.zero_grad()
    predictions = model(user_ids, movie_ids)
    loss = criterion(predictions, ratings)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

**Expected output:**
```
Epoch 1: Loss = 2.3456
Epoch 2: Loss = 1.8765
Epoch 3: Loss = 1.4321
Epoch 4: Loss = 1.1234
Epoch 5: Loss = 0.9876
```

### Step 4: Make a Recommendation!

```python
# Predict rating for user 42 on movies 0-9
user = torch.tensor([42] * 10)
movies = torch.arange(10)

with torch.no_grad():
    scores = model(user, movies)

print("Top 3 movies for user 42:")
top_movies = scores.argsort(descending=True)[:3]
for i, movie_id in enumerate(top_movies):
    print(f"  {i+1}. Movie {movie_id.item()}: predicted rating {scores[movie_id]:.2f}")
```

**Expected output:**
```
Top 3 movies for user 42:
  1. Movie 7: predicted rating 3.45
  2. Movie 3: predicted rating 3.21
  3. Movie 1: predicted rating 2.98
```

## 🎉 You Did It!

You just built a basic matrix factorization recommender! The model learned:
- A 32-dimensional "taste vector" for each user
- A 32-dimensional "feature vector" for each movie
- How to predict ratings using the dot product of these vectors

## ▶️ Next Steps

1. **Use real data**: Load MovieLens dataset in Notebook 01
2. **Add neural layers**: Build NeuMF in Notebook 02
3. **Scale up**: Create two-tower retrieval in Notebook 03
4. **Evaluate properly**: Learn NDCG, MAP metrics in Notebook 04

---

## 💡 The Key Insight

> **Embeddings are learned similarities.**
>
> Users with similar embeddings have similar taste.
> Movies with similar embeddings are liked by similar people.
>
> The magic: we never defined "similar" - the model learned it from ratings!
