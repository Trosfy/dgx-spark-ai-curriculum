# Module 1.6: Classical ML Foundations - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Do
Train an XGBoost model faster than a neural network—and often better on tabular data!

## ✅ Before You Start
- [ ] NGC PyTorch container running
- [ ] Basic understanding of train/test split

## 🚀 Let's Go!

### Step 1: Load Dataset
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.shape[1]}")
```

### Step 2: Train XGBoost
```python
start = time.time()

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    device='cuda',  # Use GPU on DGX Spark!
    random_state=42
)
model.fit(X_train, y_train)

train_time = time.time() - start
print(f"Training time: {train_time:.2f}s")
```

### Step 3: Evaluate
```python
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

**Expected output:**
```
Training samples: 455
Features: 30
Training time: 0.15s
Accuracy: 0.9737
```

### Step 4: Feature Importance
```python
import numpy as np

# Top 5 most important features
importance = model.feature_importances_
top_5 = np.argsort(importance)[-5:][::-1]

print("\nTop 5 Features:")
for idx in top_5:
    print(f"  {data.feature_names[idx]}: {importance[idx]:.3f}")
```

## 🎉 You Did It!

You just:
- ✅ Trained XGBoost with GPU acceleration
- ✅ Got 97%+ accuracy in <1 second
- ✅ Identified most important features
- ✅ Saw why XGBoost is the go-to for tabular data

In the full module, you'll:
- Compare XGBoost vs Neural Networks
- Tune hyperparameters with Optuna
- Use RAPIDS cuML for 10-100x speedup
- Build a reusable baseline framework

## ▶️ Next Steps
1. **Understand when to use what**: Read [STUDY_GUIDE.md](./STUDY_GUIDE.md)
2. **See XGBoost patterns**: Check [QUICK_REFERENCE.md](./QUICK_REFERENCE.md)
3. **Start Lab 1**: Open `notebooks/lab-1.6.1-tabular-challenge.ipynb`
4. **Having issues?**: See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
