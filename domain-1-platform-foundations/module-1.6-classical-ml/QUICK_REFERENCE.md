# Module 1.6: Classical ML Foundations - Quick Reference

## 🌲 XGBoost

### Basic Usage
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Classification
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    device='cuda',  # GPU on DGX Spark
    random_state=42
)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# Regression
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    device='cuda',
    random_state=42
)
model.fit(X_train, y_train)
rmse = np.sqrt(np.mean((model.predict(X_test) - y_test)**2))
```

### Best Hyperparameters to Tune
```python
params = {
    'n_estimators': 100-1000,      # More = better but slower
    'max_depth': 3-10,              # Deeper = more complex
    'learning_rate': 0.01-0.3,      # Lower = more trees needed
    'min_child_weight': 1-10,       # Higher = more conservative
    'subsample': 0.5-1.0,           # Row sampling
    'colsample_bytree': 0.5-1.0,    # Column sampling
    'gamma': 0-5,                   # Regularization
    'reg_alpha': 0-1,               # L1 regularization
    'reg_lambda': 0-1,              # L2 regularization
}
```

### Early Stopping
```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    early_stopping_rounds=10,
    device='cuda'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"Best iteration: {model.best_iteration}")
```

## 🔍 Optuna Hyperparameter Tuning

### Basic Pattern
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    model = xgb.XGBClassifier(**params, device='cuda', random_state=42)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Visualization
```python
# Optimization history
optuna.visualization.plot_optimization_history(study)

# Parameter importance
optuna.visualization.plot_param_importances(study)

# Parallel coordinate plot
optuna.visualization.plot_parallel_coordinate(study)
```

## 🚀 RAPIDS cuML (GPU-Accelerated)

### Drop-in Replacement for sklearn
```python
# CPU (sklearn)
from sklearn.ensemble import RandomForestClassifier

# GPU (cuML)
from cuml.ensemble import RandomForestClassifier

# Same API!
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)
```

### Common cuML Models
```python
from cuml.linear_model import LogisticRegression
from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from cuml.svm import SVC
from cuml.cluster import KMeans
from cuml.decomposition import PCA

# All have same API as sklearn equivalents
```

### cuDF for DataFrames
```python
import cudf
import cuml

# Load to GPU
df = cudf.read_csv('large_data.csv')

# cuML works directly with cuDF
X = df[feature_cols]
y = df[target_col]

model = cuml.ensemble.RandomForestClassifier(n_estimators=100)
model.fit(X, y)
```

### Performance Comparison
```python
import time
from sklearn.ensemble import RandomForestClassifier as sklearn_RF
from cuml.ensemble import RandomForestClassifier as cuml_RF

# sklearn (CPU)
start = time.time()
sklearn_rf = sklearn_RF(n_estimators=100)
sklearn_rf.fit(X_train, y_train)
sklearn_time = time.time() - start

# cuML (GPU)
start = time.time()
cuml_rf = cuml_RF(n_estimators=100)
cuml_rf.fit(X_train, y_train)
cuml_time = time.time() - start

print(f"sklearn: {sklearn_time:.2f}s")
print(f"cuML: {cuml_time:.2f}s")
print(f"Speedup: {sklearn_time/cuml_time:.1f}x")
```

## 📊 Model Comparison Framework

```python
from sklearn.model_selection import cross_val_score
import time

class BaselineFramework:
    def __init__(self, X, y, cv=5):
        self.X = X
        self.y = y
        self.cv = cv
        self.results = {}

    def evaluate(self, name, model):
        start = time.time()
        scores = cross_val_score(model, self.X, self.y, cv=self.cv)
        elapsed = time.time() - start

        self.results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'time': elapsed
        }
        return scores

    def summary(self):
        print(f"{'Model':<20} {'Accuracy':>12} {'Time (s)':>10}")
        print("-" * 45)
        for name, r in sorted(self.results.items(), key=lambda x: -x[1]['mean']):
            print(f"{name:<20} {r['mean']:.4f}±{r['std']:.3f} {r['time']:>10.2f}")
```

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: cuml` | Use RAPIDS container: `nvcr.io/nvidia/rapidsai/base:25.11-py3` |
| XGBoost GPU OOM | Reduce `max_depth` or use `tree_method='hist'` |
| cuML dtype errors | Ensure float32: `X.astype(np.float32)` |
| Optuna slow | Use `n_jobs=-1` for parallel trials |

→ For detailed solutions, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

## 📊 Quick Comparisons

### XGBoost vs LightGBM vs CatBoost
| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Speed | Fast | Faster | Medium |
| Memory | Medium | Lower | Higher |
| Categoricals | Encoding needed | Encoding needed | Native support |
| GPU | ✅ | ✅ | ✅ |
| DGX Spark | ✅ Full | ✅ Full | ✅ Full |

### When to Use Neural Networks
| Use Classical ML | Use Neural Networks |
|------------------|---------------------|
| Tabular data | Images, audio, text |
| <100K samples | >100K samples |
| Need interpretability | Complex patterns |
| Quick baseline | State-of-the-art |

## 🔗 Quick Links
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [RAPIDS cuML Docs](https://docs.rapids.ai/api/cuml/stable/)
- [Optuna Docs](https://optuna.readthedocs.io/)
