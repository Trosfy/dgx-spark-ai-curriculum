# Module 1.6: Classical ML Foundations - Troubleshooting Guide

Common issues and solutions for the Classical ML module.

---

## RAPIDS / cuML Issues

### `ImportError: No module named cuml`

**Symptom:** Cannot import cuML or RAPIDS libraries.

**Cause:** RAPIDS is not installed in your current environment.

**Solution:** Use the NGC RAPIDS container:
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/rapidsai/base:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Important:** On DGX Spark (ARM64/aarch64), always use NGC containers. Never attempt `pip install cuml`.

---

### cuML results differ from sklearn

**Symptom:** cuML model predictions slightly differ from sklearn.

**Cause:** Minor numerical differences due to GPU floating-point operations.

**Solution:** This is expected behavior. Differences are typically:
- < 0.1% for most algorithms
- Due to parallel reduction ordering
- Not significant for practical use

To verify consistency:
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from cuml.ensemble import RandomForestClassifier as CumlRF

# Train both
sklearn_model = SklearnRF(n_estimators=100, random_state=42)
cuml_model = CumlRF(n_estimators=100)

sklearn_model.fit(X_train, y_train)
cuml_model.fit(X_train, y_train)

# Compare predictions
sklearn_pred = sklearn_model.predict(X_test)
cuml_pred = cuml_model.predict(X_test).to_numpy()

agreement = (sklearn_pred == cuml_pred).mean()
print(f"Prediction agreement: {agreement:.2%}")  # Should be > 99%
```

---

### cuDF type errors

**Symptom:** `TypeError` when using cuDF DataFrames with cuML.

**Cause:** cuML often requires `float32` dtype.

**Solution:** Ensure data is float32:
```python
import cudf
import numpy as np

# From pandas
gdf = cudf.DataFrame.from_pandas(pdf)
gdf = gdf.astype('float32')

# From numpy
X = X.astype(np.float32)
gdf = cudf.DataFrame(X)
```

---

## XGBoost Issues

### XGBoost GPU Out of Memory (OOM)

**Symptom:** `CUDA out of memory` error when training XGBoost.

**Cause:** Model too complex or dataset too large for GPU memory.

**Solutions:**
1. Reduce tree depth:
   ```python
   model = xgb.XGBClassifier(
       max_depth=4,  # Reduce from 6
       device='cuda'
   )
   ```

2. Use histogram-based training (more memory efficient):
   ```python
   model = xgb.XGBClassifier(
       tree_method='hist',
       device='cuda'
   )
   ```

3. Reduce number of trees and use early stopping:
   ```python
   model = xgb.XGBClassifier(
       n_estimators=1000,
       early_stopping_rounds=10,
       device='cuda'
   )
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
   ```

---

## Optuna Issues

### Optuna optimization is slow

**Symptom:** Hyperparameter tuning takes too long.

**Cause:** Each trial runs full training, and trials are sequential.

**Solutions:**
1. Use parallel trials with `n_jobs`:
   ```python
   study.optimize(objective, n_trials=100, n_jobs=-1)
   ```

2. Reduce CV folds during tuning:
   ```python
   cv_scores = cross_val_score(model, X, y, cv=3)  # Use 3 instead of 5
   ```

3. Use pruning to stop bad trials early:
   ```python
   from optuna.integration import XGBoostPruningCallback

   def objective(trial):
       # ... define params ...
       pruning_callback = XGBoostPruningCallback(trial, 'validation-rmse')
       # Use with XGBoost native API
   ```

4. Use smaller dataset for initial exploration:
   ```python
   # Use 20% of data for hyperparameter search
   X_tune, _, y_tune, _ = train_test_split(X, y, train_size=0.2)
   ```

---

## PyTorch / NGC Container Issues

### `pip install torch` fails on ARM64

**Symptom:** Cannot install PyTorch via pip on DGX Spark.

**Cause:** DGX Spark uses ARM64/aarch64 architecture, which requires special PyTorch builds.

**Solution:** Never use `pip install torch` on DGX Spark. Always use the NGC PyTorch container:
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

### `torch.cuda.is_available()` returns False

**Symptom:** GPU not detected in PyTorch.

**Cause:** Not using NGC container or container not started with `--gpus all`.

**Solutions:**
1. Ensure you're using the NGC container (not pip-installed PyTorch)
2. Verify container started with `--gpus all`:
   ```bash
   docker run --gpus all -it nvcr.io/nvidia/pytorch:25.11-py3 bash
   ```
3. Check inside container:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show GPU name
   ```

---

## Memory Management Issues

### GPU memory not released after training

**Symptom:** GPU memory usage keeps increasing across experiments.

**Cause:** Python garbage collection doesn't automatically free GPU memory.

**Solution:** Explicitly clean up after each experiment:
```python
import gc
import torch

# After training
del model
del X_train, X_test

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Force garbage collection
gc.collect()

# For cuML/RAPIDS
try:
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
except ImportError:
    pass
```

---

## Data Loading Issues

### Dataset too large for memory

**Symptom:** MemoryError when loading large datasets.

**Solutions:**
1. Use chunked loading:
   ```python
   chunks = pd.read_csv('large_file.csv', chunksize=100000)
   for chunk in chunks:
       # Process each chunk
       pass
   ```

2. Use float32 instead of float64:
   ```python
   X = X.astype(np.float32)  # Halves memory usage
   ```

3. Load directly to GPU with cuDF:
   ```python
   import cudf
   gdf = cudf.read_csv('large_file.csv')  # Faster than pandas → cudf
   ```

---

## ❓ Frequently Asked Questions

**Q: When should I use XGBoost vs a neural network?**

Use XGBoost (or tree-based methods) when:
- Working with tabular data (spreadsheet-like data)
- Dataset has < 100K samples
- Need interpretability (feature importance)
- Want fast training and iteration
- Features are heterogeneous (mix of types)

Use neural networks when:
- Working with images, audio, or text
- Dataset is very large (> 1M samples)
- Complex feature interactions exist
- Can leverage pre-trained models/embeddings

**Key insight:** On tabular data, XGBoost wins ~70% of the time. Always start with an XGBoost baseline!

---

**Q: Why does XGBoost often beat neural networks on tabular data?**

Several reasons:
1. **Axis-aligned splits**: Trees naturally handle "if income > $50K" decisions
2. **Heterogeneous features**: Age, income, location have different scales/meanings
3. **No preprocessing needed**: Works on raw data
4. **Built-in regularization**: L1/L2 prevents overfitting
5. **Efficient with small data**: Doesn't need millions of samples

Research paper: "Why do tree-based models still outperform deep learning on tabular data?" (2022)

---

**Q: What's the difference between XGBoost, LightGBM, and CatBoost?**

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Speed | Fast | Fastest | Medium |
| Memory | Medium | Lower | Higher |
| Categoricals | Encoding needed | Encoding needed | Native support |
| GPU Support | ✅ Full | ✅ Full | ✅ Full |
| DGX Spark | ✅ Full | ✅ Full | ✅ Full |

**Recommendation:** Start with XGBoost (most documentation), try LightGBM for speed, use CatBoost for categorical-heavy data.

---

**Q: What is RAPIDS cuML?**

RAPIDS cuML is a GPU-accelerated machine learning library that provides:
- Drop-in replacements for scikit-learn algorithms
- 10-100x speedups on large datasets
- Same API as scikit-learn
- Full ARM64 support on DGX Spark

```python
# CPU (sklearn)
from sklearn.ensemble import RandomForestClassifier
# GPU (cuML) - same API!
from cuml.ensemble import RandomForestClassifier
```

---

**Q: Why can't I pip install RAPIDS?**

On DGX Spark (ARM64/aarch64 architecture), RAPIDS requires specific builds. The easiest and recommended approach is to use the NGC container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/rapidsai/base:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

**Q: When does GPU acceleration help most?**

| Scenario | GPU Speedup |
|----------|-------------|
| K-Nearest Neighbors | 50-100x |
| K-Means Clustering | 50-100x |
| PCA/SVD | 50-100x |
| Random Forest | 10-50x |
| Logistic Regression | 5-20x |
| Small datasets (<10K) | Minimal |

**Rule of thumb:** GPU acceleration helps most with:
- Large datasets (> 100K rows)
- Many iterations (K-Means, hyperparameter tuning)
- Distance calculations (KNN)

---

**Q: How many trials should I run with Optuna?**

Guidelines:
- Quick exploration: 50 trials
- Standard tuning: 100 trials
- Thorough search: 200-500 trials

Diminishing returns usually after 100 trials. Use cross-validation to avoid overfitting to validation set.

---

**Q: Should I use GridSearch or Optuna?**

| Method | Pros | Cons |
|--------|------|------|
| GridSearchCV | Simple, exhaustive | Slow with many params |
| RandomSearch | Faster, good coverage | Not adaptive |
| Optuna (Bayesian) | Learns from trials, efficient | More setup |

**Recommendation:** Use Optuna for 3+ hyperparameters, GridSearch for 1-2.

---

**Q: What hyperparameters matter most for XGBoost?**

In order of importance (typically):
1. `learning_rate` (eta): 0.01-0.3
2. `max_depth`: 3-10
3. `n_estimators`: 100-1000 (with early stopping)
4. `min_child_weight`: 1-10
5. `subsample` / `colsample_bytree`: 0.5-1.0

Use log scale for `learning_rate` and regularization parameters.

---

**Q: Why use NGC containers instead of pip install?**

DGX Spark uses ARM64/aarch64 architecture. Key reasons:
1. PyTorch ARM64 wheels require NGC container
2. RAPIDS cuML requires NGC container
3. Pre-optimized for DGX Spark hardware
4. Guaranteed compatibility

Never use `pip install torch` on DGX Spark!

---

**Q: How much can I fit in DGX Spark's 128GB unified memory?**

| Model Size | Precision | Fits? |
|------------|-----------|-------|
| XGBoost (any tabular) | N/A | ✅ Yes |
| 70B LLM | FP16 | ❌ No (needs 140GB) |
| 70B LLM | INT4 | ✅ Yes (~35GB) |
| Dataset 10M × 100 | float32 | ✅ Yes (~4GB) |

For classical ML, memory is rarely a constraint on DGX Spark.

---

**Q: What's unified memory and why does it matter?**

DGX Spark's 128GB unified memory is shared between CPU and GPU. Benefits:
- No explicit CPU↔GPU data transfers
- Can work with larger datasets than traditional GPU memory
- Simplified programming model
- Automatic memory management

---

**Q: Do I need to complete all 4 labs?**

Recommended order:
1. **Lab 1.6.1** (Required): XGBoost vs NN comparison - fundamental insight
2. **Lab 1.6.2** (Recommended): Optuna tuning - practical skill
3. **Lab 1.6.3** (Recommended): RAPIDS - DGX Spark specific
4. **Lab 1.6.4** (Optional): Baseline framework - reusable code

If short on time, complete Labs 1 and 3.

---

**Q: Can I use the BaselineExperiment class in future projects?**

Yes! The `scripts/baseline_utils.py` module is designed for reuse:

```python
from scripts.baseline_utils import BaselineExperiment

exp = BaselineExperiment(X, y, task='classification')
exp.add_default_models()
exp.run()
exp.report()
```

Copy the `scripts/` folder to your projects.

---

## Still Having Issues?

1. Check DGX Spark User Guide: https://docs.nvidia.com/dgx/dgx-spark/
2. RAPIDS documentation: https://docs.rapids.ai/
3. XGBoost documentation: https://xgboost.readthedocs.io/
4. Post questions on the course forum

---

*Troubleshooting guide for Module 1.6 - Classical ML Foundations*
