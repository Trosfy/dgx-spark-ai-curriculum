# Module 1.6: Classical ML Foundations - FAQ

Frequently asked questions about classical ML and this module.

---

## General Questions

### Q: When should I use XGBoost vs a neural network?

**A:** Use XGBoost (or tree-based methods) when:
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

### Q: Why does XGBoost often beat neural networks on tabular data?

**A:** Several reasons:
1. **Axis-aligned splits**: Trees naturally handle "if income > $50K" decisions
2. **Heterogeneous features**: Age, income, location have different scales/meanings
3. **No preprocessing needed**: Works on raw data
4. **Built-in regularization**: L1/L2 prevents overfitting
5. **Efficient with small data**: Doesn't need millions of samples

Research paper: "Why do tree-based models still outperform deep learning on tabular data?" (2022)

---

### Q: What's the difference between XGBoost, LightGBM, and CatBoost?

**A:**
| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Speed | Fast | Fastest | Medium |
| Memory | Medium | Lower | Higher |
| Categoricals | Encoding needed | Encoding needed | Native support |
| GPU Support | ✅ Full | ✅ Full | ✅ Full |
| DGX Spark | ✅ Full | ✅ Full | ✅ Full |

**Recommendation:** Start with XGBoost (most documentation), try LightGBM for speed, use CatBoost for categorical-heavy data.

---

## RAPIDS Questions

### Q: What is RAPIDS cuML?

**A:** RAPIDS cuML is a GPU-accelerated machine learning library that provides:
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

### Q: Why can't I pip install RAPIDS?

**A:** On DGX Spark (ARM64/aarch64 architecture), RAPIDS requires specific builds. The easiest and recommended approach is to use the NGC container:

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/rapidsai/base:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

---

### Q: When does GPU acceleration help most?

**A:**
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

## Hyperparameter Tuning Questions

### Q: How many trials should I run with Optuna?

**A:** Guidelines:
- Quick exploration: 50 trials
- Standard tuning: 100 trials
- Thorough search: 200-500 trials

Diminishing returns usually after 100 trials. Use cross-validation to avoid overfitting to validation set.

---

### Q: Should I use GridSearch or Optuna?

**A:**
| Method | Pros | Cons |
|--------|------|------|
| GridSearchCV | Simple, exhaustive | Slow with many params |
| RandomSearch | Faster, good coverage | Not adaptive |
| Optuna (Bayesian) | Learns from trials, efficient | More setup |

**Recommendation:** Use Optuna for 3+ hyperparameters, GridSearch for 1-2.

---

### Q: What hyperparameters matter most for XGBoost?

**A:** In order of importance (typically):
1. `learning_rate` (eta): 0.01-0.3
2. `max_depth`: 3-10
3. `n_estimators`: 100-1000 (with early stopping)
4. `min_child_weight`: 1-10
5. `subsample` / `colsample_bytree`: 0.5-1.0

Use log scale for `learning_rate` and regularization parameters.

---

## DGX Spark Questions

### Q: Why use NGC containers instead of pip install?

**A:** DGX Spark uses ARM64/aarch64 architecture. Key reasons:
1. PyTorch ARM64 wheels require NGC container
2. RAPIDS cuML requires NGC container
3. Pre-optimized for DGX Spark hardware
4. Guaranteed compatibility

Never use `pip install torch` on DGX Spark!

---

### Q: How much can I fit in DGX Spark's 128GB unified memory?

**A:**
| Model Size | Precision | Fits? |
|------------|-----------|-------|
| XGBoost (any tabular) | N/A | ✅ Yes |
| 70B LLM | FP16 | ❌ No (needs 140GB) |
| 70B LLM | INT4 | ✅ Yes (~35GB) |
| Dataset 10M × 100 | float32 | ✅ Yes (~4GB) |

For classical ML, memory is rarely a constraint on DGX Spark.

---

### Q: What's unified memory and why does it matter?

**A:** DGX Spark's 128GB unified memory is shared between CPU and GPU. Benefits:
- No explicit CPU↔GPU data transfers
- Can work with larger datasets than traditional GPU memory
- Simplified programming model
- Automatic memory management

---

## Module-Specific Questions

### Q: Do I need to complete all 4 labs?

**A:** Recommended order:
1. **Lab 1.6.1** (Required): XGBoost vs NN comparison - fundamental insight
2. **Lab 1.6.2** (Recommended): Optuna tuning - practical skill
3. **Lab 1.6.3** (Recommended): RAPIDS - DGX Spark specific
4. **Lab 1.6.4** (Optional): Baseline framework - reusable code

If short on time, complete Labs 1 and 3.

---

### Q: Can I use the BaselineExperiment class in future projects?

**A:** Yes! The `scripts/baseline_utils.py` module is designed for reuse:

```python
from scripts.baseline_utils import BaselineExperiment

exp = BaselineExperiment(X, y, task='classification')
exp.add_default_models()
exp.run()
exp.report()
```

Copy the `scripts/` folder to your projects.

---

*FAQ for Module 1.6 - Classical ML Foundations*
