# Module 1.6: Classical ML Foundations - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Train** and evaluate tree-based models (Random Forest, XGBoost)
2. **Explain** the bias-variance tradeoff in classical ML context
3. **Perform** hyperparameter tuning with cross-validation
4. **Accelerate** classical ML with RAPIDS cuML on DGX Spark

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | Tabular Data Challenge | XGBoost vs NN | ~2 hr | Know when to use each |
| 2 | Hyperparameter Optimization | Optuna tuning | ~2 hr | 100 trials with visualization |
| 3 | RAPIDS Acceleration | cuML on GPU | ~2 hr | 10-100x speedup |
| 4 | Baseline Comparison | Framework | ~2 hr | Reusable comparison tool |

**Total time**: ~8 hours

## 🔑 Core Concepts

### Tree-Based Methods
**What**: Decision trees, Random Forests, Gradient Boosting (XGBoost/LightGBM).
**Why it matters**: Often outperform neural networks on tabular data. Interpretable and fast.
**First appears in**: Lab 1

### Bias-Variance Tradeoff
**What**: Balance between underfitting (high bias) and overfitting (high variance).
**Why it matters**: Fundamental to model selection and hyperparameter tuning.
**First appears in**: Lab 1

### Gradient Boosting
**What**: Build models sequentially, each correcting the previous one's errors.
**Why it matters**: XGBoost dominates Kaggle competitions on tabular data.
**First appears in**: Lab 1, Lab 2

### RAPIDS cuML
**What**: GPU-accelerated scikit-learn-compatible ML library.
**Why it matters**: 10-100x faster training on DGX Spark's unified memory.
**First appears in**: Lab 3

## 🔗 How This Module Connects

```
    Module 1.5              Module 1.6                Module 1.7
    ───────────────────────────────────────────────────────────────
    Neural Networks    ──►   Classical ML        ──►   Capstone

    Deep learning            When NOT to use DL        Combine everything
    Complex models           Fast baselines            Build autograd
    GPU training             XGBoost supremacy         MNIST example
```

**Builds on**:
- Module 1.5: Comparison point for neural networks

**Prepares for**:
- **Module 1.7**: Classical ML baselines for MicroGrad+ testing
- **All future modules**: Always start with an XGBoost baseline!

## 📊 When to Use What

### Decision Guide
```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION GUIDE                            │
├──────────────────┬──────────────────────────────────────────┤
│ Data Type        │ Recommendation                           │
├──────────────────┼──────────────────────────────────────────┤
│ Tabular (<100K)  │ XGBoost first, neural net if needed     │
│ Tabular (>1M)    │ XGBoost or LightGBM with GPU            │
│ Images           │ Deep learning (CNNs, ViT)               │
│ Text             │ Transformers (BERT, LLMs)               │
│ Time series      │ Try both, XGBoost often wins            │
│ Need explainability │ Trees, linear models                 │
│ Many features    │ Random Forest, Lasso                    │
│ Few samples      │ Classical ML (less overfit risk)        │
└──────────────────┴──────────────────────────────────────────┘
```

### Method Comparison
| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| XGBoost | Fast, accurate, handles missing values | Not for images/text |
| Random Forest | Robust, parallel | Slower than boosting |
| Logistic Regression | Interpretable, fast | Linear only |
| Neural Network | Universal, flexible | Needs lots of data |

## 📖 Recommended Approach

**Standard path** (8 hours):
1. Lab 1: Compare XGBoost and neural networks
2. Lab 2: Learn Optuna for hyperparameter search
3. Lab 3: Experience RAPIDS GPU acceleration
4. Lab 4: Build reusable baseline framework

**Quick path** (if experienced with sklearn, 4-5 hours):
1. Focus on Lab 1 comparison insights
2. Skim Lab 2, focus on Optuna patterns
3. Complete Lab 3 RAPIDS (DGX Spark specific!)
4. Quick pass on Lab 4 framework

## 📋 Before You Start
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute XGBoost demo
→ See [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for XGBoost and cuML patterns
→ See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common questions
→ Ensure NGC container has XGBoost installed
