# Module 1.6: Classical ML Foundations

**Domain:** 1 - Platform Foundations
**Duration:** Week 6 (6-8 hours)
**Prerequisites:** Module 1.5 (Neural Network Fundamentals)
**Priority:** P2 Medium

---

## Overview

Before diving deeper into deep learning, it's essential to understand when classical ML algorithms outperform neural networks—and they often do! This module covers tree-based methods, linear models, and GPU-accelerated classical ML with RAPIDS cuML on your DGX Spark.

Classical ML provides fast baselines, interpretable models, and often wins on tabular data. Every deep learning experiment should start with an XGBoost baseline.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Apply classical ML algorithms as baselines for comparison
- ✅ Explain when classical ML outperforms deep learning
- ✅ Use scikit-learn and XGBoost effectively
- ✅ Accelerate classical ML with RAPIDS cuML on DGX Spark

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 1.6.1 | Train and evaluate tree-based models (Random Forest, XGBoost) | Apply |
| 1.6.2 | Explain bias-variance tradeoff in classical ML context | Understand |
| 1.6.3 | Perform hyperparameter tuning with cross-validation | Apply |
| 1.6.4 | Accelerate classical ML with RAPIDS cuML | Apply |

---

## Topics

### 1.6.1 Tree-Based Methods

- **Decision Trees**
  - Information gain and Gini impurity
  - Tree construction algorithm
  - Pruning strategies
  - Interpretability advantages

- **Random Forests**
  - Bagging (Bootstrap Aggregating)
  - Feature importance
  - Out-of-bag error estimation

- **Gradient Boosting**
  - XGBoost: The Kaggle champion
  - LightGBM: Faster training
  - CatBoost: Native categorical support
  - When to use each

### 1.6.2 Linear Models

- **Logistic Regression**
  - Maximum likelihood estimation
  - Regularization (L1/L2)
  - Feature coefficients as importance

- **Support Vector Machines**
  - Kernel trick intuition
  - RBF vs linear kernels
  - When SVMs still shine

- **Regularized Regression**
  - Ridge (L2): Weight shrinkage
  - Lasso (L1): Feature selection
  - ElasticNet: Best of both

### 1.6.3 Model Selection

- **Cross-Validation**
  - K-fold, stratified, time-series splits
  - Nested CV for hyperparameter tuning
  - Avoiding data leakage

- **Hyperparameter Tuning**
  - GridSearchCV basics
  - Optuna for Bayesian optimization
  - Early stopping strategies

- **When to Use Classical vs Deep Learning**
  - Tabular data: XGBoost often wins
  - Small datasets (<10K): Classical ML
  - Interpretability required: Trees/linear
  - Unstructured data: Deep learning

### 1.6.4 GPU Acceleration with RAPIDS

- **cuML Overview**
  - Drop-in scikit-learn replacement
  - 10-100x speedups on GPU
  - ARM64 support on DGX Spark

- **cuDF for Data Preprocessing**
  - GPU-accelerated DataFrames
  - Pandas compatibility mode
  - Memory management

- **Performance Comparison**
  - CPU vs GPU benchmarks
  - When GPU acceleration helps most

---

## Labs

### Lab 1.6.1: Tabular Data Challenge
**Time:** 2 hours

Compare XGBoost with a simple neural network on tabular data.

**Instructions:**
1. Open `notebooks/lab-1.6.1-tabular-challenge.ipynb`
2. Load a tabular dataset (housing prices, credit scoring, etc.)
3. Train XGBoost with default parameters
4. Train a simple MLP with similar training time
5. Compare accuracy, training time, interpretability
6. Document when each approach excels

**Deliverable:** Comparison notebook with analysis of when to use each approach

---

### Lab 1.6.2: Hyperparameter Optimization
**Time:** 2 hours

Use Optuna to tune XGBoost hyperparameters.

**Instructions:**
1. Open `notebooks/lab-1.6.2-hyperparameter-optimization.ipynb`
2. Define XGBoost parameter search space
3. Use Optuna with 100 trials
4. Visualize optimization history
5. Plot feature importance
6. Compare tuned vs default performance

**Deliverable:** Tuned model with visualization of optimization process

---

### Lab 1.6.3: RAPIDS Acceleration
**Time:** 2 hours

Port scikit-learn pipeline to cuML for GPU acceleration.

**Instructions:**
1. Open `notebooks/lab-1.6.3-rapids-acceleration.ipynb`
2. Load a large dataset (1M+ rows)
3. Train Random Forest with scikit-learn (time it)
4. Train same model with cuML (time it)
5. Verify results match
6. Benchmark on multiple algorithms

**Deliverable:** Benchmark showing 10-100x speedup with cuML

---

### Lab 1.6.4: Baseline Comparison Framework
**Time:** 2 hours

Create reusable framework for baseline experiments.

**Instructions:**
1. Open `notebooks/lab-1.6.4-baseline-framework.ipynb`
2. Create a `BaselineExperiment` class
3. Include: XGBoost, Random Forest, Logistic Regression
4. Add: cross-validation, timing, feature importance
5. Test on 3 different datasets
6. Generate comparison reports

**Deliverable:** Reusable baseline framework for future projects

---

## Guidance

### When to Use Classical ML

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION GUIDE                            │
├─────────────────────────────────────────────────────────────┤
│ Data Type        │ Recommendation                           │
├──────────────────┼──────────────────────────────────────────┤
│ Tabular (<100K)  │ XGBoost first, neural net if needed     │
│ Tabular (>1M)    │ XGBoost or LightGBM                     │
│ Images           │ Deep learning (CNNs, ViT)               │
│ Text             │ Transformers (BERT, LLMs)               │
│ Time series      │ Try both, XGBoost often wins            │
│ Need explain     │ Decision trees, linear models           │
└──────────────────┴──────────────────────────────────────────┘
```

### RAPIDS cuML on DGX Spark

```python
# Import cuML (GPU) instead of sklearn (CPU)
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LogisticRegression
import cudf

# Load data with cuDF (GPU DataFrame)
df = cudf.read_csv("large_dataset.csv")

# Train on GPU - same API as scikit-learn!
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)  # 10-100x faster than sklearn
```

### XGBoost Best Practices

```python
import xgboost as xgb

# Good default parameters for tabular data
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'tree_method': 'hist',  # Fast histogram-based
    'device': 'cuda',       # Use GPU on DGX Spark
    'early_stopping_rounds': 10
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### Optuna Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    model = xgb.XGBClassifier(**params, device='cuda')
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## Milestone Checklist

Use this checklist to track your progress:

- [ ] XGBoost model trained and evaluated
- [ ] Neural network comparison completed
- [ ] Optuna hyperparameter optimization done
- [ ] RAPIDS cuML benchmark showing speedup
- [ ] Baseline comparison framework created
- [ ] Can explain when classical ML beats deep learning
- [ ] Feature importance analysis documented

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `ImportError: No module named cuml` | Use NGC container with RAPIDS installed |
| cuML results differ from sklearn | Minor numerical differences are normal |
| XGBoost GPU OOM | Reduce `max_depth` or use `tree_method='hist'` |
| Optuna slow | Use `n_jobs=-1` for parallel trials |
| cuDF type errors | Ensure dtypes match (float32 preferred) |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save completed notebooks and scripts
3. ➡️ Proceed to [Module 1.7: Capstone — MicroGrad+](../module-1.7-capstone-micrograd/)

---

## Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Kaggle's XGBoost Tutorial](https://www.kaggle.com/learn/intro-to-xgboost)
