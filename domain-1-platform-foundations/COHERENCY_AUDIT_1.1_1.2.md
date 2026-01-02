# Coherency Audit Report: Modules 1.1 & 1.2

**Modules Reviewed:** 1.1 (DGX Spark Platform Mastery), 1.2 (Python for AI/ML)
**Files Analyzed:** 15+ notebooks, READMEs, and support scripts
**Inconsistencies Found:** 21
**Curriculum Version:** v2.0
**Audit Date:** 2026-01-02

---

## 📊 Summary

| Category | Issues Found |
|----------|--------------|
| Code ↔ Explanation | 2 |
| Code ↔ Table | 1 |
| Cross-File | 2 |
| Cross-Module | 2 |
| Terminology | 0 |
| Values | 2 |
| Testing Platform | 0 |
| Documentation | 2 |
| **Exercise Coherency (NEW)** | **10** |
| **TOTAL** | **21** |

---

## 🔴 HIGH IMPACT (Confuses Learners)

### Issue 1: GPU Memory Monitoring Returns `[N/A]` on Unified Memory

**Type:** Code ↔ Explanation Mismatch / Platform Incompatibility

**Location:**
- File: `module-1.1-dgx-spark-platform/labs/lab-1.1.1-system-exploration.ipynb`
- Section: Cell 4 (nvidia-smi query)
- Also affects: `lab-1.1.5-ollama-benchmarking.ipynb` (get_gpu_memory function)

**The Inconsistency:**

What's WRITTEN (lab-1.1.1):
```markdown
# Get detailed GPU information in a query format
!nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,...
```

What's SHOWN (actual output):
```
name, memory.total [MiB], memory.free [MiB], memory.used [MiB], ...
NVIDIA GB10, [N/A], [N/A], [N/A], 39, 3.74 W
```

**Why It's Confusing:**
- The curriculum teaches `nvidia-smi` as THE way to monitor GPU memory
- On DGX Spark with unified memory, nvidia-smi returns `[N/A]` for all memory values
- Students expect to see "128GB" but see "[N/A]" - appears broken
- Lab 1.1.5 benchmarks all show `Memory: 0.0 GB` which is misleading

**Fix:**

**Option A - Update explanations and add alternatives:**

In lab-1.1.1, add this explanation after the nvidia-smi cell:

```markdown
### 🔍 Understanding Unified Memory Reporting

On DGX Spark, `nvidia-smi` shows `[N/A]` for GPU memory because the 128GB
is **unified memory** shared between CPU and GPU - not dedicated VRAM.

**Alternative monitoring approaches:**

1. **System memory (includes GPU allocation):**
   ```bash
   free -h
   ```

2. **PyTorch CUDA memory (inside NGC container):**
   ```python
   import torch
   allocated = torch.cuda.memory_allocated() / 1e9
   reserved = torch.cuda.memory_reserved() / 1e9
   ```

3. **Ollama model memory (for LLM workloads):**
   ```bash
   ollama ps  # Shows memory per loaded model
   ```
```

**Option B - Update get_gpu_memory() in lab-1.1.5:**

```python
def get_model_memory_ollama() -> float:
    """Get memory usage of currently loaded Ollama model."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("models"):
                # Sum memory of all loaded models
                total_bytes = sum(m.get("size", 0) for m in data["models"])
                return total_bytes / (1024**3)  # Convert to GB
        return 0.0
    except:
        return 0.0

def get_system_memory_used() -> float:
    """Get system memory used (includes GPU on unified memory)."""
    try:
        result = subprocess.run(['free', '-b'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            return int(parts[2]) / (1024**3)  # used column in GB
    except:
        return 0.0
```

**Recommended:** Option A + Option B combined

---

### Issue 2: Benchmark Expected Results Missing MoE/Sparse Model Considerations

**Type:** Values Mismatch / Incomplete Reference Data

**Location:**
- File: `module-1.1-dgx-spark-platform/labs/lab-1.1.5-ollama-benchmarking.ipynb`
- Section: Cell 15 (EXPECTED_RESULTS comparison)
- Also: README.md expected results table

**The Inconsistency:**

What's DEFINED:
```python
EXPECTED_RESULTS = {
    "3b": {"prefill": 5000, "decode": 80, "memory": 3},
    "8b": {"prefill": 3000, "decode": 45, "memory": 6},
    "70b": {"prefill": 500, "decode": 15, "memory": 45},
}
```

What ACTUALLY happens with MoE models (from benchmark output):
```
devstral-2:123b:   Prefill: 1262.6 tok/s, Decode: 2.6 tok/s  (MoE)
deepseek-r1:70b:   Prefill: 57.0 tok/s,   Decode: 4.5 tok/s  (Dense with reasoning)
gpt-oss:120b:      Prefill: 2212.6 tok/s, Decode: 42.5 tok/s (MoE)
nemotron-3-nano:30b: Prefill: 338.1 tok/s, Decode: 62.8 tok/s (Dense)
```

**Why It's Confusing:**
- Comparison function tries to match "123b" → matches "3b" expected results (wrong!)
- Students see "⚠️ 3%" warnings for MoE models that are performing correctly
- Dense 70B vs MoE 120B have wildly different characteristics
- No guidance on expected MoE vs dense performance

**Fix:**

Update EXPECTED_RESULTS to include architecture awareness:

```python
EXPECTED_RESULTS = {
    # Dense models
    "3b": {"prefill": 5000, "decode": 80, "memory": 3, "type": "dense"},
    "8b": {"prefill": 3000, "decode": 45, "memory": 6, "type": "dense"},
    "32b": {"prefill": 1000, "decode": 25, "memory": 20, "type": "dense"},
    "70b": {"prefill": 500, "decode": 15, "memory": 45, "type": "dense"},

    # MoE/Sparse models (higher prefill, lower decode due to expert routing)
    "moe-24b": {"prefill": 2000, "decode": 14, "memory": 14, "type": "moe"},
    "moe-120b": {"prefill": 2000, "decode": 40, "memory": 65, "type": "moe"},

    # Reasoning models (much slower due to long chains of thought)
    "r1-70b": {"prefill": 60, "decode": 5, "memory": 45, "type": "reasoning"},
}

# Model name → expected key mapping
MODEL_TYPE_MAP = {
    "devstral": "moe",
    "magistral": "moe",
    "gpt-oss": "moe",
    "deepseek-r1": "reasoning",
    "qwen3": "dense",
    "nemotron": "dense",
}
```

Also add explanation:
```markdown
### Understanding Model Architecture Performance

| Architecture | Prefill | Decode | Notes |
|--------------|---------|--------|-------|
| Dense | High | Moderate | All parameters active |
| MoE (Mixture of Experts) | Very High | Lower | Expert routing overhead on decode |
| Reasoning (R1-style) | Low | Very Low | Long chain-of-thought generation |
```

---

### Issue 3: sklearn Used Before Being Taught

**Type:** Cross-Module Coherency / Prerequisites Mismatch

**Location:**
- File: `module-1.2-python-for-ai/labs/lab-1.2.2-dataset-preprocessing-pipeline.ipynb`
- Section: Cell 30 (train_test_split import)

**The Inconsistency:**

What's USED in lab:
```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

What's in PREREQUISITES (lab-1.2.2):
```markdown
- Completed: Lab 1.2.1 (NumPy Broadcasting)
- Knowledge of: Basic Python classes, NumPy basics
```

**Why It's Confusing:**
- Students haven't learned sklearn yet (it's not covered until later)
- No explanation of what train_test_split does or why
- Students may not have sklearn installed if not using NGC container

**Fix:**

**Option A - Add sklearn explanation inline:**

```markdown
### 🔧 Splitting Data: Train vs Test

Before we preprocess, we need to split our data. This prevents **data leakage** -
where information from test data influences our preprocessing.

We'll use scikit-learn's `train_test_split` function:

```python
from sklearn.model_selection import train_test_split

# Split: 80% training, 20% testing
# random_state ensures reproducibility
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

> **Note:** If sklearn is not installed, run: `pip install scikit-learn`
> In NGC containers, it's pre-installed.
```

**Option B - Provide a pure NumPy alternative:**

```python
def train_test_split_simple(df, test_size=0.2, random_state=42):
    """Simple train/test split using NumPy (no sklearn needed)."""
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)
    test_n = int(n * test_size)
    test_idx, train_idx = indices[:test_n], indices[test_n:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
```

**Recommended:** Option A (explain sklearn since it's essential for ML)

---

### Issue 4: `np.linalg.norm()` Used in Exercise Without Prior Teaching

**Type:** Exercise Coherency / Untaught Function in Solution

**Location:**
- File: `module-1.2-python-for-ai/labs/lab-1.2.1-numpy-broadcasting-lab.ipynb`
- Section: Exercise 2 (Cells 23-24) - Cosine Similarity

**The Inconsistency:**

What's ASKED in Exercise 2:
```markdown
**Task:** Implement cosine similarity between all pairs of vectors using broadcasting.

Cosine similarity formula:
$$\text{cosine\_sim}(a, b) = \frac{a \cdot b}{\|a\| \cdot \|b\|}$$
```

What's in the HINT:
```python
# 1. First normalize each embedding to unit length
# 2. Then cosine similarity is just the dot product!
# 3. Use `embeddings @ embeddings.T` after normalization
```

What's MISSING:
- **How to compute the norm** `||a||` is never taught
- Students need `np.linalg.norm()` or `np.sqrt(np.sum(x**2, axis=1))`
- Neither approach is shown before this exercise

**Why It's Confusing:**
- The formula shows `||a||` (norm) but doesn't explain how to compute it
- Hint says "normalize" but normalization isn't taught
- Students must discover `np.linalg.norm()` on their own

**Fix:**

Add a teaching cell BEFORE Exercise 2:

```python
# Vector Norms and Normalization
# ==============================
# The L2 norm (Euclidean length) of a vector is computed with np.linalg.norm()

v = np.array([3, 4])
print(f"Vector: {v}")
print(f"L2 Norm: {np.linalg.norm(v)}")  # sqrt(3² + 4²) = 5

# Normalizing = dividing by norm to get unit length
v_normalized = v / np.linalg.norm(v)
print(f"Normalized: {v_normalized}")
print(f"Length after normalizing: {np.linalg.norm(v_normalized)}")  # 1.0

# For a matrix of vectors (normalize each row):
vectors = np.random.randn(5, 3)
norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # keepdims for broadcasting!
normalized_vectors = vectors / norms
```

---

### Issue 5: ReLU Activation Used in Challenge Without Definition

**Type:** Exercise Coherency / Undefined Function in Challenge

**Location:**
- File: `module-1.2-python-for-ai/labs/lab-1.2.1-numpy-broadcasting-lab.ipynb`
- Section: Challenge (Cells 39-40) - Mini Neural Network

**The Inconsistency:**

What's ASKED in Challenge:
```markdown
**Implement a mini neural network forward pass using only NumPy broadcasting!**

Create a 2-layer network:
1. Input: (batch_size, 784) - flattened MNIST images
2. Hidden: (784, 256) weights + (256,) bias with ReLU activation
3. Output: (256, 10) weights + (10,) bias with softmax
```

What's in the TEMPLATE:
```python
def forward(x, w1, b1, w2, b2):
    # TODO: Implement using broadcasting
    # hidden = relu(x @ w1 + b1)  # <-- What is relu()?
    # output = softmax(hidden @ w2 + b2)
    pass
```

What's MISSING:
- `relu()` function is never defined anywhere
- Students see `relu()` but don't know if it's a NumPy function or custom
- No explanation of what ReLU even is

**Why It's Confusing:**
- Challenge casually uses `relu()` as if students know it
- ReLU is covered in Module 1.5 (Neural Networks) - future module!
- Students may search for `np.relu` which doesn't exist

**Fix:**

Add activation functions cell BEFORE the Challenge:

```python
# Activation Functions for Neural Networks
# ========================================
# Activation functions introduce non-linearity. Here are the most common:

def relu(x):
    """Rectified Linear Unit: max(0, x)"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x)), outputs 0-1"""
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    """Softmax: converts logits to probabilities"""
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

# Example:
x = np.array([-2, -1, 0, 1, 2])
print(f"Input:   {x}")
print(f"ReLU:    {relu(x)}")      # [-2,-1,0,1,2] → [0,0,0,1,2]
print(f"Sigmoid: {sigmoid(x).round(3)}")
```

---

### Issue 6: `df.groupby()` and `transform()` Used in Exercise Without Teaching

**Type:** Exercise Coherency / Untaught Pandas Functions

**Location:**
- File: `module-1.2-python-for-ai/labs/lab-1.2.2-dataset-preprocessing-pipeline.ipynb`
- Section: Exercise 1 (Cells 15-16) - Group-Based Imputation

**The Inconsistency:**

What's ASKED:
```markdown
**Task:** Implement group-based imputation.
Instead of using the global median for `income`, impute using the median income
*for each education level*.
```

What's in the HINT:
```python
df['income'] = df.groupby('education')['income'].transform(
    lambda x: x.fillna(x.median())
)
```

What's MISSING:
- `groupby()` is never taught in this lab or prior labs
- `transform()` is never taught
- Lambda functions with transform pattern is advanced Pandas

**Why It's Confusing:**
- Students are expected to use functions they've never seen
- The hint IS the solution - but students can't understand it
- Copy-pasting without understanding defeats the learning objective

**Fix:**

Add a teaching cell BEFORE Exercise 1:

```python
# Pandas GroupBy Operations
# =========================
# groupby() lets you split data by categories and apply operations per group

# Basic groupby - aggregate statistics per group
print(df.groupby('education')['income'].median())

# transform() - returns values aligned with original DataFrame
# Unlike agg(), transform returns same-length output
df['income_group_median'] = df.groupby('education')['income'].transform('median')

# Powerful pattern: Fill missing with group median
df['income'] = df.groupby('education')['income'].transform(
    lambda x: x.fillna(x.median())  # For each education group, fill NaN with that group's median
)
```

---

### Issue 7: `np.argpartition()` and `np.take_along_axis()` in Exercise Solution

**Type:** Exercise Coherency / Advanced Functions in Solution

**Location:**
- File: `module-1.2-python-for-ai/labs/lab-1.2.5-profiling-exercise.ipynb`
- Section: Exercise Solution (Cell 23) - KNN Optimization

**The Inconsistency:**

What's in the SOLUTION HINT:
```python
# Use argpartition for efficiency (doesn't fully sort)
# This is O(n) instead of O(n log n) for full sort
indices = np.argpartition(sq_distances, k, axis=1)[:, :k]

# Get the actual distances for these indices
row_indices = np.arange(len(query_points))[:, np.newaxis]
top_k_sq_distances = sq_distances[row_indices, indices]

# Sort within the k (argpartition doesn't sort)
sorted_within_k = np.argsort(top_k_sq_distances, axis=1)
final_indices = np.take_along_axis(indices, sorted_within_k, axis=1)
```

What's MISSING:
- `np.argpartition()` - never taught
- `np.take_along_axis()` - never taught
- Advanced NumPy indexing patterns - not covered

**Why It's Confusing:**
- Solution uses functions students have never seen
- These are advanced NumPy functions most users don't know
- Students can't implement the optimization without discovering these

**Fix:**

Add a teaching cell BEFORE the exercise:

```python
# Advanced NumPy: Efficient Top-K Selection
# =========================================

# argpartition() - partial sort, O(n) instead of O(n log n)
# Returns indices that would partition the array
arr = np.array([5, 2, 8, 1, 9, 3])
k = 3
# Get indices of k smallest elements (not sorted!)
top_k_idx = np.argpartition(arr, k)[:k]
print(f"Top {k} indices: {top_k_idx}")
print(f"Top {k} values: {arr[top_k_idx]}")

# take_along_axis() - advanced indexing along an axis
# Useful when you have indices from argpartition/argsort
arr_2d = np.array([[5, 2, 8], [1, 9, 3]])
sorted_idx = np.argsort(arr_2d, axis=1)
sorted_arr = np.take_along_axis(arr_2d, sorted_idx, axis=1)
print(f"Sorted: {sorted_arr}")
```

---

## 🟡 MEDIUM IMPACT (Inconsistent but Not Blocking)

### Issue M1: Docker Command Port Mapping Inconsistency

**Location:** `lab-1.1.2-memory-architecture-lab.ipynb` cell 1

**Inconsistency:**
- Lab 1.1.2 shows: `docker run ... -p 8888:8888 ...`
- README Guidance section shows: `docker run ... bash` (no port mapping)
- Both are correct but for different use cases

**Fix:** Add clarification:
```markdown
> **Note:** The `-p 8888:8888` flag is only needed when running Jupyter Lab inside
> the container. For interactive bash sessions, you can omit the port mapping.
```
✅ Already present in lab-1.1.2 - just ensure consistency in other labs.

---

### Issue M2: Preprocessor Class Imported Without Prior Building

**Location:** `lab-1.2.2-dataset-preprocessing-pipeline.ipynb` cells 28-34

**Inconsistency:**
- Lab says "Now let's use our production-ready Preprocessor class"
- But students haven't built it - they're asked to import a pre-made one
- The learning objective says "Build a reusable Preprocessor class"

**Why It's Confusing:**
- Students expect to BUILD the class, not just USE it
- The class is imported from `scripts/preprocessing_pipeline.py`
- Mismatch between learning objective and actual activity

**Fix Options:**

**Option A - Change learning objective:**
```markdown
- [ ] ~~Build~~ **Use** a reusable `Preprocessor` class for ML preprocessing
```

**Option B - Have students build a simpler version first:**
Add cells before the import that have students implement a basic preprocessor:
```python
# Exercise: Build a SimplePreprocessor class
class SimplePreprocessor:
    """Your implementation here - handle numeric scaling only."""
    pass
```
Then show them the production version as a "here's a more complete implementation."

**Recommended:** Option B (better learning experience)

---

### Issue M3: ROC Curve Exercise Without Prior Teaching

**Location:** `lab-1.2.3-visualization-dashboard.ipynb` Exercise section

**Inconsistency:**
- Exercise asks: "Create ROC Curve, Precision-Recall Curve"
- Hint shows `from sklearn.metrics import roc_curve, auc`
- These concepts aren't taught in Module 1.2

**Fix:** Either:
1. Remove this exercise and save for Module 1.6 (Classical ML)
2. Add a brief explanation of ROC/PR curves before the exercise:

```markdown
### ROC and Precision-Recall Curves (Preview)

These are evaluation metrics for classification models:
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **AUC**: Area Under the ROC Curve (0.5 = random, 1.0 = perfect)
- **PR Curve**: Precision vs Recall trade-off

You'll learn these in depth in Module 1.6. For now, here's how to create them:
```

---

### Issue M4: Einsum Attention Without Transformer Context

**Location:** `lab-1.2.4-einsum-mastery.ipynb` Part 4

**Inconsistency:**
- Lab implements scaled dot-product attention
- Transformers aren't taught until Module 2.3

**Why It's Partially OK:**
- The ELI5 explanation of attention is good
- Students don't NEED to understand transformers to learn einsum
- The focus is on einsum notation, not attention theory

**Minor Fix:** Add context note:
```markdown
> **Note:** Attention is a key mechanism in Transformer models (covered in Module 2.3).
> Here we use it as an example of complex tensor operations - focus on the einsum
> patterns, not the ML theory.
```

---

### Issue M5: `keepdims=True` Parameter Not Explained

**Location:** `lab-1.2.1-numpy-broadcasting-lab.ipynb` Exercise 1 Hint

**Inconsistency:**
- Hint uses: `np.mean(data, axis=1, keepdims=True)`
- `keepdims` parameter behavior is never explained

**Fix:** Expand the hint:
```markdown
<details>
<summary>💡 Hint</summary>

Use `np.mean(data, axis=1, keepdims=True)` to get a shape of `(5, 1)` which broadcasts correctly!

**Why `keepdims=True`?**
- Without it: `np.mean(data, axis=1)` returns shape `(5,)` - a 1D array
- With it: returns shape `(5, 1)` - a 2D column vector
- Shape `(5, 1)` broadcasts against `(5, 4)` correctly!
</details>
```

---

### Issue M6: `pd.get_dummies()` Used Without Formal Introduction

**Location:** `lab-1.2.2-dataset-preprocessing-pipeline.ipynb` Cell 20

**Inconsistency:**
- Cell 20 suddenly uses: `pd.get_dummies(df_encoded['employment_type'], prefix='emp', dtype=int)`
- No introduction or explanation of this function

**Fix:** Add introduction in Cell 19:
```python
# Method 2: One-Hot Encoding (for nominal data)
# Employment type has no natural order, so we use one-hot encoding
#
# pd.get_dummies() creates binary columns for each category:
# - prefix: adds a prefix to new column names
# - dtype: sets the data type (int for 0/1 instead of True/False)

employment_dummies = pd.get_dummies(
    df_encoded['employment_type'],
    prefix='emp',      # Columns will be: emp_Full-time, emp_Part-time, etc.
    dtype=int          # Use 0/1 instead of False/True
)
```

---

### Issue M7: `GridSpec` Used Without Prior Teaching

**Location:** `lab-1.2.3-visualization-dashboard.ipynb` Cell 19

**Inconsistency:**
- Dashboard creation uses: `gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)`
- GridSpec is not introduced before this point
- Students trying to modify the layout won't understand how

**Fix:** Add teaching cell before Cell 19:
```python
# Advanced Subplot Layout: GridSpec
# ==================================
# For complex layouts, GridSpec gives you more control than plt.subplots()

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
#             ↑  ↑              ↑             ↑
#           rows cols     height space   width space

# Access individual cells:
ax1 = fig.add_subplot(gs[0, 0])  # Top-left
ax2 = fig.add_subplot(gs[0, 1])  # Top-right
ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left
ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right

# GridSpec also supports spanning:
# ax_wide = fig.add_subplot(gs[0, :])  # Spans full top row
```

---

## 🟢 LOW IMPACT (Style/Polish)

### Issue L1: Memory Estimate Shown as 0.0 GB Throughout

**Location:** All benchmark outputs in lab-1.1.5

**Issue:** Every model shows `Memory: 0.0 GB` due to nvidia-smi limitation

**Suggestion:** Either fix the memory function (see Issue 1) or remove the column from output until fixed.

---

### Issue L2: System Info Script Shows [N/A] for GPU Memory

**Location:** `lab-1.1.1-system-exploration.ipynb` cell 31 (get_system_info function)

**Issue:** Output shows `Memory: [N/A]` for GPU

**Suggestion:** Update to use system memory or PyTorch memory instead:
```python
info["gpu"]["memory_total"] = "128GB unified (use 'free -h' to check)"
```

---

## 📋 CONSISTENCY CHECKLISTS

### Docker Command Consistency

| File | --gpus all | -it | --rm | -v workspace | -v hf_cache | --ipc=host | Container Tag | Port |
|------|------------|-----|------|--------------|-------------|------------|---------------|------|
| README.md | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 25.11-py3 | ❌ |
| lab-1.1.2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 25.11-py3 | ✅ (for Jupyter) |
| lab-1.1.3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 25.11-py3 | ❌ |

**Status:** ✅ Consistent (port mapping differences are intentional and documented)

### Terminology Consistency

| Term | Module 1.1 | Module 1.2 | Consistent? |
|------|----------|----------|-------------|
| Token gen speed | decode tok/s | N/A | ✅ |
| Unified memory | 128GB unified | 128GB unified | ✅ |
| Container type | NGC container | NGC container | ✅ |

### Value Consistency

| Value | Module 1.1 | Module 1.2 | Consistent? |
|-------|----------|----------|-------------|
| GPU Memory | 128GB / [N/A] issue | N/A | ⚠️ Reporting issue |
| Container Tag | 25.11-py3 | 25.11-py3 | ✅ |

---

## 🔧 BULK FIX RECOMMENDATIONS

### Fix Category 1: GPU Memory Monitoring

**Priority: HIGH**

All notebooks should acknowledge unified memory limitations:

1. Update `lab-1.1.1-system-exploration.ipynb`:
   - Add explanation after nvidia-smi cell
   - Add alternative monitoring methods

2. Update `lab-1.1.5-ollama-benchmarking.ipynb`:
   - Replace `get_gpu_memory()` with `get_model_memory_ollama()` or system memory
   - Or remove memory column from output until proper solution

3. Update README.md Common Issues table:
   ```markdown
   | nvidia-smi shows [N/A] for memory | This is expected on unified memory - use `free -h` or PyTorch memory tracking |
   ```

### Fix Category 2: Benchmark Expectations

**Priority: HIGH**

Update expected results to account for model architecture:

1. Add MoE and reasoning model baselines
2. Improve model-to-baseline matching logic
3. Add explanatory table about architecture performance differences

### Fix Category 3: Prerequisites and Teaching Order

**Priority: MEDIUM**

1. Lab 1.2.2: Add sklearn explanation or NumPy alternative
2. Lab 1.2.2: Either change objective or add class-building exercise
3. Lab 1.2.3: Add brief ROC/PR explanation or move exercise
4. Lab 1.2.4: Add note about attention being covered later

---

## 📄 Documentation Coherency Matrix

### Tier 1: Core Docs

| Module | README | QUICKSTART | QUICK_REF | TROUBLESHOOTING | Status |
|--------|--------|------------|-----------|-----------------|--------|
| 1.1 | ✅ | ✅ | ✅ | ✅ | Complete |
| 1.2 | ✅ | ✅ | ✅ | ✅ | Complete |

### Tier 2: Module-Specific

| Module | LAB_PREP | ELI5 | STUDY_GUIDE | SOLUTIONS | Status |
|--------|----------|------|-------------|-----------|--------|
| 1.1 | ✅ | ✅ | ✅ | ✅ (5 labs) | Complete |
| 1.2 | ✅ | ✅ | ✅ | ✅ (5 labs) | Complete |

### Content Matches README

| Doc | Matches README? | Issues |
|-----|-----------------|--------|
| 1.1 QUICKSTART | ✅ | None |
| 1.1 TROUBLESHOOTING | ⚠️ | Missing unified memory [N/A] issue |
| 1.2 QUICKSTART | ✅ | None |
| 1.2 TROUBLESHOOTING | ✅ | None |

---

## ✅ SIGN-OFF CHECKLIST

### HIGH IMPACT (7 issues - blocks student learning)
- [ ] Issue 1: GPU memory monitoring - add unified memory explanation and alternatives
- [ ] Issue 2: Benchmark expectations - add MoE/sparse/reasoning model baselines
- [ ] Issue 3: sklearn - add train_test_split explanation or NumPy alternative
- [ ] Issue 4: `np.linalg.norm()` - add vector normalization teaching cell in 1.2.1
- [ ] Issue 5: ReLU activation - add activation functions cell before Challenge in 1.2.1
- [ ] Issue 6: `groupby()/transform()` - add Pandas groupby teaching cell in 1.2.2
- [ ] Issue 7: `argpartition()/take_along_axis()` - add advanced NumPy cell in 1.2.5

### MEDIUM IMPACT (7 issues - confuses but doesn't block)
- [ ] Issue M1: Docker port mapping clarification
- [ ] Issue M2: Preprocessor class - change objective or add building exercise
- [ ] Issue M3: ROC curve - add explanation or move to Module 1.6
- [ ] Issue M4: Einsum attention - add transformer context note
- [ ] Issue M5: `keepdims` - expand hint with explanation
- [ ] Issue M6: `pd.get_dummies()` - add formal introduction
- [ ] Issue M7: `GridSpec` - add teaching cell before dashboard

### LOW IMPACT (2 issues - polish)
- [ ] Issue L1: Memory column shows 0.0 GB - fix or remove
- [ ] Issue L2: System info shows [N/A] - update messaging

**Coherency Status:** NEEDS FIXES (7 HIGH, 7 MEDIUM, 2 LOW)

---

## 📊 Exercise Coherency Quick Reference

| Lab | Exercise | Untaught Function | Priority |
|-----|----------|-------------------|----------|
| 1.2.1 | Exercise 2 | `np.linalg.norm()` | HIGH |
| 1.2.1 | Challenge | `relu()` | HIGH |
| 1.2.2 | Exercise 1 | `groupby()`, `transform()` | HIGH |
| 1.2.5 | Exercise | `argpartition()`, `take_along_axis()` | HIGH |
| 1.2.1 | Exercise 1 | `keepdims=True` explained | MEDIUM |
| 1.2.2 | Part 3 | `pd.get_dummies()` | MEDIUM |
| 1.2.3 | Part 5 | `GridSpec` | MEDIUM |
| 1.2.3 | Exercise | `roc_curve`, `auc` | MEDIUM |

---

*Audit by ConsistencyAuditor SPARK - Curriculum v2.0*
*Session: claude/review-curriculum-coherency-ffwST*
