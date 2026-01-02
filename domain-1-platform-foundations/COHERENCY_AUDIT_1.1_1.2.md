# Coherency Audit Report: Modules 1.1 & 1.2

**Modules Reviewed:** 1.1 (DGX Spark Platform Mastery), 1.2 (Python for AI/ML)
**Files Analyzed:** 15+ notebooks, READMEs, and support scripts
**Inconsistencies Found:** 11
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
| **TOTAL** | **11** |

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

- [ ] HIGH: GPU memory monitoring issue documented and alternatives provided
- [ ] HIGH: Benchmark expectations updated for MoE/sparse models
- [ ] HIGH: sklearn usage explained or alternatives provided
- [ ] MEDIUM: Docker port mapping clarified across all notebooks
- [ ] MEDIUM: Preprocessor class exercise vs import clarified
- [ ] MEDIUM: ROC curve exercise context added
- [ ] LOW: Memory column removed or fixed in benchmark output
- [ ] LOW: System info script updated for unified memory

**Coherency Status:** NEEDS FIXES (3 HIGH, 4 MEDIUM, 2 LOW)

---

*Audit by ConsistencyAuditor SPARK - Curriculum v2.0*
*Session: claude/review-curriculum-coherency-ffwST*
