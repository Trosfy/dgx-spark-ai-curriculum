# Coherency Audit Report - Module 3.3

**Module(s) Reviewed:** Module 3.3 - Deployment & Inference
**Files Analyzed:** 14 (README, 6 notebooks, 3 solution notebooks, 3 scripts, data README)
**Inconsistencies Found:** 6 (All Fixed)
**Audit Date:** 2025-12-30
**Auditor:** ConsistencyAuditor SPARK

---

## Summary

| Category | Issues Found | Status |
|----------|--------------|--------|
| Code ↔ Explanation | 0 | ✅ |
| Code ↔ Table | 0 | ✅ |
| Cross-File | 5 | ✅ Fixed |
| Cross-Module | 1 | ✅ Fixed |
| Terminology | 0 | ✅ |
| Values | 0 | ✅ |
| **TOTAL** | **6** | **✅ All Fixed** |

---

## Issues Fixed

### Issue 1: Container Version in 01-engine-benchmark.ipynb

**Type:** Cross-File Version Inconsistency

**Location:**
- File: `notebooks/01-engine-benchmark.ipynb`
- Cell: 6 (markdown)

**The Inconsistency:**
PyTorch container tag was `24.10-py3` instead of the standard `25.11-py3`:
```bash
nvcr.io/nvidia/pytorch:24.10-py3
```

**Fix Applied:**
Updated container tag to `25.11-py3`.

---

### Issue 2: Container Version in 02-vllm-deployment.ipynb

**Type:** Cross-File Version Inconsistency

**Location:**
- File: `notebooks/02-vllm-deployment.ipynb`
- Cells: 4 (markdown), 5 (code), 25 (code)

**The Inconsistency:**
PyTorch container tag was `24.10-py3` in multiple cells:
- Cell 4: Docker examples in markdown
- Cell 5: `generate_vllm_command()` function output
- Cell 25: Cleanup docker stop command

**Fix Applied:**
Updated all three cells to use `25.11-py3`.

---

### Issue 3: Container Version in 03-tensorrt-llm-optimization.ipynb

**Type:** Cross-File Version Inconsistency

**Location:**
- File: `notebooks/03-tensorrt-llm-optimization.ipynb`
- Cells: 5 (code), 26 (code)

**The Inconsistency:**
TensorRT-LLM container tag was `24.10-trtllm-python-py3`:
```python
trtllm_container = "nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3"
```

**Fix Applied:**
Updated container tag to `25.11-trtllm-python-py3` in both cells.

---

### Issue 4: Missing Container Version in 03-tensorrt-llm-optimization.ipynb (Common Mistakes Section)

**Type:** Incomplete Container Tag

**Location:**
- File: `notebooks/03-tensorrt-llm-optimization.ipynb`
- Cell: 20 (markdown - Common Mistakes section)

**The Inconsistency:**
Container tags in the "Common Mistakes" examples were missing version numbers:
```bash
# ❌ Wrong - Build will fail or be very slow
docker run --gpus all -it nvcr.io/nvidia/tritonserver:trtllm-python-py3

# ✅ Right - Allocate enough shared memory
docker run --gpus all --shm-size=16g -it nvcr.io/nvidia/tritonserver:trtllm-python-py3
```

**Fix Applied (2025-12-30):**
Added version `25.11-` to both container tags:
```bash
# ❌ Wrong - Build will fail or be very slow
docker run --gpus all -it nvcr.io/nvidia/tritonserver:25.11-trtllm-python-py3

# ✅ Right - Allocate enough shared memory
docker run --gpus all --shm-size=16g -it nvcr.io/nvidia/tritonserver:25.11-trtllm-python-py3
```

---

### Issue 5: Container Version in solutions/03-tensorrt-llm-optimization-solution.ipynb

**Type:** Outdated Container Version in Solution

**Location:**
- File: `solutions/03-tensorrt-llm-optimization-solution.ipynb`
- Cell: 2 (code)

**The Inconsistency:**
Solution notebook still used old container version `24.10-trtllm-python-py3`:
```python
container_cmd = """
docker run --gpus all -it --rm \\
    ...
    nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 \\
    bash
"""
```

**Fix Applied (2025-12-30):**
Updated container tag to `25.11-trtllm-python-py3`.

---

### Issue 6: Previous Audit Incomplete

**Type:** Audit Report Inconsistency

**Location:**
- File: `COHERENCY_AUDIT_REPORT.md`

**The Inconsistency:**
Previous audit report did not catch issues #4 and #5.

**Fix Applied:**
Comprehensive re-audit completed, all issues now documented and fixed.

---

## Comprehensive Consistency Verification

### NGC Container Tag Consistency

| File | Container | Version | Status |
|------|-----------|---------|--------|
| README.md | pytorch | 25.11-py3 | ✅ Consistent |
| README.md | tritonserver | 25.11-trtllm-python-py3 | ✅ Consistent |
| 01-engine-benchmark.ipynb | pytorch | 25.11-py3 | ✅ Fixed |
| 02-vllm-deployment.ipynb | pytorch | 25.11-py3 | ✅ Fixed |
| 03-tensorrt-llm-optimization.ipynb | tritonserver | 25.11-trtllm-python-py3 | ✅ Fixed |
| 03-tensorrt-llm-optimization.ipynb (cell 20) | tritonserver | 25.11-trtllm-python-py3 | ✅ Fixed |
| solutions/03-tensorrt-llm-optimization-solution.ipynb | tritonserver | 25.11-trtllm-python-py3 | ✅ Fixed |
| 04-speculative-decoding.ipynb | N/A | N/A | ✅ N/A |
| 05-production-api.ipynb | N/A | N/A | ✅ N/A |
| 06-ollama-web-ui.ipynb | N/A | N/A | ✅ N/A |

### Docker Flag Consistency

| Flag | README | Notebooks | Status |
|------|--------|-----------|--------|
| `--gpus all` | ✅ | ✅ | ✅ Consistent |
| `-it` | ✅ | ✅ | ✅ Consistent |
| `--rm` | ✅ | ✅ | ✅ Consistent |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | ✅ | ✅ | ✅ Consistent |
| `--ipc=host` | ✅ | ✅ | ✅ Consistent |

### Inference Engine Terminology Consistency

| Term | Usage | Status |
|------|-------|--------|
| Ollama URL | `http://localhost:11434` | ✅ Consistent |
| vLLM URL | `http://localhost:8000` | ✅ Consistent |
| SGLang URL | `http://localhost:30000` | ✅ Consistent |
| TensorRT-LLM URL | `http://localhost:8000` | ✅ Consistent |
| Model name (HF) | `meta-llama/Llama-3.1-8B-Instruct` | ✅ Consistent |
| Model name (Ollama) | `llama3.1:8b` | ✅ Consistent |
| DGX Spark memory | 128GB unified memory | ✅ Consistent |

### Value Consistency

| Value | README | Notebooks | Status |
|-------|--------|-----------|--------|
| vLLM gpu-memory-utilization | 0.9 | 0.9 | ✅ Consistent |
| max-model-len | 4096 | 4096 | ✅ Consistent |
| dtype | bfloat16 | bfloat16 | ✅ Consistent |
| TRT-LLM shm-size | 16g | 16g | ✅ Consistent |

### Cross-Module Consistency Check

| Element | Module 1 | Module 11 | Module 12 | Status |
|---------|----------|-----------|-----------|--------|
| PyTorch container | 25.11-py3 | 25.11-py3 | 25.11-py3 | ✅ Consistent |
| DGX Spark memory | 128GB | 128GB | 128GB | ✅ Consistent |
| Buffer cache command | ✅ Same | ✅ Same | ✅ Same | ✅ Consistent |
| HuggingFace cache mount | ✅ Same | ✅ Same | ✅ Same | ✅ Consistent |

---

## ✅ SIGN-OFF

- [x] All HIGH impact issues resolved
- [x] Docker commands standardized
- [x] NGC container versions consistent (pytorch: `25.11-py3`, tritonserver: `25.11-trtllm-python-py3`)
- [x] All example commands include version numbers
- [x] Solution notebooks match main notebooks
- [x] Inference engine URLs consistent
- [x] Model names consistent
- [x] vLLM configuration values consistent
- [x] README ↔ Notebook alignment verified
- [x] Cross-module consistency verified

**Coherency Status:** ✅ CONSISTENT

---

*Audit by ConsistencyAuditor SPARK*
*Report generated: 2025-12-30*
*Last updated: 2025-12-30 (Additional fixes applied to cell 20 and solution notebook)*
