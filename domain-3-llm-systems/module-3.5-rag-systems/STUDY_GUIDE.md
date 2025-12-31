# Module 3.5: RAG Systems & Vector Databases - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Design** and implement complete RAG architectures from scratch
2. **Use** vector databases (ChromaDB, FAISS, Qdrant) effectively
3. **Implement** advanced retrieval (hybrid search, reranking)
4. **Evaluate** RAG systems with RAGAS and custom metrics

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-3.5.1-basic-rag.ipynb | End-to-end RAG pipeline | ~3 hr | Working retrieval + generation |
| 2 | lab-3.5.2-chunking.ipynb | Chunking strategies | ~2 hr | Optimal chunk size selection |
| 3 | lab-3.5.3-vector-dbs.ipynb | Database comparison | ~2 hr | ChromaDB vs FAISS vs Qdrant |
| 4 | lab-3.5.4-hybrid-search.ipynb | Dense + sparse fusion | ~2 hr | Better retrieval quality |
| 5 | lab-3.5.5-reranking.ipynb | Cross-encoder reranking | ~2 hr | 2-stage retrieval pipeline |
| 6 | lab-3.5.6-evaluation.ipynb | RAGAS framework | ~2 hr | Systematic quality metrics |
| 7 | lab-3.5.7-production-rag.ipynb | Production system | ~2 hr | Error handling, caching, monitoring |

**Total time**: ~15 hours

## 🔑 Core Concepts

### Retrieval-Augmented Generation (RAG)
**What**: Retrieve relevant documents, then generate answers using them as context
**Why it matters**: Accurate, up-to-date, traceable answers from your own data
**First appears in**: Lab 3.5.1

### Vector Embeddings
**What**: Dense numerical representations of text that capture semantic meaning
**Why it matters**: Enable "search by meaning" instead of just keyword matching
**First appears in**: Lab 3.5.1

### Hybrid Search
**What**: Combining dense (embedding) and sparse (BM25) retrieval
**Why it matters**: Best of both worlds—meaning + keywords
**First appears in**: Lab 3.5.4

### Reranking
**What**: Using a cross-encoder to re-score initial retrieval results
**Why it matters**: Significantly improves retrieval precision
**First appears in**: Lab 3.5.5

## 🔗 How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 3.4          ──►     Module 3.5          ──►    Module 3.6
Test-Time Compute           RAG Systems               AI Agents
(reasoning)                 (retrieval)               (tool use)
```

**Builds on**:
- **LLM inference** from Module 3.3 (using models for generation)
- **Reasoning strategies** from Module 3.4 (can combine with RAG)
- **GPU acceleration** from Module 1.3 (FAISS on GPU)

**Prepares for**:
- Module 3.6 uses RAG as a tool for agents
- Module 4.3 covers RAG evaluation in production
- Module 4.5 builds RAG demos with Gradio

## 📖 Recommended Approach

### Standard Path (12-15 hours):
1. **Day 1: Fundamentals (Labs 1-3)**
   - Lab 3.5.1 builds complete pipeline
   - Lab 3.5.2 optimizes chunking
   - Lab 3.5.3 compares databases

2. **Day 2: Advanced Retrieval (Labs 4-5)**
   - Lab 3.5.4 adds hybrid search
   - Lab 3.5.5 adds reranking

3. **Day 3: Evaluation & Production (Labs 6-7)**
   - Lab 3.5.6 establishes quality metrics
   - Lab 3.5.7 adds production features

### Quick Path (8-10 hours, if experienced):
1. Do Lab 3.5.1 (complete pipeline)
2. Skip to Lab 3.5.4 (hybrid search)
3. Lab 3.5.5 (reranking)
4. Lab 3.5.6 (evaluation)

## 📋 Before You Start
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute RAG demo
→ See [ELI5.md](./ELI5.md) for concept explanations
