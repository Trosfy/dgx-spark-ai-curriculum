# Module 3.5: RAG Systems & Vector Databases

**Domain:** 3 - LLM Systems
**Duration:** Weeks 24-25 (12-15 hours)
**Prerequisites:** Module 3.4 (Test-Time Compute)
**Priority:** P0 Critical

---

## Overview

Retrieval-Augmented Generation (RAG) is how you make LLMs useful for real-world applications. Instead of relying solely on the model's training data, RAG retrieves relevant information from your own documents and uses it to generate accurate, grounded responses. This is *the* most requested skill in LLM job postings.

**ELI5:** Imagine you're a student taking an open-book exam. RAG is like giving the LLM access to its own textbook—it can look up the right information before answering, instead of trying to remember everything from memory.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Build production-ready RAG pipelines from scratch
- ✅ Select and configure appropriate vector databases
- ✅ Optimize retrieval quality with advanced techniques (hybrid search, reranking)
- ✅ Evaluate RAG systems using industry-standard metrics

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.5.1 | Design and implement complete RAG architectures | Create |
| 3.5.2 | Use vector databases (ChromaDB, FAISS, Qdrant) effectively | Apply |
| 3.5.3 | Implement advanced retrieval (hybrid search, reranking) | Apply |
| 3.5.4 | Evaluate RAG systems with RAGAS and custom metrics | Evaluate |

---

## Topics

### 3.5.1 RAG Fundamentals

- **What is RAG?**
  - Retrieval-Augmented Generation concept
  - Why RAG beats fine-tuning for factual knowledge
  - RAG architecture patterns

- **When to Use RAG**
  - Dynamic, frequently updated information
  - Domain-specific knowledge bases
  - Citation and source requirements
  - Large document collections

- **RAG vs Fine-Tuning**
  - RAG: Cheaper, updatable, traceable sources
  - Fine-tuning: Better for style/behavior, no retrieval latency
  - Hybrid approaches

### 3.5.2 Document Processing

- **Document Loading**
  - PDF extraction (PyPDF2, pdfplumber)
  - Word documents (python-docx)
  - HTML parsing (BeautifulSoup)
  - Markdown and code files

- **Chunking Strategies**
  - Fixed size with overlap
  - Semantic chunking (by meaning boundaries)
  - Sentence-based chunking
  - Recursive character splitting
  - Chunk size tradeoffs (smaller = precise, larger = context)

- **Metadata Enrichment**
  - Source tracking
  - Section headings
  - Creation/modification dates
  - Custom tags and categories

### 3.5.3 Embedding Models

- **How Embeddings Work**
  - Text → Dense vector representation
  - Semantic similarity via cosine distance

- **Model Selection**
  - BGE (bge-large-en-v1.5) - excellent quality
  - Nomic-embed-text - runs on Ollama
  - E5 models - good multilingual support
  - OpenAI ada-002 - API-based baseline

- **Running Embeddings on DGX Spark**
  - GPU-accelerated embedding generation
  - Batch processing for large corpora
  - Memory management for large models

### 3.5.4 Vector Databases

- **ChromaDB**
  - Simple, Python-native
  - Perfect for development and small-medium scale
  - Persistent storage

- **FAISS (Facebook AI Similarity Search)**
  - GPU-accelerated on DGX Spark
  - Index types: Flat, IVF, HNSW
  - Best for pure performance

- **Qdrant**
  - Production-ready features
  - Advanced filtering
  - Scalar/binary quantization

- **Index Types and Tradeoffs**
  - Flat (exact): 100% recall, slow
  - IVF: Fast, good recall
  - HNSW: Fast, excellent recall

### 3.5.5 Advanced Retrieval

- **Hybrid Search**
  - Dense retrieval (embeddings)
  - Sparse retrieval (BM25/keyword)
  - Fusion methods (RRF, linear combination)

- **Query Expansion**
  - LLM-based query rewriting
  - Multi-query retrieval

- **Reranking**
  - Cross-encoder rerankers
  - BGE-reranker, Cohere rerank
  - Two-stage retrieval pipeline

- **Hypothetical Document Embedding (HyDE)**
  - Generate hypothetical answer
  - Embed the hypothetical for retrieval

### 3.5.6 RAG Evaluation

- **Retrieval Metrics**
  - Recall@K: Are relevant docs in top K?
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

- **Generation Metrics (RAGAS)**
  - Faithfulness: Is answer grounded in retrieved docs?
  - Answer Relevancy: Does answer address the question?
  - Context Precision: Are retrieved docs relevant?
  - Context Recall: Are all relevant docs retrieved?

---

## Labs

### Lab 3.5.1: Basic RAG Pipeline
**Time:** 3 hours

Build a complete RAG system from scratch.

**Instructions:**
1. Open `labs/lab-3.5.1-basic-rag.ipynb`
2. Load sample documents (PDFs, markdown)
3. Implement chunking with RecursiveCharacterTextSplitter
4. Generate embeddings with BGE-large
5. Store in ChromaDB
6. Implement retrieval + generation pipeline
7. Test with 10 questions

**Deliverable:** Working end-to-end RAG pipeline

---

### Lab 3.5.2: Chunking Strategies Comparison
**Time:** 2 hours

Find the optimal chunking for your use case.

**Instructions:**
1. Open `labs/lab-3.5.2-chunking.ipynb`
2. Implement fixed-size chunking (256, 512, 1024 tokens)
3. Implement semantic chunking (by section breaks)
4. Implement sentence-based chunking
5. Create evaluation set of 20 Q&A pairs
6. Measure retrieval quality for each strategy
7. Document best practices

**Deliverable:** Chunking comparison report with recommendations

---

### Lab 3.5.3: Vector Database Comparison
**Time:** 2 hours

Evaluate different vector database options.

**Instructions:**
1. Open `labs/lab-3.5.3-vector-dbs.ipynb`
2. Implement same RAG with ChromaDB
3. Implement with FAISS (GPU-accelerated)
4. Implement with Qdrant
5. Benchmark: indexing time, query latency, memory usage
6. Test filtering and metadata queries
7. Document pros/cons of each

**Deliverable:** Vector database comparison report

---

### Lab 3.5.4: Hybrid Search Implementation
**Time:** 2 hours

Combine dense and sparse retrieval.

**Instructions:**
1. Open `labs/lab-3.5.4-hybrid-search.ipynb`
2. Implement BM25 sparse retrieval
3. Implement dense retrieval with embeddings
4. Implement Reciprocal Rank Fusion (RRF)
5. Compare: dense-only vs sparse-only vs hybrid
6. Find optimal fusion weights

**Deliverable:** Hybrid search with measured improvement

---

### Lab 3.5.5: Reranking Pipeline
**Time:** 2 hours

Add cross-encoder reranking for quality boost.

**Instructions:**
1. Open `labs/lab-3.5.5-reranking.ipynb`
2. Implement two-stage retrieval (top-50 → rerank → top-5)
3. Load BGE-reranker-large
4. Benchmark quality improvement
5. Measure latency tradeoff
6. Find optimal first-stage K value

**Deliverable:** Reranking pipeline with quality/latency analysis

---

### Lab 3.5.6: RAGAS Evaluation Framework
**Time:** 2 hours

Build systematic RAG evaluation.

**Instructions:**
1. Open `labs/lab-3.5.6-evaluation.ipynb`
2. Install and configure RAGAS
3. Create evaluation dataset (50 Q&A pairs with ground truth)
4. Measure: faithfulness, relevancy, precision, recall
5. Create evaluation dashboard
6. Set quality thresholds for production

**Deliverable:** RAGAS evaluation framework with benchmarks

---

### Lab 3.5.7: Production RAG System
**Time:** 2 hours

Build a production-ready RAG pipeline.

**Instructions:**
1. Open `labs/lab-3.5.7-production-rag.ipynb`
2. Add error handling and retries
3. Implement caching (query → result)
4. Add logging and monitoring
5. Create health check endpoint
6. Handle edge cases (empty results, long queries)
7. Benchmark throughput

**Deliverable:** Production-ready RAG with monitoring

---

## Guidance

### Local RAG Stack on DGX Spark

```python
# Embeddings: GPU-accelerated on DGX Spark
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cuda"
)

# Vector DB: ChromaDB for development
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

# LLM: 32B via Ollama for best quality
import ollama
response = ollama.chat(model="qwen3:32b", messages=[...])

# Reranker: BGE-reranker for quality boost
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-large", device="cuda")
```

### Basic RAG Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load and chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)

# 2. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"}
)

# 3. Store in vector database
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# 5. RAG query function
def rag_query(question: str) -> str:
    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate answer
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="qwen3:32b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
```

### Hybrid Search with BM25

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, documents, embeddings, alpha=0.5):
        # Dense retrieval
        self.embeddings = embeddings
        self.doc_embeddings = embeddings.encode([d.page_content for d in documents])

        # Sparse retrieval (BM25)
        tokenized = [d.page_content.lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)

        self.documents = documents
        self.alpha = alpha  # Weight for dense vs sparse

    def retrieve(self, query: str, k: int = 5):
        # Dense scores
        query_emb = self.embeddings.encode([query])
        dense_scores = np.dot(self.doc_embeddings, query_emb.T).flatten()

        # Sparse scores (BM25)
        sparse_scores = self.bm25.get_scores(query.lower().split())

        # Normalize and combine
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-6)

        hybrid_scores = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        # Return top-k
        top_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return [self.documents[i] for i in top_indices]
```

### Reranking Pipeline

```python
from sentence_transformers import CrossEncoder

class RerankingRetriever:
    def __init__(self, base_retriever, reranker_model="BAAI/bge-reranker-large"):
        self.base_retriever = base_retriever
        self.reranker = CrossEncoder(reranker_model, device="cuda")

    def retrieve(self, query: str, k: int = 5, first_stage_k: int = 50):
        # First stage: get more candidates
        candidates = self.base_retriever.retrieve(query, k=first_stage_k)

        # Rerank with cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Return top-k after reranking
        sorted_indices = np.argsort(scores)[-k:][::-1]
        return [candidates[i] for i in sorted_indices]
```

### DGX Spark Performance

| Component | Configuration | Memory | Speed |
|-----------|--------------|--------|-------|
| BGE-large embeddings | FP16 | ~2GB | ~1000 docs/sec |
| BGE-reranker-large | FP16 | ~2GB | ~100 pairs/sec |
| ChromaDB | Default | ~1GB + index | ~1ms/query |
| FAISS (GPU) | IVF-Flat | ~1GB + index | ~0.1ms/query |
| Qwen3 32B | Q4 | ~20GB | ~25 decode tok/s |

**Total for full RAG stack:** ~50GB, leaving plenty of headroom on 128GB system.

---

## DGX Spark Setup

### NGC Container Launch

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Install RAG Dependencies

```bash
# Inside NGC container
pip install \
    langchain langchain-community langchain-huggingface \
    chromadb faiss-cpu qdrant-client \
    sentence-transformers \
    rank_bm25 \
    ragas \
    pypdf2 pdfplumber python-docx beautifulsoup4 \
    nltk \
    ollama

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Ollama Setup

```bash
# Start Ollama in a separate terminal
ollama serve

# Pull models for RAG
ollama pull qwen3:32b            # LLM for generation
ollama pull qwen3-embedding:8b   # Alternative local embeddings
```

---

## Milestone Checklist

- [ ] Basic RAG pipeline working end-to-end
- [ ] Chunking strategies compared with quality metrics
- [ ] Multiple vector databases tested and compared
- [ ] Hybrid search improving retrieval quality
- [ ] Reranking pipeline with measured improvement
- [ ] RAGAS evaluation framework created
- [ ] Production-ready RAG system built

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Embeddings OOM | Use smaller batch size or FP16 |
| Poor retrieval quality | Increase chunk overlap, try hybrid search |
| ChromaDB slow on large corpus | Switch to FAISS with GPU or Qdrant |
| Hallucinations despite RAG | Check faithfulness score, improve prompting |
| Reranker too slow | Reduce first-stage K or batch reranking |
| BM25 memory issues | Use streaming or incremental indexing |

---

## Next Steps

After completing this module:
1. ✅ Verify all milestones are checked
2. 📁 Save reusable RAG components to `scripts/`
3. ➡️ Proceed to [Module 3.6: AI Agents & Agentic Systems](../module-3.6-ai-agents/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 3.4: Test-Time Compute](../module-3.4-test-time-compute/) | **Module 3.5: RAG Systems** | [Module 3.6: AI Agents](../module-3.6-ai-agents/) |

---

## Study Materials

| Document | Purpose | Time |
|----------|---------|------|
| [QUICKSTART.md](./QUICKSTART.md) | 5-minute working RAG demo | 5 min |
| [ELI5.md](./ELI5.md) | Plain-language concept explanations | 15 min |
| [PREREQUISITES.md](./PREREQUISITES.md) | Skills check before starting | 10 min |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning path and objectives | 10 min |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Commands, patterns, and values | Reference |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup checklist | 15 min |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors, solutions, and FAQs | Reference |

---

## Resources

- [RAG Paper](https://arxiv.org/abs/2005.11401) - Original RAG work
- [RAGAS Documentation](https://docs.ragas.io/) - Evaluation framework
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [BGE Embedding Models](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [Hybrid Search Guide](https://www.pinecone.io/learn/hybrid-search/)
