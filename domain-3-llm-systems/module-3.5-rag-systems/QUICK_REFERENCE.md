# Module 3.5: RAG Systems & Vector Databases - Quick Reference

## 🚀 Essential Commands

### Install RAG Dependencies
```bash
pip install \
    langchain langchain-community langchain-huggingface \
    chromadb faiss-cpu qdrant-client \
    sentence-transformers \
    rank_bm25 ragas \
    pypdf2 pdfplumber \
    ollama
```

### NGC Container
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

## 📊 Key Values

### Embedding Models
| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| bge-large-en-v1.5 | 1024 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| qwen3-embedding:8b | 4096 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

> **2025 Tier 1**: Use `qwen3-embedding:8b` for production (#1 MTEB multilingual, 100+ langs)

### Chunk Size Guidelines
| Size | Use Case |
|------|----------|
| 256 | Very precise retrieval, Q&A |
| 512 | Good balance (default) |
| 1024 | More context, less precise |

### DGX Spark RAG Performance
| Component | Memory | Speed |
|-----------|--------|-------|
| BGE-large embeddings | ~2GB | ~1000 docs/sec |
| BGE-reranker-large | ~2GB | ~100 pairs/sec |
| ChromaDB | ~1GB + index | ~1ms/query |
| FAISS (GPU) | ~1GB + index | ~0.1ms/query |

## 🔧 Common Patterns

### Pattern: Basic RAG Pipeline
```python
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# 1. Create embeddings
embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# 2. Create vector store
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")

# 3. Add documents
embeddings = embed_model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# 4. Query
def rag_query(question):
    query_emb = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=5)
    context = "\n".join(results['documents'][0])

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = ollama.chat(model="qwen3:32b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
```

### Pattern: Chunking with LangChain
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

### Pattern: Hybrid Search
```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, documents, embed_model, alpha=0.5):
        self.documents = documents
        self.doc_embeddings = embed_model.encode([d for d in documents])
        self.bm25 = BM25Okapi([d.lower().split() for d in documents])
        self.alpha = alpha

    def retrieve(self, query, k=5):
        # Dense scores
        query_emb = embed_model.encode([query])
        dense_scores = np.dot(self.doc_embeddings, query_emb.T).flatten()

        # Sparse scores
        sparse_scores = self.bm25.get_scores(query.lower().split())

        # Normalize and combine
        dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-6)
        sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.ptp() + 1e-6)
        hybrid = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        top_k = np.argsort(hybrid)[-k:][::-1]
        return [self.documents[i] for i in top_k]
```

### Pattern: Reranking
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large", device="cuda")

def rerank(query, documents, k=5):
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)
    top_k = np.argsort(scores)[-k:][::-1]
    return [documents[i] for i in top_k]
```

### Pattern: RAGAS Evaluation
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(result)
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Chunks too small | Use 512+ tokens with overlap |
| No chunk overlap | Add 50-100 token overlap |
| Wrong embedding model | Use same model for indexing and querying |
| Ignoring metadata | Include source, section in metadata |
| No reranking | Add cross-encoder for better precision |
| Not evaluating | Use RAGAS to measure quality |

## 🔗 Quick Links
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [RAGAS Documentation](https://docs.ragas.io/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [BGE Models](https://huggingface.co/BAAI/bge-large-en-v1.5)
