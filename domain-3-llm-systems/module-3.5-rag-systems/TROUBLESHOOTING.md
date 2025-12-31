# Module 3.5: RAG Systems & Vector Databases - Troubleshooting Guide

## 🔍 Quick Diagnosis

### Symptom Categories
| Symptom | Jump to Section |
|---------|-----------------|
| Embedding model won't load | [Embedding Issues](#embedding-issues) |
| Vector database errors | [Vector Database Issues](#vector-database-issues) |
| Poor retrieval quality | [Retrieval Quality Issues](#retrieval-quality-issues) |
| LLM not generating | [Generation Issues](#generation-issues) |
| Memory errors | [Memory Issues](#memory-issues) |
| Slow performance | [Performance Issues](#performance-issues) |

---

## Embedding Issues

### ❌ Error: "Could not load model"
```
OSError: Can't load tokenizer for 'BAAI/bge-large-en-v1.5'
```

**Cause**: Model not downloaded or network issue

**Solution**:
```python
# Force download with explicit cache
from sentence_transformers import SentenceTransformer
import os

os.environ['HF_HOME'] = '/root/.cache/huggingface'
model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    cache_folder="/root/.cache/huggingface"
)
```

---

### ❌ Error: "CUDA out of memory during encoding"
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Cause**: Batch size too large or model already using GPU memory

**Solution**:
```python
# Use smaller batch size
embeddings = model.encode(
    documents,
    batch_size=8,  # Reduce from default 32
    show_progress_bar=True
)

# Or encode on CPU if needed
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
```

---

### ❌ Error: "Embedding dimensions don't match"
```
ValueError: Embedding dimension mismatch: expected 1024, got 384
```

**Cause**: Different embedding models used for indexing vs querying

**Solution**:
```python
# Always use the SAME model for indexing and querying
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Define once

# Indexing
index_model = SentenceTransformer(EMBEDDING_MODEL)
doc_embeddings = index_model.encode(documents)

# Querying - use same model!
query_model = SentenceTransformer(EMBEDDING_MODEL)  # Same!
query_embedding = query_model.encode([query])
```

---

## Vector Database Issues

### ❌ ChromaDB: "sqlite3.OperationalError"
```
sqlite3.OperationalError: table already exists
```

**Cause**: Database corruption or version mismatch

**Solution**:
```python
# Option 1: Delete and recreate
import shutil
shutil.rmtree("./chroma_db", ignore_errors=True)

# Option 2: Use in-memory client for testing
import chromadb
client = chromadb.Client()  # Ephemeral, no persistence
```

---

### ❌ ChromaDB: "Collection already exists"
```
chromadb.errors.UniqueConstraintError: Collection already exists
```

**Solution**:
```python
# Use get_or_create instead of create
collection = client.get_or_create_collection("my_collection")

# Or delete first
client.delete_collection("my_collection")
collection = client.create_collection("my_collection")
```

---

### ❌ FAISS: "ImportError: No module named 'faiss'"
**Solution**:
```bash
# Install CPU version
pip install faiss-cpu

# NOT just 'faiss' - that's a different package!
```

---

### ❌ FAISS: "Index dimension mismatch"
```
RuntimeError: Dimension of query vector doesn't match index
```

**Solution**:
```python
# Check dimensions match
print(f"Index dimension: {index.d}")
print(f"Query dimension: {query_embedding.shape[1]}")

# Recreate index with correct dimension
import faiss
d = 1024  # Must match your embedding model
index = faiss.IndexFlatL2(d)
```

---

### ❌ Qdrant: "Connection refused"
```
qdrant_client.http.exceptions.UnexpectedResponse: Connection refused
```

**Solution**:
```bash
# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Or use in-memory mode
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")  # No server needed
```

---

## Retrieval Quality Issues

### ❌ Problem: Irrelevant documents retrieved

**Diagnosis checklist**:
1. Are chunks too large? (Getting whole documents instead of relevant parts)
2. Are chunks too small? (Missing context)
3. Is the embedding model appropriate for your domain?

**Solutions**:

```python
# 1. Adjust chunk size
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # Try 256-1024
    chunk_overlap=50,    # Add overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""]
)

# 2. Use hybrid search (dense + sparse)
from rank_bm25 import BM25Okapi

# Combine BM25 with embeddings
bm25_scores = bm25.get_scores(query.split())
dense_scores = cosine_similarity(query_emb, doc_embs)
hybrid_scores = 0.5 * normalize(bm25_scores) + 0.5 * normalize(dense_scores)

# 3. Add reranking
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-large")
pairs = [(query, doc) for doc in retrieved_docs]
rerank_scores = reranker.predict(pairs)
```

---

### ❌ Problem: Missing relevant documents

**Cause**: Query doesn't semantically match documents

**Solutions**:

```python
# 1. Increase k (retrieve more candidates)
results = collection.query(query_embeddings=query_emb, n_results=20)  # Not just 5

# 2. Use query expansion
def expand_query(query):
    # Add synonyms or rephrasings
    return [
        query,
        f"What is {query}?",
        f"Tell me about {query}",
    ]

expanded = expand_query("DGX Spark memory")
all_results = []
for q in expanded:
    results = retrieve(q, k=5)
    all_results.extend(results)

# 3. Use HyDE (Hypothetical Document Embeddings)
def hyde_query(query, llm):
    # Generate hypothetical answer
    hypo_answer = llm.generate(f"Answer this: {query}")
    # Embed the hypothetical answer instead of query
    return embed_model.encode([hypo_answer])
```

---

### ❌ Problem: Duplicate results

**Solution**:
```python
# Deduplicate by content hash
def deduplicate(results, threshold=0.95):
    seen = set()
    unique = []
    for doc in results:
        # Simple hash-based dedup
        doc_hash = hash(doc[:100])  # First 100 chars
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique.append(doc)
    return unique
```

---

## Generation Issues

### ❌ Problem: LLM ignores retrieved context

**Cause**: Prompt doesn't properly integrate context

**Solution**:
```python
# Better prompt template
prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Instructions:
- Use ONLY information from the context above
- Quote relevant parts when possible
- If unsure, say so

Answer:"""
```

---

### ❌ Problem: LLM hallucinates despite context

**Solutions**:
```python
# 1. Lower temperature for more deterministic output
response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": prompt}],
    options={"temperature": 0.1}  # Lower = more focused
)

# 2. Use a more explicit system prompt
messages = [
    {"role": "system", "content": "You are a factual assistant. Only use information from the provided context. Never make up facts."},
    {"role": "user", "content": prompt}
]

# 3. Add citation requirements
prompt += "\nFor each claim, cite the source document in brackets [Source: doc_name]."
```

---

### ❌ Error: Ollama connection refused
```
ConnectionError: Connection refused at localhost:11434
```

**Solution**:
```bash
# Start Ollama service
ollama serve

# Or in background
nohup ollama serve &

# Check if running
curl http://localhost:11434/api/tags
```

---

## Memory Issues

### ❌ Error: GPU OOM during RAG pipeline

**Diagnosis**:
```python
import torch
print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

**Solutions**:
```python
# 1. Use smaller embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dim vs 1024

# 2. Encode in batches
for i in range(0, len(docs), 100):
    batch = docs[i:i+100]
    embeddings = model.encode(batch)
    # Process and clear
    torch.cuda.empty_cache()

# 3. Move reranker to CPU after use
reranker = CrossEncoder("BAAI/bge-reranker-large", device="cuda")
scores = reranker.predict(pairs)
del reranker
torch.cuda.empty_cache()

# 4. Use CPU for ChromaDB operations
# ChromaDB embedding functions default to CPU
```

---

### ❌ Error: RAM exhaustion with large document collections

**Solution**:
```python
# 1. Stream documents instead of loading all
def stream_documents(file_paths):
    for path in file_paths:
        with open(path) as f:
            yield f.read()

# 2. Use persistent vector database
client = chromadb.PersistentClient(path="./chroma_db")

# 3. Process in chunks
for batch in batched(documents, 1000):
    embeddings = model.encode(batch)
    collection.add(documents=batch, embeddings=embeddings.tolist())
```

---

## Performance Issues

### ❌ Problem: Embedding is slow

**Solutions**:
```python
# 1. Use GPU
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

# 2. Increase batch size (if memory allows)
embeddings = model.encode(documents, batch_size=64)

# 3. Use FP16
model.half()  # Convert to half precision

# 4. Use faster model for development
model = SentenceTransformer("all-MiniLM-L6-v2")  # Much faster
```

---

### ❌ Problem: Vector search is slow

**Solutions**:
```python
# 1. Use FAISS with GPU
import faiss
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# 2. Use approximate search (for large collections)
# Instead of IndexFlatL2, use IVF or HNSW
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(embeddings)
index.add(embeddings)

# 3. Set search parameters
index.nprobe = 10  # Search 10 clusters (trade-off: speed vs accuracy)
```

---

### ❌ Problem: Reranking is too slow

**Solutions**:
```python
# 1. Reduce candidates before reranking
candidates = retrieve(query, k=50)  # Get 50
top_candidates = candidates[:20]    # Rerank only top 20
reranked = rerank(query, top_candidates)

# 2. Use smaller reranker model
reranker = CrossEncoder("BAAI/bge-reranker-base")  # Not large

# 3. Batch reranking
scores = reranker.predict(pairs, batch_size=32)  # Batch processing
```

---

## RAGAS Evaluation Issues

### ❌ Error: "OpenAI API key not found"
```
AuthenticationError: No API key provided
```

**Solution**:
```python
# Use local LLM with RAGAS
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama

llm = LangchainLLMWrapper(Ollama(model="llama3.1:8b"))
result = evaluate(dataset, metrics=metrics, llm=llm)
```

---

### ❌ Error: "Metrics require certain columns"
```
ValueError: Dataset must contain 'contexts' column
```

**Solution**:
```python
from datasets import Dataset

# Ensure all required columns present
data = {
    "question": questions,
    "answer": answers,
    "contexts": [[ctx] for ctx in contexts],  # List of lists!
    "ground_truth": ground_truths  # For some metrics
}
dataset = Dataset.from_dict(data)
```

---

## 🆘 Still Stuck?

1. **Check logs**: Most errors have detailed stack traces
2. **Simplify**: Test each component (embedding, retrieval, generation) separately
3. **Restart kernel**: Clear GPU memory with fresh start
4. **Check versions**: `pip list | grep -E "langchain|chromadb|sentence"`
5. **Consult docs**:
   - [ChromaDB Docs](https://docs.trychroma.com/)
   - [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
   - [Sentence Transformers](https://www.sbert.net/)
   - [RAGAS Docs](https://docs.ragas.io/)
