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
    model="qwen3:8b",
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

llm = LangchainLLMWrapper(Ollama(model="qwen3:8b"))
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

## ❓ Frequently Asked Questions

**Q: When should I use RAG vs fine-tuning?**

**A**: Use this decision framework:

| Scenario | RAG | Fine-tuning |
|----------|-----|-------------|
| Frequently changing data | ✅ Best | ❌ Expensive to retrain |
| Need source citations | ✅ Natural | ❌ Hard to add |
| Proprietary knowledge | ✅ Easy | ✅ Works too |
| Style/tone changes | ❌ Limited | ✅ Better |
| Domain vocabulary | ⚠️ Partial | ✅ Better |
| Reduce hallucination | ✅ Grounded | ⚠️ Can still hallucinate |

**Default recommendation**: Start with RAG. It's faster to implement, easier to update, and provides citations. Add fine-tuning later if needed for style or specialized vocabulary.

---

**Q: What chunk size should I use?**

**A**: It depends on your content and queries:

| Chunk Size | Best For | Trade-off |
|------------|----------|-----------|
| 256 tokens | Precise Q&A, specific facts | May lose context |
| 512 tokens | General purpose (default) | Good balance |
| 1024 tokens | Complex topics, narrative | Less precise retrieval |

**Experiment**: Try multiple sizes and measure retrieval quality with RAGAS.

---

**Q: Which embedding model should I choose?**

**A**:

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Development, testing |
| bge-base-en-v1.5 | 768 | Medium | Better | Production (balanced) |
| bge-large-en-v1.5 | 1024 | Slower | Best | Production (quality) |
| qwen3-embedding:8b | 768 | Medium | Good | Long documents (8K context) |

**For DGX Spark**: Use `bge-large-en-v1.5` for production—it fits easily in memory and provides excellent quality.

---

**Q: Which vector database should I use?**

**A**:

| Database | Best For | Trade-off |
|----------|----------|-----------|
| ChromaDB | Learning, prototyping | Limited scalability |
| FAISS | GPU acceleration, large scale | No built-in persistence |
| Qdrant | Production, filtering | More complex setup |
| Milvus | Enterprise scale | Heavy infrastructure |

**For DGX Spark**: Start with ChromaDB for learning, use FAISS for GPU acceleration, consider Qdrant for production features.

---

**Q: How many documents should I retrieve (k)?**

**A**: The right `k` depends on your pipeline:

**Without reranking**:
- `k=3-5`: When context window is limited
- `k=5-10`: General recommendation

**With reranking**:
- Retrieve `k=50-100`: Cast a wide net
- Rerank to `k=5-10`: Get the best ones

---

**Q: What is alpha in hybrid search?**

**A**: Alpha controls the balance between dense (embedding) and sparse (keyword) search:

```
final_score = alpha * dense_score + (1 - alpha) * sparse_score
```

| Alpha | Emphasis | Best For |
|-------|----------|----------|
| 1.0 | Pure dense | Semantic search only |
| 0.7 | Mostly dense | General use (default) |
| 0.5 | Balanced | Mixed content |
| 0.3 | Mostly sparse | Technical docs with specific terms |
| 0.0 | Pure sparse | Exact keyword matching |

**Tip**: Tune alpha on a validation set of queries with known relevant documents.

---

**Q: When is BM25 better than embeddings?**

**A**: BM25 (sparse search) excels at:
- **Exact terms**: Product codes, error messages, IDs
- **Rare words**: Technical jargon not well-represented in embeddings
- **Short queries**: Single-word or very short queries
- **Out-of-domain content**: Content the embedding model wasn't trained on

Use hybrid search to get the best of both.

---

**Q: Is reranking worth the latency?**

**A**: Usually yes, but measure it:

| Scenario | Latency Impact | Quality Improvement |
|----------|----------------|---------------------|
| Simple queries | +50-100ms | Marginal |
| Complex queries | +50-100ms | Significant (10-30%) |
| Noisy retrieval | +50-100ms | Very significant (20-50%) |

**Recommendation**: Add reranking if:
- Your first-stage retrieval is noisy
- Precision matters more than latency
- You're using hybrid search (reranker helps combine)

---

**Q: How does a cross-encoder reranker work?**

**A**: Unlike bi-encoders (which encode query and doc separately), cross-encoders:

1. Take query + document as a **single input**
2. Process them **together** with full attention
3. Output a **relevance score**

This is more accurate but slower (can't pre-compute document embeddings).

---

**Q: What RAGAS metrics should I track?**

**A**: Core metrics for RAG evaluation:

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Faithfulness** | Answer based only on context | >0.9 |
| **Answer Relevancy** | Answer addresses the question | >0.8 |
| **Context Precision** | Retrieved docs are relevant | >0.7 |
| **Context Recall** | All relevant docs retrieved | >0.7 |

---

**Q: How do I create a test set for RAG evaluation?**

**A**: Three approaches:

**1. Manual creation** (most accurate): Create questions with ground truth answers and expected documents

**2. LLM-generated** (faster): Have an LLM generate questions from your documents

**3. Synthetic with RAGAS**: Use RAGAS TestsetGenerator to create test sets automatically

---

**Q: How do I handle document updates?**

**A**: Strategies for keeping your index current:

**1. Full rebuild** (simplest): Delete and recreate entire collection

**2. Incremental updates** (more efficient): Delete old versions and add updated documents

**3. Versioned collections**: Create new collection versions and switch when ready

---

**Q: How do I add caching to RAG?**

**A**: Cache at multiple levels:
1. Cache embeddings for repeated text
2. Cache retrieval results for identical queries
3. Cache final answers for frequently asked questions

---

**Q: How do I handle multi-modal content (images, tables)?**

**A**: Options for non-text content:

**1. Extract text descriptions**: Use vision models or convert tables to markdown

**2. Use multi-modal embeddings**: Models like CLIP can embed images and text in same space

**3. Store metadata and retrieve later**: Index descriptions but keep references to original content

---

**Q: My RAG answers are generic, not using the context**

**A**: Common causes and fixes:

1. **Weak prompt template**: Make context more prominent and add explicit instructions to use only the provided context
2. **Context too far from question**: Put context closer to where the model generates
3. **Model not following instructions**: Try a more instruction-tuned model

---

**Q: Retrieval is slow, what can I do?**

**A**: Optimization strategies:

| Optimization | Improvement | Complexity |
|--------------|-------------|------------|
| Use GPU for FAISS | 10x faster | Low |
| Use approximate search (IVF, HNSW) | 10-100x faster | Medium |
| Pre-filter with metadata | Variable | Low |
| Reduce embedding dimension | 2-3x faster | Medium |
| Cache frequent queries | Huge for repeats | Low |

---

**Q: How do I debug poor retrieval?**

**A**: Systematic debugging process:

1. Check what's being retrieved and their scores
2. Check if relevant doc exists in collection
3. Compare embeddings directly
4. Test with exact document text

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
