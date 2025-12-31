# Module 3.5: RAG Systems & Vector Databases - Frequently Asked Questions

## Concepts & Design

### Q: When should I use RAG vs fine-tuning?
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

### Q: What chunk size should I use?
**A**: It depends on your content and queries:

| Chunk Size | Best For | Trade-off |
|------------|----------|-----------|
| 256 tokens | Precise Q&A, specific facts | May lose context |
| 512 tokens | General purpose (default) | Good balance |
| 1024 tokens | Complex topics, narrative | Less precise retrieval |

**Experiment**: Try multiple sizes and measure retrieval quality with RAGAS:
```python
for chunk_size in [256, 512, 1024]:
    results = run_rag_pipeline(chunk_size=chunk_size)
    score = evaluate_with_ragas(results)
    print(f"{chunk_size}: {score}")
```

---

### Q: Which embedding model should I choose?
**A**:

| Model | Dimensions | Speed | Quality | Best For |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Development, testing |
| bge-base-en-v1.5 | 768 | Medium | Better | Production (balanced) |
| bge-large-en-v1.5 | 1024 | Slower | Best | Production (quality) |
| nomic-embed-text | 768 | Medium | Good | Long documents (8K context) |

**For DGX Spark**: Use `bge-large-en-v1.5` for production—it fits easily in memory and provides excellent quality.

---

### Q: Which vector database should I use?
**A**:

| Database | Best For | Trade-off |
|----------|----------|-----------|
| ChromaDB | Learning, prototyping | Limited scalability |
| FAISS | GPU acceleration, large scale | No built-in persistence |
| Qdrant | Production, filtering | More complex setup |
| Milvus | Enterprise scale | Heavy infrastructure |

**For DGX Spark**: Start with ChromaDB for learning, use FAISS for GPU acceleration, consider Qdrant for production features.

---

### Q: How many documents should I retrieve (k)?
**A**: The right `k` depends on your pipeline:

**Without reranking**:
- `k=3-5`: When context window is limited
- `k=5-10`: General recommendation

**With reranking**:
- Retrieve `k=50-100`: Cast a wide net
- Rerank to `k=5-10`: Get the best ones

```python
# Two-stage retrieval
candidates = retriever.retrieve(query, k=50)  # Many candidates
reranked = reranker.rerank(query, candidates, k=5)  # Top 5 after reranking
```

---

## Hybrid Search

### Q: What is alpha in hybrid search?
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

### Q: When is BM25 better than embeddings?
**A**: BM25 (sparse search) excels at:
- **Exact terms**: Product codes, error messages, IDs
- **Rare words**: Technical jargon not well-represented in embeddings
- **Short queries**: Single-word or very short queries
- **Out-of-domain content**: Content the embedding model wasn't trained on

Use hybrid search to get the best of both.

---

## Reranking

### Q: Is reranking worth the latency?
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

### Q: How does a cross-encoder reranker work?
**A**: Unlike bi-encoders (which encode query and doc separately), cross-encoders:

1. Take query + document as a **single input**
2. Process them **together** with full attention
3. Output a **relevance score**

This is more accurate but slower (can't pre-compute document embeddings).

```python
# Bi-encoder (fast, approximate)
query_emb = model.encode(query)
doc_emb = model.encode(document)
score = cosine_similarity(query_emb, doc_emb)

# Cross-encoder (slow, accurate)
score = cross_encoder.predict([(query, document)])
```

---

## Evaluation

### Q: What RAGAS metrics should I track?
**A**: Core metrics for RAG evaluation:

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Faithfulness** | Answer based only on context | >0.9 |
| **Answer Relevancy** | Answer addresses the question | >0.8 |
| **Context Precision** | Retrieved docs are relevant | >0.7 |
| **Context Recall** | All relevant docs retrieved | >0.7 |

```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

---

### Q: How do I create a test set for RAG evaluation?
**A**: Three approaches:

**1. Manual creation** (most accurate):
```python
test_set = [
    {
        "question": "What is DGX Spark's memory?",
        "ground_truth": "DGX Spark has 128GB unified memory",
        "expected_docs": ["dgx_spark_specs.txt"]
    },
    # ... more examples
]
```

**2. LLM-generated** (faster):
```python
def generate_qa(document, llm):
    prompt = f"Generate 3 questions about this document:\n{document}"
    questions = llm.generate(prompt)
    return questions
```

**3. Synthetic with RAGAS**:
```python
from ragas.testset import TestsetGenerator
generator = TestsetGenerator()
testset = generator.generate_with_langchain_docs(documents, n=20)
```

---

## Production

### Q: How do I handle document updates?
**A**: Strategies for keeping your index current:

**1. Full rebuild** (simplest):
```python
def rebuild_index():
    collection.delete()  # Clear everything
    collection = client.create_collection("docs")
    add_all_documents(collection)
```

**2. Incremental updates** (more efficient):
```python
def update_document(doc_id, new_content):
    # Delete old
    collection.delete(ids=[doc_id])
    # Add new
    embedding = model.encode([new_content])
    collection.add(documents=[new_content], embeddings=[embedding], ids=[doc_id])
```

**3. Versioned collections**:
```python
def create_new_version():
    version = datetime.now().strftime("%Y%m%d")
    new_collection = client.create_collection(f"docs_v{version}")
    # Populate new collection
    # Switch alias when ready
```

---

### Q: How do I add caching to RAG?
**A**: Cache at multiple levels:

```python
from functools import lru_cache
import hashlib

# 1. Cache embeddings
@lru_cache(maxsize=10000)
def get_embedding(text):
    return tuple(model.encode([text])[0])

# 2. Cache retrieval results
retrieval_cache = {}
def cached_retrieve(query, k=5):
    cache_key = hashlib.md5(f"{query}_{k}".encode()).hexdigest()
    if cache_key not in retrieval_cache:
        retrieval_cache[cache_key] = retriever.retrieve(query, k)
    return retrieval_cache[cache_key]

# 3. Cache final answers (for identical queries)
@lru_cache(maxsize=1000)
def cached_rag(query):
    return rag_pipeline(query)
```

---

### Q: How do I handle multi-modal content (images, tables)?
**A**: Options for non-text content:

**1. Extract text descriptions**:
```python
# For images: Use vision model to generate descriptions
description = vision_model.describe(image)
# Index the description

# For tables: Convert to text
table_text = table.to_markdown()
```

**2. Use multi-modal embeddings**:
```python
# Models like CLIP can embed images and text in same space
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-B-32')
image_embedding = model.encode(Image.open('image.png'))
```

**3. Store metadata and retrieve later**:
```python
collection.add(
    documents=["Figure 1 shows the architecture"],
    embeddings=[...],
    metadatas=[{"type": "figure", "path": "figure1.png"}]
)
```

---

## Troubleshooting

### Q: My RAG answers are generic, not using the context
**A**: Common causes and fixes:

1. **Weak prompt template**: Make context more prominent
```python
# Instead of:
prompt = f"Context: {context}\n\nQuestion: {question}"

# Use:
prompt = f"""You are answering based on the following documents ONLY.

DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS: Base your answer ONLY on the documents above. If the answer isn't in the documents, say "Not found in documents."

ANSWER:"""
```

2. **Context too far from question**: Put context closer to where the model generates
3. **Model not following instructions**: Try a more instruction-tuned model

---

### Q: Retrieval is slow, what can I do?
**A**: Optimization strategies:

| Optimization | Improvement | Complexity |
|--------------|-------------|------------|
| Use GPU for FAISS | 10x faster | Low |
| Use approximate search (IVF, HNSW) | 10-100x faster | Medium |
| Pre-filter with metadata | Variable | Low |
| Reduce embedding dimension | 2-3x faster | Medium |
| Cache frequent queries | Huge for repeats | Low |

```python
# Example: FAISS with IVF for faster search
import faiss

nlist = 100  # Number of clusters
d = 1024     # Embedding dimension

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(embeddings)
index.add(embeddings)

# Search (faster than brute force)
index.nprobe = 10  # Search 10 clusters
D, I = index.search(query_embedding, k)
```

---

### Q: How do I debug poor retrieval?
**A**: Systematic debugging process:

```python
# 1. Check what's being retrieved
results = collection.query(query_embeddings=query_emb, n_results=10)
for i, doc in enumerate(results['documents'][0]):
    print(f"{i+1}. (score: {results['distances'][0][i]:.3f}) {doc[:100]}...")

# 2. Check if relevant doc exists in collection
all_docs = collection.get()
for doc in all_docs['documents']:
    if "expected_keyword" in doc.lower():
        print(f"Found: {doc[:100]}")

# 3. Compare embeddings directly
query_emb = model.encode(["your query"])
expected_emb = model.encode(["expected document text"])
similarity = np.dot(query_emb[0], expected_emb[0])
print(f"Direct similarity: {similarity:.3f}")

# 4. Test with exact document text
exact_result = collection.query(query_embeddings=expected_emb.tolist(), n_results=5)
print("Results when querying with expected doc embedding:", exact_result)
```

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for common patterns
- See [ELI5.md](./ELI5.md) for concept explanations
- Consult documentation:
  - [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/)
  - [ChromaDB Docs](https://docs.trychroma.com/)
  - [RAGAS Docs](https://docs.ragas.io/)
