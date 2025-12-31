# Module 3.5: RAG Systems & Vector Databases - Prerequisites

## 📋 Required Before Starting

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 8GB | 16GB+ (for large embeddings + reranker) |
| System RAM | 16GB | 32GB (for large document collections) |
| Storage | 20GB free | 50GB (for vector database indexes) |

### Software Requirements
- [ ] Python 3.10+
- [ ] CUDA 12.0+
- [ ] NGC PyTorch container or local PyTorch with CUDA
- [ ] Ollama installed with at least one model

### Python Package Dependencies
```bash
pip install \
    langchain langchain-community langchain-huggingface \
    chromadb faiss-cpu qdrant-client \
    sentence-transformers \
    rank_bm25 ragas \
    pypdf2 pdfplumber \
    ollama
```

---

## ✅ Knowledge Prerequisites

### Must Have (Essential)

#### 1. Python Programming
**Why needed**: All RAG implementations use Python
**Self-check**: Can you write a class with methods and use list comprehensions?

```python
# You should understand this code
class DocumentProcessor:
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def process(self, documents):
        return [self._chunk(doc) for doc in documents]

    def _chunk(self, doc):
        # Split document into chunks
        pass
```

**If not ready**: Review Python OOP basics first.

#### 2. Basic LLM Usage
**Why needed**: RAG augments LLM generation
**Self-check**: Have you called an LLM API (Ollama, OpenAI, etc.)?

```python
# You should be comfortable with this pattern
import ollama

response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response['message']['content'])
```

**If not ready**: Complete Module 3.3 labs first.

#### 3. NumPy Array Operations
**Why needed**: Embeddings are NumPy arrays
**Self-check**: Can you do basic array operations?

```python
import numpy as np

# You should understand these operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)           # Similarity
normalized = a / np.linalg.norm(a)   # Normalization
sorted_idx = np.argsort(a)[::-1]     # Sorting by value
```

**If not ready**: Review NumPy basics (30 minutes).

---

### Should Have (Helpful)

#### 4. Understanding of Semantic Similarity
**Why helpful**: Core concept behind embedding-based retrieval
**Self-check**: Do you understand why "car" and "automobile" should have similar embeddings?

**Quick primer**:
- Similar meanings → similar vector representations
- Cosine similarity measures how "aligned" two vectors are
- High similarity (close to 1) = semantically related

#### 5. Basic Database Concepts
**Why helpful**: Vector databases extend traditional DB concepts
**Self-check**: Do you understand indexes, queries, and CRUD operations?

**Quick primer if needed**:
- **Index**: Data structure for fast lookups
- **Query**: Request for specific data
- **CRUD**: Create, Read, Update, Delete operations

#### 6. JSON and API Basics
**Why helpful**: RAG systems often expose REST APIs
**Self-check**: Can you parse JSON and make HTTP requests?

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/query",
    json={"question": "What is RAG?"}
)
data = response.json()
```

---

### Nice to Have (Bonus)

#### 7. LangChain Experience
**Why helpful**: Labs use LangChain for document loading/splitting
**Self-check**: Have you used LangChain before?

**If not**: Don't worry—labs introduce it step by step.

#### 8. Information Retrieval Concepts
**Why helpful**: Understanding precision, recall, and ranking
**Self-check**: Do you know what TF-IDF or BM25 are?

**If not**: Lab 3.5.4 covers BM25 from scratch.

---

## 🧪 Quick Self-Assessment

### Practical Check (5 minutes)
Run this code. If it works, you're ready:

```python
# Prerequisites test
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded")

# 2. Create embeddings
texts = ["Hello world", "Hi there"]
embeddings = model.encode(texts)
print(f"✅ Created embeddings with shape {embeddings.shape}")

# 3. Compute similarity
similarity = np.dot(embeddings[0], embeddings[1])
print(f"✅ Similarity between texts: {similarity:.3f}")

# 4. Test Ollama
import ollama
response = ollama.chat(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Say 'ready' if you can hear me."}]
)
print(f"✅ Ollama response: {response['message']['content'][:50]}")

print("\n🎉 All prerequisites met! You're ready for Module 3.5")
```

### Knowledge Check
Answer these questions (mentally):

1. **Embeddings**: What type of data structure represents an embedding? (Answer: Array/vector of floats)
2. **Similarity**: Higher cosine similarity means texts are more ___? (Answer: Similar)
3. **RAG Purpose**: Why retrieve documents before generating? (Answer: Provide accurate, up-to-date context)

---

## 📚 Gap-Filling Resources

| Gap | Resource | Time |
|-----|----------|------|
| Python OOP | [Real Python Classes](https://realpython.com/python3-object-oriented-programming/) | 2 hr |
| NumPy basics | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) | 1 hr |
| LLM basics | Module 3.3 Labs | 4 hr |
| Semantic search intro | [Sentence Transformers docs](https://www.sbert.net/) | 1 hr |

---

## ⏭️ Ready to Start?

If you've:
- ✅ Passed the practical check
- ✅ Can answer the knowledge check questions
- ✅ Have the required software installed

→ Proceed to [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ Or jump to [QUICKSTART.md](./QUICKSTART.md) for a 5-minute RAG demo
