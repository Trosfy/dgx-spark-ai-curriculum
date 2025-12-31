# Module 3.5: RAG Systems & Vector Databases - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Create a basic RAG pipeline that retrieves and answers questions from your own documents.

## ✅ Before You Start
- [ ] DGX Spark NGC container running
- [ ] Ollama running with a model

## 🚀 Let's Go!

### Step 1: Install Dependencies
```bash
pip install chromadb sentence-transformers ollama
```

### Step 2: Create Some Documents
```python
documents = [
    "DGX Spark has 128GB unified memory and a Blackwell GPU.",
    "LoRA fine-tuning trains only 0.1% of model parameters.",
    "Quantization reduces model memory by 3-4x with minimal quality loss.",
    "RAG retrieves relevant documents before generating answers.",
    "Vector databases store embeddings for semantic search."
]
```

### Step 3: Build the Index
```python
import chromadb
from sentence_transformers import SentenceTransformer

# Create embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create vector database
client = chromadb.Client()
collection = client.create_collection("quickstart")

# Add documents
embeddings = embed_model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"✅ Indexed {len(documents)} documents")
```

### Step 4: Query with RAG
```python
import ollama

def rag_query(question):
    # Retrieve relevant documents
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=2)

    context = "\n".join(results['documents'][0])

    # Generate answer
    prompt = f"""Answer based on this context:

{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Test it!
answer = rag_query("How much memory does DGX Spark have?")
print(answer)
```

**Expected output**:
```
DGX Spark has 128GB of unified memory, along with a Blackwell GPU.
```

## 🎉 You Did It!

You just built a working RAG pipeline! The model answered using your documents, not just its training data. This same pattern scales to:
- **Thousands of documents** with FAISS GPU acceleration
- **Complex queries** with hybrid search (dense + sparse)
- **Better quality** with reranking
- **Production use** with Qdrant or similar

## ▶️ Next Steps
1. **Add more documents**: See [Lab 3.5.1](./labs/lab-3.5.1-basic-rag.ipynb)
2. **Try different chunking**: See [Lab 3.5.2](./labs/lab-3.5.2-chunking.ipynb)
3. **Full setup**: Start with [LAB_PREP.md](./LAB_PREP.md)
