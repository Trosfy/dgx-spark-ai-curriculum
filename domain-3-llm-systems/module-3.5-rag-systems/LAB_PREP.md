# Module 3.5: RAG Systems & Vector Databases - Lab Preparation

## 🎯 Purpose
This document ensures your environment is ready for all Module 3.5 labs.

## ⏱️ Setup Time: ~15 minutes

---

## 1️⃣ Environment Setup

### Option A: NGC Container (Recommended)
```bash
# Start the PyTorch container with all necessary mounts
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Option B: Local Environment
```bash
# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2️⃣ Install Dependencies

### Core RAG Libraries
```bash
pip install \
    langchain==0.3.* \
    langchain-community==0.3.* \
    langchain-huggingface==0.1.*
```

### Vector Databases
```bash
pip install \
    chromadb==0.5.* \
    faiss-cpu==1.8.* \
    qdrant-client==1.11.*
```

### Embedding & Retrieval
```bash
pip install \
    sentence-transformers==3.0.* \
    rank_bm25==0.2.*
```

### Document Processing
```bash
pip install \
    pypdf2==3.0.* \
    pdfplumber==0.11.*
```

### Evaluation
```bash
pip install \
    ragas==0.1.* \
    datasets==2.19.*
```

### LLM Client
```bash
pip install ollama==0.2.*
```

### All-in-One Command
```bash
pip install \
    langchain langchain-community langchain-huggingface \
    chromadb faiss-cpu qdrant-client \
    sentence-transformers rank_bm25 \
    pypdf2 pdfplumber \
    ragas datasets \
    ollama
```

---

## 3️⃣ Download Models

### Embedding Models (Required)
```python
from sentence_transformers import SentenceTransformer

# Download and cache embedding models
models = [
    "all-MiniLM-L6-v2",      # Fast, good for learning
    "BAAI/bge-large-en-v1.5", # High quality, production
]

for model_name in models:
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    # Test it
    emb = model.encode(["test"])
    print(f"  ✅ {model_name} - dim={emb.shape[1]}")
```

### Reranker Model (For Lab 3.5.5)
```python
from sentence_transformers import CrossEncoder

# Download reranker
reranker = CrossEncoder("BAAI/bge-reranker-large")
print("✅ Reranker model downloaded")
```

### LLM Models (via Ollama)
```bash
# Ensure Ollama is running
ollama serve &

# Pull required models
ollama pull llama3.1:8b    # Primary model for generation
ollama pull llama3.1:70b   # Optional: larger model for better quality
```

---

## 4️⃣ Verify Installation

### Run Full Verification Script
```python
#!/usr/bin/env python3
"""Module 3.5 Environment Verification"""

import sys

def check_import(name, package=None):
    """Check if a package can be imported."""
    try:
        __import__(package or name)
        print(f"✅ {name}")
        return True
    except ImportError as e:
        print(f"❌ {name}: {e}")
        return False

def main():
    print("=" * 50)
    print("Module 3.5 Environment Check")
    print("=" * 50)

    all_ok = True

    # Core libraries
    print("\n📦 Core Libraries:")
    all_ok &= check_import("LangChain", "langchain")
    all_ok &= check_import("LangChain Community", "langchain_community")
    all_ok &= check_import("Sentence Transformers", "sentence_transformers")

    # Vector databases
    print("\n🗄️ Vector Databases:")
    all_ok &= check_import("ChromaDB", "chromadb")
    all_ok &= check_import("FAISS", "faiss")
    all_ok &= check_import("Qdrant", "qdrant_client")

    # Retrieval
    print("\n🔍 Retrieval:")
    all_ok &= check_import("BM25", "rank_bm25")

    # Document processing
    print("\n📄 Document Processing:")
    all_ok &= check_import("PyPDF2", "PyPDF2")
    all_ok &= check_import("pdfplumber", "pdfplumber")

    # Evaluation
    print("\n📊 Evaluation:")
    all_ok &= check_import("RAGAS", "ragas")

    # LLM
    print("\n🤖 LLM:")
    all_ok &= check_import("Ollama", "ollama")

    # GPU check
    print("\n🖥️ GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ CUDA not available (CPU mode)")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        all_ok = False

    # Embedding model test
    print("\n🧠 Embedding Model:")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(["test"])
        print(f"✅ Embedding works: dim={emb.shape[1]}")
    except Exception as e:
        print(f"❌ Embedding: {e}")
        all_ok = False

    # ChromaDB test
    print("\n🗄️ ChromaDB Test:")
    try:
        import chromadb
        client = chromadb.Client()
        collection = client.create_collection("test")
        collection.add(documents=["test"], ids=["1"])
        results = collection.query(query_texts=["test"], n_results=1)
        print(f"✅ ChromaDB works: retrieved {len(results['documents'][0])} doc")
        client.delete_collection("test")
    except Exception as e:
        print(f"❌ ChromaDB: {e}")
        all_ok = False

    # Ollama test
    print("\n🦙 Ollama Test:")
    try:
        import ollama
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        if model_names:
            print(f"✅ Ollama running with models: {model_names[:3]}")
        else:
            print("⚠️ Ollama running but no models pulled")
    except Exception as e:
        print(f"❌ Ollama: {e}")
        print("   Run: ollama serve & ollama pull llama3.1:8b")

    # Final status
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 All checks passed! Ready for Module 3.5")
    else:
        print("⚠️ Some checks failed. Fix issues above.")
    print("=" * 50)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
```

Save and run:
```bash
python verify_module_3.5.py
```

---

## 5️⃣ Lab-Specific Setup

### Lab 3.5.1: Basic RAG
No additional setup needed.

### Lab 3.5.2: Chunking
```bash
# For document loading examples
pip install unstructured python-docx
```

### Lab 3.5.3: Vector Databases
```bash
# FAISS with GPU support (optional, for faster indexing)
pip install faiss-gpu
```

### Lab 3.5.4: Hybrid Search
No additional setup needed (rank_bm25 already installed).

### Lab 3.5.5: Reranking
Ensure reranker model is downloaded (see step 3).

### Lab 3.5.6: Evaluation
```bash
# Additional RAGAS dependencies
pip install datasets openai  # For some RAGAS metrics
```

### Lab 3.5.7: Production RAG
```bash
# Production dependencies
pip install fastapi uvicorn redis cachetools
```

---

## 6️⃣ Sample Documents (Optional)

Create test documents for labs:

```python
import os

# Create sample documents directory
os.makedirs("sample_docs", exist_ok=True)

# Sample documents about DGX Spark
documents = {
    "dgx_spark_overview.txt": """
DGX Spark is NVIDIA's desktop AI computer designed for developers, data scientists,
and students. It features a Blackwell GPU with 128GB of unified memory, making it
ideal for running large language models locally.

Key specifications:
- Grace-Blackwell GB10 processor
- 128GB unified memory (CPU + GPU shared)
- Up to 1000 TOPS AI performance
- Ubuntu Linux operating system
- Compact desktop form factor
""",
    "lora_finetuning.txt": """
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead
of updating all model parameters, LoRA adds small trainable matrices to specific
layers. This reduces memory requirements by 10-100x while maintaining quality.

Key benefits:
- Train with 8GB GPU instead of 80GB
- Multiple adapters can share one base model
- Easy to switch between specialized models
- Works with quantized models (QLoRA)
""",
    "rag_basics.txt": """
RAG (Retrieval-Augmented Generation) combines information retrieval with text
generation. When a user asks a question, the system first retrieves relevant
documents, then uses them as context for the language model.

Components:
1. Document store: Where your documents are indexed
2. Retriever: Finds relevant documents for a query
3. Generator: LLM that produces the final answer
4. Optional reranker: Improves retrieval quality
"""
}

for filename, content in documents.items():
    with open(f"sample_docs/{filename}", "w") as f:
        f.write(content.strip())
    print(f"✅ Created sample_docs/{filename}")
```

---

## ✅ Pre-Lab Checklist

Before starting each lab, verify:

- [ ] NGC container or virtual environment activated
- [ ] All dependencies installed
- [ ] Embedding models downloaded
- [ ] Ollama running with llama3.1:8b
- [ ] GPU memory available (check with `nvidia-smi`)
- [ ] Working directory has write permissions

---

## 🆘 Common Setup Issues

| Issue | Solution |
|-------|----------|
| ChromaDB sqlite3 error | `pip install pysqlite3-binary` |
| FAISS import error | Use `faiss-cpu` not `faiss` |
| Ollama connection refused | Run `ollama serve` first |
| Out of GPU memory | Restart container, use smaller model |
| Embedding model slow | First load is slow; subsequent loads are cached |

---

## ▶️ Ready to Begin?
→ Start with [QUICKSTART.md](./QUICKSTART.md) for a 5-minute demo
→ Or dive into [Lab 3.5.1](./labs/lab-3.5.1-basic-rag.ipynb) for the full experience
