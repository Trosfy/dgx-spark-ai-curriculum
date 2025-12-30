# Module 3.4: AI Agents & Agentic Systems

**Domain:** 3 - LLM Systems  
**Duration:** Weeks 22-23 (12-15 hours)  
**Prerequisites:** Module 3.3 (Deployment)

---

## Overview

Build intelligent AI agents that can use tools, retrieve information, and collaborate with other agents. This module covers RAG systems, LangChain, LlamaIndex, and multi-agent architectures.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Build AI agents using LangChain and LlamaIndex
- ✅ Implement Retrieval-Augmented Generation (RAG) systems
- ✅ Create multi-agent systems for complex tasks
- ✅ Design and implement tool-using agents

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.4.1 | Implement RAG pipeline with vector database | Apply |
| 3.4.2 | Create agents with custom tools | Apply |
| 3.4.3 | Design multi-agent architectures | Create |
| 3.4.4 | Evaluate agent performance and reliability | Evaluate |

---

## Topics

### 3.4.1 RAG Fundamentals
- Document loading and chunking
- Embedding models
- Vector databases (ChromaDB, FAISS)
- Retrieval strategies (dense, sparse, hybrid)

### 3.4.2 LangChain Framework
- Chains and composition
- Agents and tools
- Memory systems
- Callbacks and tracing

### 3.4.3 LlamaIndex
- Index types
- Query engines
- Response synthesis

### 3.4.4 LangGraph
- Stateful agents
- Graph-based orchestration
- Human-in-the-loop

### 3.4.5 Multi-Agent Systems
- Agent communication
- CrewAI framework
- Task decomposition

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 3.4.1 | RAG Pipeline | 3h | Complete RAG with ChromaDB |
| 3.4.2 | Custom Tools | 3h | 5 tools: search, calc, code, file, API |
| 3.4.3 | LlamaIndex Query Engine | 2h | Hybrid search with reranking |
| 3.4.4 | LangGraph Workflow | 3h | Multi-step with human approval |
| 3.4.5 | Multi-Agent System | 3h | 3-agent content generation team |
| 3.4.6 | Agent Benchmark | 2h | Evaluation framework |

---

## Guidance

### Local Agent Stack on DGX Spark

```python
# Embeddings: Run locally via Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# LLM: 70B model for best agent performance
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.1:70b")

# Vector DB: ChromaDB locally
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
```

### RAG Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### Custom Tool

```python
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
```

---

## Milestone Checklist

- [ ] RAG pipeline working with technical docs
- [ ] 5 custom tools implemented
- [ ] LlamaIndex query engine with citations
- [ ] LangGraph workflow with human-in-the-loop
- [ ] Multi-agent content generation system
- [ ] Agent evaluation framework

---

## DGX Spark Setup

### NGC Container Launch

```bash
# Start NGC container with all required flags
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Install Agent Dependencies

```bash
# Inside the NGC container
pip install langchain langchain-community chromadb llama-index \
    llama-index-llms-ollama llama-index-embeddings-ollama \
    langgraph rank_bm25 sentence-transformers
```

### Ollama Setup

```bash
# Start Ollama server (in a separate terminal)
ollama serve

# Pull required models
ollama pull llama3.1:8b          # Fast responses for development
ollama pull llama3.1:70b         # Best quality for production
ollama pull nomic-embed-text     # Local embeddings
```

### Verify Setup

```python
# Run this in a notebook to verify everything is working
import requests

def verify_setup():
    """Verify DGX Spark agent environment is ready."""
    checks = []

    # Check Ollama
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            checks.append(f"✅ Ollama running with models: {', '.join(models[:3])}")
        else:
            checks.append("❌ Ollama not responding properly")
    except:
        checks.append("❌ Ollama not running - start with: ollama serve")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            checks.append(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            checks.append("⚠️ No GPU detected")
    except:
        checks.append("⚠️ PyTorch not available")

    print("\n".join(checks))
    return all("✅" in c for c in checks)

verify_setup()
```

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)
