# Module 13: AI Agents & Agentic Systems

**Phase:** 3 - Advanced  
**Duration:** Weeks 22-23 (12-15 hours)  
**Prerequisites:** Module 12 (Deployment)

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
| 13.1 | Implement RAG pipeline with vector database | Apply |
| 13.2 | Create agents with custom tools | Apply |
| 13.3 | Design multi-agent architectures | Create |
| 13.4 | Evaluate agent performance and reliability | Evaluate |

---

## Topics

### 13.1 RAG Fundamentals
- Document loading and chunking
- Embedding models
- Vector databases (ChromaDB, FAISS)
- Retrieval strategies (dense, sparse, hybrid)

### 13.2 LangChain Framework
- Chains and composition
- Agents and tools
- Memory systems
- Callbacks and tracing

### 13.3 LlamaIndex
- Index types
- Query engines
- Response synthesis

### 13.4 LangGraph
- Stateful agents
- Graph-based orchestration
- Human-in-the-loop

### 13.5 Multi-Agent Systems
- Agent communication
- CrewAI framework
- Task decomposition

---

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 13.1 | RAG Pipeline | 3h | Complete RAG with ChromaDB |
| 13.2 | Custom Tools | 3h | 5 tools: search, calc, code, file, API |
| 13.3 | LlamaIndex Query Engine | 2h | Hybrid search with reranking |
| 13.4 | LangGraph Workflow | 3h | Multi-step with human approval |
| 13.5 | Multi-Agent System | 3h | 3-agent content generation team |
| 13.6 | Agent Benchmark | 2h | Evaluation framework |

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

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)
