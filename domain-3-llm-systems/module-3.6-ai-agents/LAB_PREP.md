# Module 3.6: AI Agents & Agentic Systems - Lab Preparation

## 🎯 Purpose
This document ensures your environment is ready for all Module 3.6 labs.

## ⏱️ Setup Time: ~10 minutes

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

### Option B: Local Environment (x86_64 only)
```bash
# ⚠️ WARNING: This option is for x86_64 systems only.
# DGX Spark uses ARM64/aarch64 architecture - always use Option A (NGC container)
# for proper PyTorch support. NEVER use pip install for PyTorch on ARM64.

# For x86_64 development machines only:
python -m venv agents-env
source agents-env/bin/activate
```

---

## 2️⃣ Install Dependencies

### Core Agent Libraries
```bash
pip install \
    langchain==0.3.* \
    langchain-community==0.3.* \
    langgraph==0.2.*
```

### Multi-Agent Frameworks
```bash
pip install \
    crewai==0.67.* \
    autogen==0.3.*
```

### Supporting Libraries
```bash
pip install \
    ollama==0.2.* \
    chromadb==0.5.* \
    sentence-transformers==3.0.*
```

### All-in-One Command
```bash
pip install \
    langchain langchain-community langgraph \
    crewai \
    ollama chromadb sentence-transformers
```

---

## 3️⃣ Ollama Setup

### Start Ollama Service
```bash
# Start Ollama (in a separate terminal or background)
ollama serve &
```

### Pull Required Models (2025)
```bash
# For development (faster with function calling)
ollama pull qwen3:8b              # Hybrid thinking, 0.971 F1 BFCL

# For production quality agents
ollama pull qwen3:32b             # Best quality (~20GB)
ollama pull qwq:32b               # Extended reasoning for complex tasks

# For embeddings (if using RAG tools)
ollama pull qwen3-embedding:8b    # #1 MTEB multilingual
```

### Verify Ollama via Ollama Web UI
```bash
# Check available models via Ollama Web UI API
curl http://localhost:11434/api/tags

# Test generation via Ollama Web UI API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3:8b",
  "prompt": "Hello!",
  "stream": false
}'

# Ollama Web UI endpoint: http://localhost:11434
```

---

## 4️⃣ Verify Installation

### Run Full Verification Script
```python
#!/usr/bin/env python3
"""Module 3.6 Environment Verification"""

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
    print("Module 3.6 Environment Check")
    print("=" * 50)

    all_ok = True

    # Core libraries
    print("\n📦 Core Libraries:")
    all_ok &= check_import("LangChain", "langchain")
    all_ok &= check_import("LangChain Community", "langchain_community")
    all_ok &= check_import("LangGraph", "langgraph")

    # Multi-agent frameworks
    print("\n🤖 Multi-Agent Frameworks:")
    all_ok &= check_import("CrewAI", "crewai")

    # Supporting libraries
    print("\n🔧 Supporting Libraries:")
    all_ok &= check_import("Ollama", "ollama")
    all_ok &= check_import("ChromaDB", "chromadb")

    # GPU check
    print("\n🖥️ GPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   Memory: {mem_gb:.1f} GB")
        else:
            print("⚠️ CUDA not available (CPU mode)")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        all_ok = False

    # Ollama test
    print("\n🦙 Ollama Test:")
    try:
        import ollama
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        if model_names:
            print(f"✅ Ollama running with models: {model_names[:3]}")
            if any('32b' in m for m in model_names):
                print("   ✅ 32B model available (recommended for agents)")
            else:
                print("   ⚠️ Consider: ollama pull qwen3:32b")
        else:
            print("⚠️ Ollama running but no models pulled")
            print("   Run: ollama pull qwen3:8b")
    except Exception as e:
        print(f"❌ Ollama: {e}")
        print("   Run: ollama serve & ollama pull qwen3:8b")

    # LangChain agent test
    print("\n🔗 LangChain Agent Test:")
    try:
        from langchain.tools import tool
        from langchain.agents import AgentExecutor, create_react_agent

        @tool
        def test_tool(x: str) -> str:
            """Test tool."""
            return f"Got: {x}"

        print("✅ LangChain agents work")
    except Exception as e:
        print(f"❌ LangChain agents: {e}")
        all_ok = False

    # LangGraph test
    print("\n📊 LangGraph Test:")
    try:
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class TestState(TypedDict):
            value: int

        graph = StateGraph(TestState)
        print("✅ LangGraph works")
    except Exception as e:
        print(f"❌ LangGraph: {e}")
        all_ok = False

    # Final status
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 All checks passed! Ready for Module 3.6")
    else:
        print("⚠️ Some checks failed. Fix issues above.")
    print("=" * 50)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
```

Save and run:
```bash
python verify_module_3.6.py
```

---

## 5️⃣ Lab-Specific Setup

### Lab 3.6.1: Custom Tools
No additional setup needed.

### Lab 3.6.2: ReAct Agent
```bash
# Optional: LangSmith for tracing (free tier available)
pip install langsmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key  # From smith.langchain.com
```

### Lab 3.6.3: LangGraph Workflow
No additional setup needed.

### Lab 3.6.4: Multi-Agent System
```bash
# Ensure ChromaDB for shared memory
pip install chromadb
```

### Lab 3.6.5: CrewAI Project
```bash
# CrewAI tools (optional)
pip install 'crewai[tools]'
```

### Lab 3.6.6: Agent Benchmark
```bash
# Evaluation utilities
pip install pandas matplotlib tqdm
```

---

## 6️⃣ Sample Tools (Optional)

Create reusable tools for labs:

```python
# Save as tools/common_tools.py

from langchain.tools import tool
from typing import Optional
import json

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A valid Python math expression

    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation (basic operations only)
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "Error: Only basic math operations allowed"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current datetime as a string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read

    Returns:
        File contents or error message
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        if len(content) > 5000:
            return content[:5000] + "\n... (truncated)"
        return content
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write

    Returns:
        Success or error message
    """
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def search_web(query: str) -> str:
    """Search the web for information (mock implementation).

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Mock implementation - replace with actual search API
    return f"Mock search results for: {query}"

# Export all tools
ALL_TOOLS = [calculate, get_current_time, read_file, write_file, search_web]

if __name__ == "__main__":
    # Test tools
    print(calculate.invoke("2 + 2"))
    print(get_current_time.invoke(""))
```

---

## ✅ Pre-Lab Checklist

Before starting each lab, verify:

- [ ] NGC container or virtual environment activated
- [ ] All dependencies installed
- [ ] Ollama running with qwen3:8b or qwen3:32b (verify via Ollama Web UI at http://localhost:11434)
- [ ] GPU memory available (check with `nvidia-smi`)
- [ ] Working directory has write permissions

---

## 🆘 Common Setup Issues

| Issue | Solution |
|-------|----------|
| Ollama connection refused | Run `ollama serve` first |
| LangChain import errors | Ensure `langchain-community` is installed |
| LangGraph not found | `pip install langgraph` |
| CrewAI errors | Check Python version (3.10+) |
| Agent loops forever | Add `max_iterations` to AgentExecutor |
| Out of GPU memory | Use 8B model instead of 32B |

---

## ▶️ Ready to Begin?
→ Start with [QUICKSTART.md](./QUICKSTART.md) for a 5-minute demo
→ Or dive into [Lab 3.6.1](./labs/lab-3.6.1-custom-tools.ipynb) for the full experience
