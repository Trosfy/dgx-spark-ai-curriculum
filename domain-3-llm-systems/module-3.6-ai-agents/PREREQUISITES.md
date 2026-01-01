# Module 3.6: AI Agents & Agentic Systems - Prerequisites

## 📋 Required Before Starting

### Hardware Requirements (DGX Spark)
| Component | DGX Spark Spec | Notes |
|-----------|----------------|-------|
| GPU | NVIDIA Blackwell GB10 Superchip | 6,144 CUDA cores, 192 Tensor Cores |
| Memory | 128GB unified LPDDR5X | Sufficient for 70B+ models |
| Storage | 10GB free | For models and dependencies |

**Note:** DGX Spark's 128GB unified memory enables running large models (70B+) that require significant memory for agent workloads.

### Software Requirements
- [ ] Python 3.10+
- [ ] CUDA 12.0+
- [ ] NGC PyTorch container or local PyTorch with CUDA
- [ ] Ollama installed with llama3.1:8b or 70b

### Python Package Dependencies
```bash
pip install \
    langchain langchain-community \
    langgraph \
    crewai \
    ollama \
    chromadb sentence-transformers
```

---

## ✅ Knowledge Prerequisites

### Must Have (Essential)

#### 1. Python Functions and Decorators
**Why needed**: Tools are defined using Python decorators
**Self-check**: Can you write and use decorators?

```python
# You should understand this code
from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

# Using decorators
result = greet("Alice")  # Prints "Calling greet", returns "Hello, Alice!"
```

**If not ready**: Review Python decorators (30 minutes).

#### 2. Basic LLM Usage
**Why needed**: Agents are built on top of LLMs
**Self-check**: Have you used Ollama or another LLM API?

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

#### 3. LangChain Basics
**Why needed**: Agents are built with LangChain
**Self-check**: Have you used LangChain chains or prompts?

```python
# You should understand this pattern
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1:8b")
prompt = PromptTemplate.from_template("Tell me about {topic}")
chain = prompt | llm
result = chain.invoke({"topic": "AI agents"})
```

**If not ready**: Review LangChain quickstart (1 hour).

---

### Should Have (Helpful)

#### 4. Understanding of RAG Systems
**Why helpful**: Agents often use RAG as a tool
**Self-check**: Did you complete Module 3.5?

**Quick primer if not**:
- RAG = Retrieval-Augmented Generation
- Agents can use retrievers as tools
- Vector databases store searchable embeddings

#### 5. JSON and API Basics
**Why helpful**: Tool inputs/outputs use structured data
**Self-check**: Can you parse JSON and work with APIs?

```python
import json
import requests

# Parse JSON
data = json.loads('{"name": "tool", "params": {"x": 1}}')

# Make API call
response = requests.get("http://api.example.com/data")
result = response.json()
```

#### 6. Error Handling
**Why helpful**: Agents need robust error handling
**Self-check**: Are you comfortable with try/except?

```python
try:
    result = risky_operation()
except ValueError as e:
    result = fallback_value
except Exception as e:
    log_error(e)
    raise
```

---

### Nice to Have (Bonus)

#### 7. Graph Concepts
**Why helpful**: LangGraph uses graph structures
**Self-check**: Do you understand nodes, edges, and graph traversal?

**Quick primer**:
- **Node**: A step in the workflow
- **Edge**: Connection between steps
- **Conditional edge**: Different paths based on conditions

#### 8. Async/Await Python
**Why helpful**: Some agent frameworks use async
**Self-check**: Can you write async functions?

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

# Running async code
result = asyncio.run(fetch_data())
```

**If not familiar**: Labs work without async, but it's useful for production.

---

## 🧪 Quick Self-Assessment

### Practical Check (5 minutes)
Run this code. If it works, you're ready:

```python
#!/usr/bin/env python3
"""Module 3.6 Prerequisites Check"""

# 1. Check decorators understanding
from functools import wraps

def tool_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@tool_decorator
def sample_tool(x: int) -> str:
    return f"Result: {x * 2}"

assert sample_tool(5) == "Result: 10"
print("✅ Python decorators work")

# 2. Check Ollama
import ollama
try:
    models = ollama.list()
    model_names = [m['name'] for m in models.get('models', [])]
    print(f"✅ Ollama available with models: {model_names[:3]}")
except Exception as e:
    print(f"⚠️ Ollama not running: {e}")
    print("   Run: ollama serve & ollama pull llama3.1:8b")

# 3. Check LangChain
try:
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Ollama as LCOllama
    print("✅ LangChain imports work")
except ImportError as e:
    print(f"❌ LangChain missing: {e}")

# 4. Check LangGraph
try:
    from langgraph.graph import StateGraph
    print("✅ LangGraph imports work")
except ImportError as e:
    print(f"❌ LangGraph missing: pip install langgraph")

# 5. Check CrewAI
try:
    from crewai import Agent, Task, Crew
    print("✅ CrewAI imports work")
except ImportError as e:
    print(f"⚠️ CrewAI not installed: pip install crewai")

print("\n🎉 Prerequisites check complete!")
```

### Knowledge Check
Answer these questions (mentally):

1. **ReAct**: What does the Think → Act → Observe loop do? (Answer: Reason about action, take it, see result)
2. **Tools**: Why do agents need tools? (Answer: To take actions beyond text generation)
3. **Multi-agent**: Why use multiple agents? (Answer: Specialization, parallel work)

---

## 📚 Gap-Filling Resources

| Gap | Resource | Time |
|-----|----------|------|
| Python decorators | [Real Python Decorators](https://realpython.com/primer-on-python-decorators/) | 1 hr |
| LangChain basics | [LangChain Quickstart](https://python.langchain.com/docs/get_started/quickstart) | 1 hr |
| RAG fundamentals | Module 3.5 | 4-8 hr |
| Async Python | [Real Python Async](https://realpython.com/async-io-python/) | 2 hr |

---

## ⏭️ Ready to Start?

If you've:
- ✅ Passed the practical check
- ✅ Can answer the knowledge check questions
- ✅ Completed Module 3.5 (RAG Systems)

→ Proceed to [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ Or jump to [QUICKSTART.md](./QUICKSTART.md) for a 5-minute agent demo
