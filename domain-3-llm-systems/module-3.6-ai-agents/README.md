# Module 3.6: AI Agents & Agentic Systems

**Domain:** 3 - LLM Systems
**Duration:** Week 26 (10-12 hours)
**Prerequisites:** Module 3.5 (RAG Systems)

---

## Overview

Build intelligent AI agents that can use tools, reason about tasks, and collaborate with other agents. This module builds on your RAG knowledge from Module 3.5 to create sophisticated agentic systems using LangChain, LangGraph, and multi-agent frameworks.

**Note:** RAG fundamentals are covered in [Module 3.5](../module-3.5-rag-systems/). This module focuses on agent architectures and tool use.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Build AI agents using LangChain and LangGraph
- ✅ Create custom tools for agent capabilities
- ✅ Design multi-agent systems for complex tasks
- ✅ Evaluate agent performance and reliability

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.6.1 | Create agents with custom tools | Apply |
| 3.6.2 | Design multi-agent architectures | Create |
| 3.6.3 | Implement stateful agent workflows with LangGraph | Apply |
| 3.6.4 | Evaluate agent performance and reliability | Evaluate |

---

## Topics

### 3.6.1 Agent Fundamentals

- **ReAct Pattern (Reasoning + Acting)**
  - Think → Act → Observe loop
  - Tool selection and execution

- **Agent Types**
  - Zero-shot agents
  - Conversational agents
  - Structured output agents

- **Tool Use**
  - Function calling with LLMs
  - Tool definition and schema
  - Error handling and retry

### 3.6.2 LangChain Framework

- Chains and composition
- Agents and tools
- Memory systems (conversation, summary, entity)
- Callbacks and tracing

### 3.6.3 LangGraph

- **Stateful Agents**
  - State management in graphs
  - Conditional branching

- **Graph-Based Orchestration**
  - Node and edge definitions
  - Subgraphs and composition

- **Human-in-the-Loop**
  - Breakpoints and approval flows
  - Interactive corrections

### 3.6.4 Multi-Agent Systems

- **Agent Communication**
  - Message passing
  - Shared memory

- **CrewAI Framework**
  - Role-based agents
  - Task delegation

- **Autogen Patterns**
  - Conversation patterns
  - Group chat architectures

### 3.6.5 Advanced Patterns

- **Tool Chains**
  - Sequential tool execution
  - Parallel tool calls

- **Memory and Context**
  - Short-term vs long-term memory
  - Context window management

- **Error Recovery**
  - Retry strategies
  - Fallback agents

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 3.6.1 | Custom Tools | 2h | 5 tools: search, calc, code, file, API |
| 3.6.2 | ReAct Agent | 2h | Agent with tool selection and reasoning |
| 3.6.3 | LangGraph Workflow | 2h | Multi-step with human approval |
| 3.6.4 | Multi-Agent System | 2h | 3-agent content generation team |
| 3.6.5 | CrewAI Project | 2h | Role-based agent team |
| 3.6.6 | Agent Benchmark | 2h | Evaluation framework |

---

## Guidance

### Local Agent Stack on DGX Spark

```python
# LLM: 70B model for best agent performance
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.1:70b")

# For tool calling, use the chat model
from langchain_community.chat_models import ChatOllama
chat = ChatOllama(model="llama3.1:70b")
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

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implement with your preferred search API
    return f"Search results for: {query}"

@tool
def read_file(path: str) -> str:
    """Read contents of a file."""
    with open(path, 'r') as f:
        return f.read()

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    with open(path, 'w') as f:
        f.write(content)
    return f"Written to {path}"

@tool
def run_code(code: str) -> str:
    """Execute Python code safely."""
    # Use sandboxed execution in production
    try:
        exec(code)
        return "Code executed successfully"
    except Exception as e:
        return f"Error: {e}"
```

### ReAct Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Get the ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create tools list
tools = [calculate, web_search, read_file, write_file]

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
result = agent_executor.invoke({"input": "What is 25 * 47?"})
```

### LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AgentState(TypedDict):
    messages: list
    next_action: str
    human_approved: bool

def should_continue(state):
    if state["human_approved"]:
        return "execute"
    return "await_approval"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_step)
workflow.add_node("await_approval", await_human_approval)
workflow.add_node("execute", execute_step)

workflow.add_edge("plan", "await_approval")
workflow.add_conditional_edges("await_approval", should_continue)
workflow.add_edge("execute", END)

app = workflow.compile()
```

### Multi-Agent with CrewAI

```python
from crewai import Agent, Task, Crew

# Define agents with roles
researcher = Agent(
    role="Researcher",
    goal="Gather comprehensive information",
    backstory="Expert at finding and synthesizing information",
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Create clear, engaging content",
    backstory="Skilled content writer with technical expertise",
    llm=llm
)

editor = Agent(
    role="Editor",
    goal="Ensure quality and accuracy",
    backstory="Meticulous editor with eye for detail",
    llm=llm
)

# Define tasks
research_task = Task(
    description="Research the topic: {topic}",
    agent=researcher
)

writing_task = Task(
    description="Write article based on research",
    agent=writer
)

# Create and run crew
crew = Crew(agents=[researcher, writer, editor], tasks=[research_task, writing_task])
result = crew.kickoff(inputs={"topic": "AI in 2025"})
```

---

## Milestone Checklist

- [ ] 5 custom tools implemented
- [ ] ReAct agent with tool selection working
- [ ] LangGraph workflow with human-in-the-loop
- [ ] Multi-agent content generation system
- [ ] CrewAI role-based team
- [ ] Agent evaluation framework

---

## DGX Spark Setup

### NGC Container Launch

```bash
# Start NGC container with all required flags
# Using --network=host for seamless Ollama Web UI access (http://localhost:11434)
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

**Note:** Using `--network=host` instead of `-p 8888:8888` provides seamless access to both Jupyter (port 8888) and Ollama Web UI (port 11434) without explicit port mapping.

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

## Common Issues

| Issue | Solution |
|-------|----------|
| Agent loops forever | Add max_iterations limit to AgentExecutor |
| Tool errors not handled | Add try/except in tool functions |
| Context too long | Summarize history or use sliding window |
| LangGraph state lost | Check checkpoint configuration |
| CrewAI agents conflict | Define clearer role boundaries |

---

## Next Steps

After completing this module:
1. ✅ You've completed Domain 3: LLM Systems!
2. 📁 Save your agent implementations to `scripts/`
3. ➡️ Proceed to [Domain 4: Production AI](../../domain-4-production-ai/) - [Module 4.1: Multimodal AI](../../domain-4-production-ai/module-4.1-multimodal/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 3.5: RAG Systems](../module-3.5-rag-systems/) | **Module 3.6: AI Agents** | [Module 4.1: Multimodal AI](../../domain-4-production-ai/module-4.1-multimodal/) |

---

## Study Materials

| Document | Purpose | Time |
|----------|---------|------|
| [QUICKSTART.md](./QUICKSTART.md) | 5-minute working agent demo | 5 min |
| [ELI5.md](./ELI5.md) | Plain-language concept explanations | 15 min |
| [PREREQUISITES.md](./PREREQUISITES.md) | Skills check before starting | 10 min |
| [STUDY_GUIDE.md](./STUDY_GUIDE.md) | Learning path and objectives | 10 min |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Commands, patterns, and values | Reference |
| [LAB_PREP.md](./LAB_PREP.md) | Environment setup checklist | 10 min |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | Common errors, solutions, and FAQs | Reference |

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://docs.crewai.com/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
