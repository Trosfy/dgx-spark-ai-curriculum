# Module 3.6: AI Agents & Agentic Systems - Quick Reference

## 🚀 Essential Commands

### Install Agent Dependencies
```bash
pip install \
    langchain langchain-community \
    langgraph \
    crewai \
    ollama \
    chromadb sentence-transformers
```

### NGC Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ollama:/root/.ollama \
    --ipc=host \
    --network=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

## 📊 Key Values

### Agent Types Comparison
| Type | Use Case | Framework |
|------|----------|-----------|
| ReAct Agent | General tool use | LangChain |
| Structured Output | JSON/Schema output | LangChain |
| LangGraph Agent | Complex workflows | LangGraph |
| CrewAI Agent | Role-based teams | CrewAI |
| AutoGen Agent | Conversations | AutoGen |

### DGX Spark Agent Performance
| Component | Memory | Speed |
|-----------|--------|-------|
| Llama 3.1 8B (agent) | ~8GB | ~50 tok/s |
| Llama 3.1 70B (agent) | ~45GB | ~20 tok/s |
| Tool execution | Varies | ~100ms/call |
| LangGraph state | ~100MB | ~10ms/step |

## 🔧 Common Patterns

### Pattern: Custom Tool Definition
```python
from langchain.tools import tool
from typing import Optional

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any math operations.

    Args:
        expression: A valid Python math expression (e.g., "2 + 2", "15 * 3")

    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def search_documents(query: str, max_results: int = 5) -> str:
    """Search the knowledge base for relevant documents.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Relevant document excerpts
    """
    # Implementation
    pass
```

### Pattern: ReAct Agent
```python
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# LLM
llm = Ollama(model="llama3.1:70b")

# Tools
tools = [calculate, search_documents, ...]

# ReAct prompt
react_prompt = PromptTemplate.from_template("""Answer the question using the tools available.

Tools: {tools}

Use this format:
Question: the input question
Thought: think about what to do
Action: tool name from [{tool_names}]
Action Input: input to the tool
Observation: result from the tool
... (repeat as needed)
Thought: I now know the answer
Final Answer: the final answer

Question: {input}
{agent_scratchpad}""")

# Create agent
agent = create_react_agent(llm, tools, react_prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)

# Run
result = executor.invoke({"input": "What is 25 * 47?"})
```

### Pattern: LangGraph Workflow
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]
    current_step: str
    approved: bool

def plan_node(state: AgentState) -> AgentState:
    """Plan the next action."""
    # Planning logic
    return {"current_step": "execute", "messages": ["Plan created"]}

def execute_node(state: AgentState) -> AgentState:
    """Execute the plan."""
    # Execution logic
    return {"current_step": "review", "messages": ["Executed"]}

def should_continue(state: AgentState) -> str:
    """Decide next node based on state."""
    if state["approved"]:
        return "execute"
    return "await_approval"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.add_node("await_approval", await_approval_node)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "await_approval")
workflow.add_conditional_edges("await_approval", should_continue)
workflow.add_edge("execute", END)

app = workflow.compile()

# Run
result = app.invoke({"messages": [], "current_step": "start", "approved": False})
```

### Pattern: CrewAI Team
```python
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1:70b")

# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Gather comprehensive, accurate information",
    backstory="Expert researcher with attention to detail",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Create clear, engaging content",
    backstory="Skilled writer with technical expertise",
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Ensure quality and accuracy",
    backstory="Meticulous editor with eye for detail",
    llm=llm,
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research {topic} comprehensively",
    expected_output="Detailed research notes",
    agent=researcher
)

writing_task = Task(
    description="Write article based on research",
    expected_output="Complete article draft",
    agent=writer,
    context=[research_task]
)

editing_task = Task(
    description="Edit and polish the article",
    expected_output="Final polished article",
    agent=editor,
    context=[writing_task]
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "AI Agents in 2025"})
```

### Pattern: Memory Systems
```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)

# Full conversation history
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Summarized history (saves tokens)
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Last k exchanges only
window_memory = ConversationBufferWindowMemory(
    k=5,  # Keep last 5 exchanges
    memory_key="chat_history",
    return_messages=True
)

# Use with agent
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=buffer_memory,
    verbose=True
)
```

### Pattern: Error Handling in Tools
```python
@tool
def safe_api_call(endpoint: str) -> str:
    """Call an API safely with error handling."""
    import requests
    from time import sleep

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            if attempt < max_retries - 1:
                sleep(2 ** attempt)  # Exponential backoff
                continue
            return "Error: API request timed out"
        except requests.HTTPError as e:
            return f"Error: HTTP {e.response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Agent loops forever | Set `max_iterations=10` in AgentExecutor |
| Tool errors crash agent | Add try/except in tool functions |
| Context too long | Use summary memory or sliding window |
| Vague tool descriptions | Write clear, specific descriptions with examples |
| Missing tool inputs | Define all parameters with types and descriptions |
| Agent picks wrong tool | Improve tool descriptions, reduce tool overlap |

## 🔗 Quick Links
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [AutoGen](https://microsoft.github.io/autogen/)
