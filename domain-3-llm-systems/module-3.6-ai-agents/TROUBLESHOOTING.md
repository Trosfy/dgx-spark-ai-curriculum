# Module 3.6: AI Agents & Agentic Systems - Troubleshooting Guide

## 🔍 Quick Diagnosis

### Symptom Categories
| Symptom | Jump to Section |
|---------|-----------------|
| Agent not responding | [Agent Execution Issues](#agent-execution-issues) |
| Agent loops forever | [Infinite Loop Issues](#infinite-loop-issues) |
| Tool errors | [Tool Issues](#tool-issues) |
| LangGraph problems | [LangGraph Issues](#langgraph-issues) |
| CrewAI errors | [CrewAI Issues](#crewai-issues) |
| Memory/performance | [Performance Issues](#performance-issues) |

---

## Agent Execution Issues

### ❌ Error: "Could not parse LLM output"
```
OutputParserException: Could not parse LLM output
```

**Cause**: LLM didn't follow the expected format (Thought/Action/Observation)

**Solutions**:
```python
# 1. Enable error handling
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # Auto-retry on parse errors
    max_iterations=10
)

# 2. Use a more capable model
llm = Ollama(model="llama3.1:70b")  # Better at following formats

# 3. Simplify the prompt
# Reduce number of tools or make tool descriptions clearer
```

---

### ❌ Error: "Agent stopped due to max iterations"
```
AgentExecutor stopped due to max_iterations_exceeded
```

**Cause**: Agent couldn't complete task within iteration limit

**Solutions**:
```python
# 1. Increase max iterations
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=20  # Default is often 10
)

# 2. Check if task is too complex
# Break into subtasks or simplify the request

# 3. Check for tool failures causing loops
# Enable verbose to see what's happening
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

---

### ❌ Error: "Ollama connection refused"
```
ConnectionError: Connection refused at localhost:11434
```

**Solution**:
```bash
# Start Ollama service
ollama serve

# In background
nohup ollama serve &

# Verify it's running
curl http://localhost:11434/api/tags
```

---

### ❌ Agent gives wrong answers despite correct tools

**Cause**: Poor tool selection or reasoning

**Solutions**:
```python
# 1. Improve tool descriptions
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Use this for ANY math calculation including:
    - Basic arithmetic: 2 + 2, 10 * 5
    - Complex expressions: (15 + 3) * 2
    - Percentages: 0.15 * 100

    Args:
        expression: A valid Python math expression

    Returns:
        The numerical result
    """
    return str(eval(expression))

# 2. Reduce tool overlap
# Don't have multiple tools that could handle similar tasks

# 3. Use a better model
llm = Ollama(model="llama3.1:70b")  # Much better reasoning
```

---

## Infinite Loop Issues

### ❌ Agent keeps repeating the same action

**Cause**: Agent not observing results or stuck in reasoning loop

**Solutions**:
```python
# 1. Set strict iteration limit
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    return_intermediate_steps=True  # Debug what's happening
)

# 2. Check observation handling
result = executor.invoke({"input": "question"})
print("Steps:", result.get("intermediate_steps", []))

# 3. Improve tool error messages
@tool
def my_tool(query: str) -> str:
    """Tool description."""
    try:
        result = do_something(query)
        return f"Success: {result}"  # Clear success message
    except Exception as e:
        return f"Error: {e}. Try a different approach."  # Guides agent
```

---

### ❌ Agent alternates between two tools

**Cause**: Tools returning results that trigger each other

**Solution**:
```python
# Add early termination check
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    early_stopping_method="generate"  # Try to generate answer when stuck
)

# Or implement custom stopping
class CustomAgent:
    def should_stop(self, iterations, last_tool):
        if iterations > 5 and last_tool == self.previous_tool:
            return True  # Same tool twice = stop
        self.previous_tool = last_tool
        return False
```

---

## Tool Issues

### ❌ Error: "Tool not found"
```
ValueError: Tool 'my_tool' not found
```

**Solution**:
```python
# Ensure tool is in the tools list
tools = [calculate, search, my_tool]  # Include your tool

# Verify tool names match
for tool in tools:
    print(f"Tool: {tool.name}")  # Check exact names
```

---

### ❌ Error: "Missing required argument"
```
TypeError: my_tool() missing required argument: 'param'
```

**Solution**:
```python
# Define tool with proper signature
@tool
def my_tool(param1: str, param2: str = "default") -> str:
    """Tool description.

    Args:
        param1: Required parameter description
        param2: Optional parameter with default
    """
    return f"Result: {param1}, {param2}"

# Make sure docstring documents all parameters
```

---

### ❌ Tool execution crashes the agent

**Solution**:
```python
# Always wrap tool logic in try/except
@tool
def safe_tool(query: str) -> str:
    """Safe tool with error handling."""
    try:
        result = risky_operation(query)
        return f"Success: {result}"
    except ValueError as e:
        return f"Invalid input: {e}"
    except ConnectionError as e:
        return f"Connection failed: {e}. Try again later."
    except Exception as e:
        return f"Unexpected error: {e}. Try a different approach."
```

---

### ❌ Tool returns too much data

**Cause**: Large tool outputs overwhelm context window

**Solution**:
```python
@tool
def search_documents(query: str) -> str:
    """Search with truncation."""
    results = do_search(query)

    # Truncate if too long
    MAX_CHARS = 2000
    if len(results) > MAX_CHARS:
        results = results[:MAX_CHARS] + "\n... (truncated)"

    return results
```

---

## LangGraph Issues

### ❌ Error: "State validation failed"
```
ValidationError: State missing required key
```

**Solution**:
```python
from typing import TypedDict, Annotated
from operator import add

# Properly define state with all required fields
class AgentState(TypedDict):
    messages: Annotated[list, add]  # Accumulates
    current_step: str
    approved: bool

# Initialize state with all keys
initial_state = {
    "messages": [],
    "current_step": "start",
    "approved": False
}

result = app.invoke(initial_state)
```

---

### ❌ Error: "Edge target node not found"
```
ValueError: Edge target 'missing_node' not in graph
```

**Solution**:
```python
# Ensure all referenced nodes exist
workflow = StateGraph(AgentState)

# Add ALL nodes first
workflow.add_node("plan", plan_node)
workflow.add_node("execute", execute_node)
workflow.add_node("review", review_node)  # Don't forget this!

# Then add edges
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", "review")  # 'review' must exist
```

---

### ❌ Graph execution hangs

**Cause**: Infinite loop in conditional edges

**Solution**:
```python
def should_continue(state: AgentState) -> str:
    """Conditional edge with guaranteed termination."""
    # Always have an exit condition
    if state.get("iterations", 0) > 10:
        return "end"  # Force exit

    if state["approved"]:
        return "execute"

    return "await_approval"

# Add iteration counter in nodes
def some_node(state: AgentState) -> AgentState:
    iterations = state.get("iterations", 0) + 1
    return {**state, "iterations": iterations}
```

---

### ❌ State not persisting between nodes

**Solution**:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add persistence
memory = SqliteSaver.from_conn_string(":memory:")  # Or file path

app = workflow.compile(checkpointer=memory)

# Use thread_id for persistence
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(initial_state, config)
```

---

## CrewAI Issues

### ❌ Error: "Agent execution failed"
```
CrewAIError: Agent failed to complete task
```

**Solution**:
```python
from crewai import Agent, Task, Crew

# Give agents clearer instructions
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, comprehensive information",
    backstory="You are a meticulous researcher who verifies facts",
    llm=llm,
    verbose=True,  # See what's happening
    max_iter=5,    # Limit iterations
    allow_delegation=False  # Simpler execution
)
```

---

### ❌ Agents delegating in circles

**Cause**: Agents keep passing tasks to each other

**Solution**:
```python
# Disable delegation
agent = Agent(
    role="Writer",
    goal="Write content",
    backstory="...",
    llm=llm,
    allow_delegation=False  # Don't delegate
)

# Or limit to specific agents
from crewai import Task

task = Task(
    description="Write an article",
    agent=writer,
    expected_output="Article text"
    # No delegation happens for this task
)
```

---

### ❌ CrewAI runs very slowly

**Solutions**:
```python
# 1. Use faster model for simpler agents
fast_llm = Ollama(model="llama3.1:8b")
quality_llm = Ollama(model="llama3.1:70b")

# Use fast model for routine work
researcher = Agent(role="Researcher", llm=fast_llm, ...)

# Use quality model for critical work
editor = Agent(role="Editor", llm=quality_llm, ...)

# 2. Reduce verbosity
crew = Crew(agents=[...], tasks=[...], verbose=False)

# 3. Limit iterations
agent = Agent(..., max_iter=3)
```

---

## Performance Issues

### ❌ Agent responses are very slow

**Solutions**:
```python
# 1. Use smaller model
llm = Ollama(model="llama3.1:8b")  # vs 70b

# 2. Reduce tool count
# Only include tools actually needed
tools = [calculate, search]  # Not 10+ tools

# 3. Cache tool results
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str) -> str:
    return expensive_search(query)

@tool
def search(query: str) -> str:
    """Cached search."""
    return cached_search(query)
```

---

### ❌ Out of GPU memory

**Solutions**:
```python
# 1. Use smaller model
llm = Ollama(model="llama3.1:8b")

# 2. Clear cache between runs
import torch
torch.cuda.empty_cache()

# 3. Use CPU for some operations
# Ollama handles this automatically

# 4. Reduce context length
# Summarize conversation history
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)
```

---

### ❌ Context window exceeded

```
Error: Context length exceeded
```

**Solutions**:
```python
# 1. Use sliding window memory
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)  # Last 5 exchanges only

# 2. Truncate tool outputs
@tool
def search(query: str) -> str:
    """Search with truncation."""
    results = do_search(query)
    if len(results) > 1000:
        return results[:1000] + "..."
    return results

# 3. Summarize intermediate steps
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

---

## Debugging Tips

### Enable Verbose Output
```python
# LangChain
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# CrewAI
crew = Crew(agents=[...], tasks=[...], verbose=True)

# LangGraph
for step in app.stream(initial_state):
    print(step)  # See each step
```

### Use LangSmith Tracing
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
```

```python
# Traces will appear at smith.langchain.com
result = executor.invoke({"input": "test"})
```

---

## 🆘 Still Stuck?

1. **Enable verbose mode**: See exactly what's happening
2. **Simplify**: Start with one tool, add more gradually
3. **Check model**: 70B models reason much better than 8B
4. **Review prompts**: Clear tool descriptions help
5. **Consult docs**:
   - [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
   - [LangGraph](https://langchain-ai.github.io/langgraph/)
   - [CrewAI](https://docs.crewai.com/)
