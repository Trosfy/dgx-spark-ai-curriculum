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

# Verify via Ollama Web UI API
curl http://localhost:11434/api/tags

# Ollama Web UI endpoint: http://localhost:11434
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

## ❓ Frequently Asked Questions

**Q: What's the difference between an agent and a chain?**

**A**:

| Feature | Chain | Agent |
|---------|-------|-------|
| **Control flow** | Fixed, predefined | Dynamic, LLM decides |
| **Tools** | Can use, but sequence is fixed | Selects tools based on task |
| **Iteration** | Usually single pass | Loops until task complete |
| **Complexity** | Simpler to debug | More powerful but harder to debug |

**Use chains when**: You know the exact steps needed.
**Use agents when**: The LLM needs to decide what to do.

---

**Q: When should I use LangGraph vs a simple agent?**

**A**: Use LangGraph when you need:

- **Conditional branching**: Different paths based on outcomes
- **Human-in-the-loop**: Approval steps before actions
- **Complex state**: Multiple pieces of information to track
- **Retry loops**: Automatic retry on failure
- **Multi-step workflows**: Beyond simple tool selection

Use simple agents when:
- Task is straightforward (answer question, do calculation)
- No approval needed
- Single-session interaction

---

**Q: How do I choose between CrewAI and custom multi-agent?**

**A**:

| Factor | CrewAI | Custom Multi-Agent |
|--------|--------|-------------------|
| **Setup time** | Fast | Slower |
| **Flexibility** | Medium | High |
| **Role-based tasks** | Excellent | Manual |
| **Customization** | Limited | Full control |
| **Learning curve** | Low | Medium |

**Use CrewAI when**: You want quick role-based agent teams.
**Use custom when**: You need specific agent behaviors or communication patterns.

---

**Q: What model should I use for agents?**

**A**:

| Model | Best For | Trade-off |
|-------|----------|-----------|
| llama3.1:8b | Development, testing | Fast but may struggle with complex reasoning |
| llama3.1:70b | Production, complex tasks | Slow but excellent reasoning |
| GPT-4 | Best quality (if API OK) | Cost, API dependency |

**Recommendation**: Develop with 8B, deploy with 70B.

---

**Q: How many tools should an agent have?**

**A**: General guidelines:

| Tool Count | Recommendation |
|------------|----------------|
| 1-5 | Ideal for most cases |
| 6-10 | OK if tools are distinct |
| 10+ | Consider grouping or using sub-agents |

More tools = more confusion. If agent picks wrong tools:
1. Reduce tool count
2. Make descriptions more distinct
3. Group related tools into one

---

**Q: How should I write tool descriptions?**

**A**: Follow this pattern:
- One-line summary of what this tool does
- "Use this when you need to:" with specific use cases
- "Do NOT use this for:" what it can't do
- Clear argument descriptions
- Return value description
- Examples

Good descriptions prevent wrong tool selection.

---

**Q: Should tools have side effects?**

**A**: Best practices:

| Tool Type | Side Effects | Recommendation |
|-----------|--------------|----------------|
| Read-only | None | Safe, use freely |
| Write | Yes | Require confirmation |
| External API | Depends | Add rate limiting |
| Irreversible | Yes | Human approval required |

For risky operations, use LangGraph with approval steps.

---

**Q: How do I debug an agent that's not working?**

**A**: Step-by-step debugging:

1. Enable verbose mode
2. Run and inspect
3. Check intermediate steps to see which tools were called and what they returned

---

**Q: How do I prevent agents from looping forever?**

**A**: Multiple safeguards:

1. Set max iterations
2. Set max execution time
3. Use early stopping
4. Track tool usage patterns

---

**Q: How do I handle tool failures gracefully?**

**A**: Error handling at multiple levels:

**Level 1**: In the tool - wrap in try/except and return helpful error messages

**Level 2**: In the executor - enable handle_parsing_errors and set max_iterations

---

**Q: How do agents communicate in multi-agent systems?**

**A**: Common patterns:

**1. Sequential (Pipeline)**: Agent A → output → Agent B → output → Agent C

**2. Shared State (Blackboard)**: All agents read/write to shared state

**3. Message Passing**: Agents append to and read from message queue

**4. Direct Invocation (CrewAI)**: CrewAI handles communication automatically via tasks

---

**Q: How do I handle conflicting agent outputs?**

**A**: Resolution strategies:

**1. Voting**: Multiple agents vote, majority wins

**2. Hierarchy**: Senior agent resolves conflicts

**3. Confidence-based**: Highest confidence wins

---

**Q: How do I make agents faster?**

**A**: Optimization strategies:

| Strategy | Impact | Complexity |
|----------|--------|------------|
| Use smaller model | High | Low |
| Cache tool results | Medium | Low |
| Reduce tool count | Medium | Low |
| Parallel tool calls | High | Medium |
| Pre-compute embeddings | Medium | Medium |

---

**Q: How do I monitor agents in production?**

**A**: Key metrics to track:
- Total runs
- Successful vs failed runs
- Average duration
- Average iterations
- Tool usage statistics

---

**Q: How do I test agents?**

**A**: Testing strategy:

1. **Unit test individual tools**: Verify each tool works correctly
2. **Test agent behavior**: Check that correct tools are selected
3. **Integration tests**: Test with expected outcomes for various queries

---

**Q: Can agents learn from experience?**

**A**: Approaches to agent learning:

**1. Long-term Memory**: Store successful patterns in vector database

**2. Prompt Refinement**: Add successful examples to prompt

**3. Tool Preference Learning**: Track which tools work best for which queries

---

**Q: How do I implement human-in-the-loop?**

**A**: Using LangGraph:

Create nodes that wait for human approval and conditional edges that route to approval nodes for risky operations.

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
