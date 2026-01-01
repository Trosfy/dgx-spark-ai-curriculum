# Module 3.6: AI Agents & Agentic Systems - Frequently Asked Questions

## Concepts & Design

### Q: What's the difference between an agent and a chain?
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

### Q: When should I use LangGraph vs a simple agent?
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

### Q: How do I choose between CrewAI and custom multi-agent?
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

### Q: What model should I use for agents?
**A**:

| Model | Best For | Trade-off |
|-------|----------|-----------|
| llama3.1:8b | Development, testing | Fast but may struggle with complex reasoning |
| llama3.1:70b | Production, complex tasks | Slow but excellent reasoning |
| GPT-4 | Best quality (if API OK) | Cost, API dependency |

**Recommendation**: Develop with 8B, deploy with 70B.

---

## Tool Design

### Q: How many tools should an agent have?
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

### Q: How should I write tool descriptions?
**A**: Follow this pattern:

```python
@tool
def my_tool(param: str) -> str:
    """One-line summary of what this tool does.

    Use this when you need to:
    - Specific use case 1
    - Specific use case 2

    Do NOT use this for:
    - What this tool can't do

    Args:
        param: Description of what to provide

    Returns:
        What the tool returns

    Examples:
        my_tool("example input") -> "example output"
    """
```

Good descriptions prevent wrong tool selection.

---

### Q: Should tools have side effects?
**A**: Best practices:

| Tool Type | Side Effects | Recommendation |
|-----------|--------------|----------------|
| Read-only | None | Safe, use freely |
| Write | Yes | Require confirmation |
| External API | Depends | Add rate limiting |
| Irreversible | Yes | Human approval required |

For risky operations, use LangGraph with approval steps:

```python
def should_continue(state):
    if state["action_type"] == "delete":
        return "await_approval"  # Human must approve
    return "execute"
```

---

## Debugging & Reliability

### Q: How do I debug an agent that's not working?
**A**: Step-by-step debugging:

```python
# 1. Enable verbose mode
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# 2. Run and inspect
result = executor.invoke({"input": "test"})

# 3. Check what happened
for step in result["intermediate_steps"]:
    action, observation = step
    print(f"Tool: {action.tool}")
    print(f"Input: {action.tool_input}")
    print(f"Output: {observation}")
    print("---")
```

---

### Q: How do I prevent agents from looping forever?
**A**: Multiple safeguards:

```python
# 1. Set max iterations
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # Stop after 10 steps
    max_execution_time=60  # Stop after 60 seconds
)

# 2. Use early stopping
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    early_stopping_method="generate"  # Try to answer when stuck
)

# 3. Track tool usage patterns
# If same tool called 3x with same input, force stop
```

---

### Q: How do I handle tool failures gracefully?
**A**: Error handling at multiple levels:

```python
# Level 1: In the tool
@tool
def my_tool(query: str) -> str:
    """Tool with error handling."""
    try:
        result = do_something(query)
        return f"Success: {result}"
    except ValidationError:
        return "Error: Invalid input. Please provide a valid query."
    except ConnectionError:
        return "Error: Service unavailable. Try again later."
    except Exception as e:
        return f"Error: {e}. Try a different approach."

# Level 2: In the executor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # Retry on parse errors
    max_iterations=10
)
```

---

## Multi-Agent Systems

### Q: How do agents communicate in multi-agent systems?
**A**: Common patterns:

**1. Sequential (Pipeline)**:
```
Agent A → output → Agent B → output → Agent C
```

**2. Shared State (Blackboard)**:
```python
shared_state = {"research": None, "draft": None, "final": None}
# All agents read/write to shared state
```

**3. Message Passing**:
```python
messages = []
def agent_a():
    messages.append({"from": "A", "content": "..."})

def agent_b():
    a_messages = [m for m in messages if m["from"] == "A"]
    # Process messages from A
```

**4. Direct Invocation (CrewAI)**:
```python
# CrewAI handles communication automatically via tasks
task1 = Task(agent=researcher, ...)
task2 = Task(agent=writer, context=[task1])  # Gets task1 output
```

---

### Q: How do I handle conflicting agent outputs?
**A**: Resolution strategies:

**1. Voting**: Multiple agents vote, majority wins
```python
votes = [agent1.decide(), agent2.decide(), agent3.decide()]
result = max(set(votes), key=votes.count)
```

**2. Hierarchy**: Senior agent resolves conflicts
```python
if agent1.output != agent2.output:
    result = senior_agent.resolve(agent1.output, agent2.output)
```

**3. Confidence-based**: Highest confidence wins
```python
outputs = [(agent.output, agent.confidence) for agent in agents]
result = max(outputs, key=lambda x: x[1])[0]
```

---

## Performance & Production

### Q: How do I make agents faster?
**A**: Optimization strategies:

| Strategy | Impact | Complexity |
|----------|--------|------------|
| Use smaller model | High | Low |
| Cache tool results | Medium | Low |
| Reduce tool count | Medium | Low |
| Parallel tool calls | High | Medium |
| Pre-compute embeddings | Medium | Medium |

```python
# Caching example
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_lookup(query: str) -> str:
    return expensive_operation(query)

@tool
def search(query: str) -> str:
    """Cached search tool."""
    return cached_lookup(query)
```

---

### Q: How do I monitor agents in production?
**A**: Key metrics to track:

```python
import time
from dataclasses import dataclass

@dataclass
class AgentMetrics:
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    avg_duration_ms: float = 0
    avg_iterations: float = 0
    tool_usage: dict = None

metrics = AgentMetrics(tool_usage={})

def monitored_run(executor, input_data):
    start = time.time()
    try:
        result = executor.invoke(input_data)
        metrics.successful_runs += 1

        # Track tool usage
        for step in result.get("intermediate_steps", []):
            tool_name = step[0].tool
            metrics.tool_usage[tool_name] = metrics.tool_usage.get(tool_name, 0) + 1

        return result
    except Exception as e:
        metrics.failed_runs += 1
        raise
    finally:
        metrics.total_runs += 1
        duration = (time.time() - start) * 1000
        # Update rolling average
        metrics.avg_duration_ms = (
            metrics.avg_duration_ms * (metrics.total_runs - 1) + duration
        ) / metrics.total_runs
```

---

### Q: How do I test agents?
**A**: Testing strategy:

```python
import pytest

# 1. Unit test individual tools
def test_calculate_tool():
    result = calculate.invoke("2 + 2")
    assert "4" in result

# 2. Test agent behavior with mock tools
def test_agent_uses_correct_tool():
    result = executor.invoke(
        {"input": "What is 15 * 23?"},
    )
    # Check that calculate tool was used
    steps = result.get("intermediate_steps", [])
    tools_used = [step[0].tool for step in steps]
    assert "calculate" in tools_used

# 3. Integration tests with expected outcomes
@pytest.mark.parametrize("question,expected", [
    ("What is 2+2?", "4"),
    ("What time is it?", "current time"),
])
def test_agent_answers(question, expected):
    result = executor.invoke({"input": question})
    assert expected.lower() in result["output"].lower()
```

---

## Advanced Topics

### Q: Can agents learn from experience?
**A**: Approaches to agent learning:

**1. Long-term Memory**:
```python
# Store successful patterns
memory_store = ChromaDB()

def remember_success(query, tools_used, result):
    memory_store.add(
        documents=[f"Query: {query}, Tools: {tools_used}, Result: {result}"],
        metadatas=[{"success": True}]
    )

# Retrieve similar past experiences
def recall(query):
    return memory_store.query(query, n=3)
```

**2. Prompt Refinement**:
```python
# Add successful examples to prompt
examples = get_successful_examples(task_type)
prompt = f"""
Here are examples of similar successful tasks:
{examples}

Now handle: {current_task}
"""
```

**3. Tool Preference Learning**:
```python
# Track which tools work best for which queries
tool_success_rate = defaultdict(lambda: {"success": 0, "total": 0})

def update_stats(tool_name, success):
    tool_success_rate[tool_name]["total"] += 1
    if success:
        tool_success_rate[tool_name]["success"] += 1
```

---

### Q: How do I implement human-in-the-loop?
**A**: Using LangGraph:

```python
from langgraph.graph import StateGraph, END

def await_human_approval(state):
    """Node that waits for human approval."""
    # In production, this would notify human and wait
    print(f"Awaiting approval for: {state['pending_action']}")
    # For demo, auto-approve
    return {"approved": True}

def should_continue(state):
    if state.get("requires_approval") and not state.get("approved"):
        return "await_approval"
    return "execute"

workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_step)
workflow.add_node("await_approval", await_human_approval)
workflow.add_node("execute", execute_step)

workflow.add_conditional_edges("plan", should_continue)
workflow.add_edge("await_approval", "execute")
workflow.add_edge("execute", END)
```

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for common patterns
- See [ELI5.md](./ELI5.md) for concept explanations
- Consult documentation:
  - [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
  - [LangGraph](https://langchain-ai.github.io/langgraph/)
  - [CrewAI](https://docs.crewai.com/)
