# Module 3.6: AI Agents & Agentic Systems - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Create a ReAct agent that can use tools to answer questions—thinking, acting, and observing in a loop.

## ✅ Before You Start
- [ ] DGX Spark NGC container running
- [ ] Ollama running with llama3.1:8b or 70b

## 🚀 Let's Go!

### Step 1: Install Dependencies
```bash
pip install langchain langchain-community ollama
```

### Step 2: Create Custom Tools
```python
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for any math operations."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_knowledge(query: str) -> str:
    """Search a knowledge base for information."""
    # Simple mock knowledge base
    knowledge = {
        "dgx spark": "DGX Spark has 128GB unified memory and a Blackwell GPU.",
        "lora": "LoRA trains only 0.1% of model parameters for efficient fine-tuning.",
        "rag": "RAG retrieves documents before generating to reduce hallucinations."
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "No information found for that query."

tools = [calculate, get_current_time, search_knowledge]
print(f"✅ Created {len(tools)} tools")
```

### Step 3: Create the Agent
```python
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# Initialize LLM
llm = Ollama(model="llama3.1:8b")

# ReAct prompt template
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# Create the agent
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Show reasoning
    max_iterations=5,
    handle_parsing_errors=True
)
print("✅ Agent created")
```

### Step 4: Run the Agent
```python
# Test with a math question
result = agent_executor.invoke({
    "input": "What is 25 multiplied by 47, and what time is it now?"
})
print(f"\nFinal Answer: {result['output']}")
```

**Expected output** (with verbose=True):
```
> Entering new AgentExecutor chain...
Thought: I need to calculate 25 * 47 and get the current time
Action: calculate
Action Input: 25 * 47
Observation: Result: 1175
Thought: Now I need to get the current time
Action: get_current_time
Action Input:
Observation: 2025-01-15 14:30:22
Thought: I now know the final answer
Final Answer: 25 multiplied by 47 equals 1175, and the current time is 2025-01-15 14:30:22.
```

### Step 5: Try Knowledge Search
```python
result = agent_executor.invoke({
    "input": "How much memory does DGX Spark have?"
})
print(f"\nFinal Answer: {result['output']}")
```

## 🎉 You Did It!

You just built a working AI agent that can:
- **Think** about what to do
- **Act** by selecting and using tools
- **Observe** the results
- **Repeat** until it has the answer

This same pattern scales to:
- **More tools**: File operations, web search, APIs
- **Complex workflows**: LangGraph for multi-step tasks
- **Multiple agents**: CrewAI for team collaboration

## ▶️ Next Steps
1. **Add more tools**: See [Lab 3.6.1](./labs/lab-3.6.1-custom-tools.ipynb)
2. **Build workflows**: See [Lab 3.6.3](./labs/lab-3.6.3-langgraph.ipynb)
3. **Full setup**: Start with [LAB_PREP.md](./LAB_PREP.md)
