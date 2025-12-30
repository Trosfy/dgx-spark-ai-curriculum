# Example: AI Agent Swarm

This is a minimal working example for Option C: AI Agent Swarm.

## What This Example Includes

- `base_agent.py` - Base agent class with shared interface
- `specialized_agents.py` - Researcher, Coder, Analyst agents
- `coordinator.py` - Multi-agent orchestration
- `demo.py` - Quick demonstration script

## Quick Start

```bash
# Navigate to this directory
cd examples/option-c-agent-swarm

# Run the demo
python demo.py
```

## Files Overview

### base_agent.py

A base agent implementation with:
- Standardized message passing
- Tool registration
- Memory access

### specialized_agents.py

Specialized agents including:
- `ResearcherAgent` - Web search and summarization
- `CoderAgent` - Code generation and analysis
- `AnalystAgent` - Data analysis and insights

### coordinator.py

Multi-agent orchestration with:
- Task decomposition
- Agent selection
- Result aggregation
- Human-in-the-loop checkpoints

### demo.py

Interactive demo showing:
- Multi-agent task execution
- Agent communication
- Coordinated problem solving

## Extending This Example

To build your full capstone, you'll want to:

1. **Add real LLM backend** for each agent
2. **Implement shared memory** with vector store
3. **Build coordination protocols** for complex tasks
4. **Add safety controls** and human-in-the-loop
5. **Create monitoring dashboard** for agent activity
6. **Build an API** for task submission

## Memory Requirements

This example is designed to run on DGX Spark:
- Multiple agents share single model
- OR load multiple smaller models
- 4-bit quantization
- Estimated: ~10GB GPU memory for single shared model

## Next Steps

1. Review the code in each file
2. Run the demo to see agents working together
3. Add your specific agent types
4. Follow the main Option C notebook for full implementation
