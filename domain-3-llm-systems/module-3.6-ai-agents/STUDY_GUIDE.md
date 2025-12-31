# Module 3.6: AI Agents & Agentic Systems - Study Guide

## 🎯 Learning Objectives
By the end of this module, you will be able to:
1. **Create** agents with custom tools using LangChain
2. **Design** multi-agent architectures for complex tasks
3. **Implement** stateful workflows with LangGraph
4. **Evaluate** agent performance and reliability

## 🗺️ Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-3.6.1-custom-tools.ipynb | Building tools | ~2 hr | 5 custom tools |
| 2 | lab-3.6.2-react-agent.ipynb | ReAct pattern | ~2 hr | Working agent with tool selection |
| 3 | lab-3.6.3-langgraph.ipynb | Workflow graphs | ~2 hr | Multi-step with human approval |
| 4 | lab-3.6.4-multi-agent.ipynb | Agent collaboration | ~2 hr | 3-agent content team |
| 5 | lab-3.6.5-crewai.ipynb | CrewAI framework | ~2 hr | Role-based agent team |
| 6 | lab-3.6.6-evaluation.ipynb | Agent benchmarking | ~2 hr | Evaluation framework |

**Total time**: ~12 hours

## 🔑 Core Concepts

### ReAct Pattern
**What**: Reasoning + Acting loop where agent thinks, acts, and observes
**Why it matters**: Enables agents to solve complex problems step-by-step
**First appears in**: Lab 3.6.2

### Tool Calling
**What**: Agents selecting and invoking functions to perform actions
**Why it matters**: Extends LLM capabilities beyond text generation
**First appears in**: Lab 3.6.1

### LangGraph
**What**: Graph-based orchestration for stateful agent workflows
**Why it matters**: Enables complex, multi-step agent behaviors with conditionals
**First appears in**: Lab 3.6.3

### Multi-Agent Systems
**What**: Multiple specialized agents collaborating on tasks
**Why it matters**: Better results through specialization and parallel work
**First appears in**: Lab 3.6.4

## 🔗 How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 3.5          ──►     Module 3.6          ──►    Domain 4
RAG Systems                 AI Agents                  Production AI
(retrieval)                 (tool use)                 (deployment)
```

**Builds on**:
- **RAG systems** from Module 3.5 (agents use retrievers as tools)
- **LLM inference** from Module 3.3 (agents need fast LLM responses)
- **Prompt engineering** (good prompts = better agent reasoning)

**Prepares for**:
- Domain 4 covers deploying agents in production
- Module 4.5 builds Gradio UIs for agents
- Real-world applications combine all techniques

## 📖 Recommended Approach

### Standard Path (10-12 hours):
1. **Day 1: Fundamentals (Labs 1-2)**
   - Lab 3.6.1 builds custom tools
   - Lab 3.6.2 creates ReAct agent

2. **Day 2: Advanced Workflows (Labs 3-4)**
   - Lab 3.6.3 builds LangGraph workflows
   - Lab 3.6.4 creates multi-agent systems

3. **Day 3: Frameworks & Evaluation (Labs 5-6)**
   - Lab 3.6.5 explores CrewAI
   - Lab 3.6.6 builds evaluation framework

### Quick Path (6-8 hours, if experienced):
1. Do Lab 3.6.1 (tools) + Lab 3.6.2 (ReAct)
2. Do Lab 3.6.3 (LangGraph) - essential for production
3. Choose: Lab 3.6.4 (multi-agent) OR Lab 3.6.5 (CrewAI)
4. Lab 3.6.6 (evaluation) if building for production

## 📋 Before You Start
→ See [LAB_PREP.md](./LAB_PREP.md) for environment setup
→ See [QUICKSTART.md](./QUICKSTART.md) for 5-minute agent demo
→ See [ELI5.md](./ELI5.md) for concept explanations
→ Ensure Module 3.5 (RAG) is completed
