#!/usr/bin/env python3
"""
Agent Swarm Demo

Quick demonstration of multi-agent coordination.
"""

from base_agent import SharedMemory
from specialized_agents import ResearcherAgent, CoderAgent, AnalystAgent, ReviewerAgent
from coordinator import AgentCoordinator


def demo_individual_agents():
    """Demonstrate individual agent capabilities."""
    print("=" * 60)
    print("PART 1: INDIVIDUAL AGENTS")
    print("=" * 60)

    # Shared memory
    memory = SharedMemory()

    # Create agents
    agents = {
        "researcher": ResearcherAgent(memory),
        "coder": CoderAgent(memory),
        "analyst": AnalystAgent(memory),
        "reviewer": ReviewerAgent(memory),
    }

    # Test each
    tasks = {
        "researcher": "What are the key components of a RAG system?",
        "coder": "Write a function to compute attention scores",
        "analyst": "Analyze the performance of different quantization methods",
        "reviewer": "Review the following code for best practices: def foo(): pass"
    }

    for name, task in tasks.items():
        print(f"\n{name.upper()} AGENT")
        print("-" * 40)
        result = agents[name].process(task)
        print(f"Success: {result.success}")
        print(f"Time: {result.execution_time:.2f}s")
        print(f"Output preview: {result.output[:100]}...")


def demo_agent_communication():
    """Demonstrate agent-to-agent communication."""
    print("\n" + "=" * 60)
    print("PART 2: AGENT COMMUNICATION")
    print("=" * 60)

    memory = SharedMemory()
    researcher = ResearcherAgent(memory)
    coder = CoderAgent(memory)

    # Researcher sends message to coder
    print("\n1. Researcher sends request to Coder:")
    msg = researcher.send_message(
        "coder",
        "I found that we need to implement cosine similarity. Can you help?"
    )
    print(f"   From: {msg.sender} -> To: {msg.receiver}")
    print(f"   Content: {msg.content}")

    # Coder receives and responds
    print("\n2. Coder receives message:")
    messages = coder.receive_messages()
    for m in messages:
        print(f"   Received from {m.sender}: {m.content[:50]}...")

    # Add facts to shared memory
    print("\n3. Agents share knowledge:")
    researcher.add_to_memory("similarity_method", "cosine")
    coder.add_to_memory("implementation", "numpy.dot")

    print(f"   Facts in shared memory: {len(memory.facts)}")
    for fact in memory.facts:
        print(f"   - {fact['key']}: {fact['value']} (from {fact['source']})")


def demo_coordinated_task():
    """Demonstrate coordinated multi-agent task."""
    print("\n" + "=" * 60)
    print("PART 3: COORDINATED TASK EXECUTION")
    print("=" * 60)

    # Create coordinator
    coordinator = AgentCoordinator(require_human_approval=False)

    # Complex task
    task = """
    Research the best embedding models for semantic search,
    implement a simple vector store,
    and analyze the retrieval performance.
    """

    print(f"\nTask: {task.strip()}")

    result = coordinator.execute_task(task)

    print("\n" + "-" * 40)
    print("COORDINATION RESULTS")
    print("-" * 40)
    print(f"Final Status: {result.status.value}")

    for st in result.subtasks:
        status_icon = "v" if st.status.value == "completed" else "x"
        print(f"  [{status_icon}] {st.id}: {st.assigned_agent}")


def demo_human_in_loop():
    """Demonstrate human-in-the-loop safety."""
    print("\n" + "=" * 60)
    print("PART 4: HUMAN-IN-THE-LOOP")
    print("=" * 60)

    print("""
In production, the coordinator can pause at key checkpoints:

1. After task decomposition - approve the plan
2. Before critical actions - confirm execution
3. After completion - validate results

Example checkpoints:
- [CHECKPOINT] Plan: 4 subtasks. Approve? (y/n)
- [CHECKPOINT] About to execute code. Continue? (y/n)
- [CHECKPOINT] Task complete. Accept results? (y/n)

This ensures human oversight for:
- Sensitive operations
- Resource-intensive tasks
- External API calls
- Irreversible actions
""")


def main():
    print("=" * 60)
    print("AI AGENT SWARM DEMO")
    print("=" * 60)
    print("""
This demo shows the core components of a multi-agent system:

1. Individual Agents - Specialized capabilities
2. Agent Communication - Message passing and shared memory
3. Coordinated Tasks - Multi-agent workflows
4. Human-in-the-Loop - Safety checkpoints
""")

    demo_individual_agents()
    demo_agent_communication()
    demo_coordinated_task()
    demo_human_in_loop()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("""
For your full capstone:

1. Add real LLM backend for each agent
2. Implement vector-based shared memory
3. Build sophisticated task decomposition with LLM
4. Add monitoring and logging
5. Create web interface for human oversight
6. Implement agent-specific fine-tuning
7. Add safety guardrails and output validation
""")


if __name__ == "__main__":
    main()
