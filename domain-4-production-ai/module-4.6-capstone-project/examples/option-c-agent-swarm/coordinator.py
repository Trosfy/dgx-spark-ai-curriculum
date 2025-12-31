#!/usr/bin/env python3
"""
Agent Coordinator

Orchestrates multiple agents to solve complex tasks.
This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from base_agent import BaseAgent, AgentResult, SharedMemory, MessageType
from specialized_agents import ResearcherAgent, CoderAgent, AnalystAgent, ReviewerAgent
import time


class TaskStatus(Enum):
    """Status of a coordinated task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """A subtask assigned to an agent."""
    id: str
    description: str
    assigned_agent: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[AgentResult] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CoordinatedTask:
    """A complex task broken into subtasks."""
    id: str
    description: str
    subtasks: List[SubTask]
    status: TaskStatus = TaskStatus.PENDING
    final_result: Optional[str] = None


class AgentCoordinator:
    """
    Coordinates multiple agents to solve complex tasks.

    Responsibilities:
    - Task decomposition
    - Agent selection
    - Workflow orchestration
    - Result aggregation
    """

    def __init__(self, require_human_approval: bool = True):
        self.memory = SharedMemory()
        self.require_human_approval = require_human_approval

        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {
            "researcher": ResearcherAgent(self.memory),
            "coder": CoderAgent(self.memory),
            "analyst": AnalystAgent(self.memory),
            "reviewer": ReviewerAgent(self.memory),
        }

        self.task_history: List[CoordinatedTask] = []

    def decompose_task(self, task: str) -> List[SubTask]:
        """
        Break a complex task into subtasks.

        Args:
            task: Complex task description

        Returns:
            List of subtasks
        """
        # Simple rule-based decomposition (use LLM in production)
        subtasks = []
        task_lower = task.lower()

        # Research subtask
        if any(kw in task_lower for kw in ["research", "find", "learn", "understand"]):
            subtasks.append(SubTask(
                id="research",
                description=f"Research: {task}",
                assigned_agent="researcher"
            ))

        # Coding subtask
        if any(kw in task_lower for kw in ["code", "implement", "write", "function", "script"]):
            subtasks.append(SubTask(
                id="code",
                description=f"Implement: {task}",
                assigned_agent="coder",
                dependencies=["research"] if "research" in [s.id for s in subtasks] else []
            ))

        # Analysis subtask
        if any(kw in task_lower for kw in ["analyze", "evaluate", "compare", "benchmark"]):
            subtasks.append(SubTask(
                id="analyze",
                description=f"Analyze: {task}",
                assigned_agent="analyst",
                dependencies=[s.id for s in subtasks]  # Depends on all previous
            ))

        # Always add review at the end
        if subtasks:
            subtasks.append(SubTask(
                id="review",
                description="Review all outputs for quality",
                assigned_agent="reviewer",
                dependencies=[s.id for s in subtasks]
            ))
        else:
            # Default: just research and review
            subtasks = [
                SubTask(id="research", description=task, assigned_agent="researcher"),
                SubTask(id="review", description="Review findings", assigned_agent="reviewer", dependencies=["research"])
            ]

        return subtasks

    def select_agent(self, subtask: SubTask) -> BaseAgent:
        """Select appropriate agent for subtask."""
        return self.agents.get(subtask.assigned_agent, self.agents["researcher"])

    def execute_subtask(self, subtask: SubTask) -> AgentResult:
        """Execute a single subtask."""
        agent = self.select_agent(subtask)
        subtask.status = TaskStatus.IN_PROGRESS

        print(f"\n[Coordinator] Assigning to {agent.name}: {subtask.description[:50]}...")

        result = agent.process(subtask.description)
        subtask.result = result
        subtask.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED

        return result

    def execute_task(self, task: str) -> CoordinatedTask:
        """
        Execute a complex task using multiple agents.

        Args:
            task: Task description

        Returns:
            CoordinatedTask with all results
        """
        print("=" * 60)
        print(f"COORDINATED TASK: {task}")
        print("=" * 60)

        # Decompose
        subtasks = self.decompose_task(task)
        coordinated_task = CoordinatedTask(
            id=f"task_{len(self.task_history)}",
            description=task,
            subtasks=subtasks,
            status=TaskStatus.IN_PROGRESS
        )

        print(f"\nDecomposed into {len(subtasks)} subtasks:")
        for st in subtasks:
            deps = f" (depends on: {st.dependencies})" if st.dependencies else ""
            print(f"  - [{st.assigned_agent}] {st.id}{deps}")

        # Human approval checkpoint
        if self.require_human_approval:
            print("\n[CHECKPOINT] Plan created. Approve to continue? (y/n)")
            # In production, wait for actual input
            approval = "y"  # Auto-approve for demo
            if approval.lower() != "y":
                coordinated_task.status = TaskStatus.FAILED
                coordinated_task.final_result = "Cancelled by user"
                return coordinated_task

        # Execute subtasks in order (respecting dependencies)
        completed = set()

        for subtask in subtasks:
            # Check dependencies
            if not all(dep in completed for dep in subtask.dependencies):
                print(f"[Coordinator] Waiting for dependencies: {subtask.dependencies}")
                continue

            result = self.execute_subtask(subtask)

            if result.success:
                completed.add(subtask.id)
            else:
                print(f"[Coordinator] Subtask {subtask.id} failed!")
                coordinated_task.status = TaskStatus.FAILED
                break

        # Aggregate results
        if all(st.status == TaskStatus.COMPLETED for st in subtasks):
            coordinated_task.status = TaskStatus.COMPLETED
            coordinated_task.final_result = self._aggregate_results(subtasks)
        else:
            coordinated_task.status = TaskStatus.FAILED
            coordinated_task.final_result = "Some subtasks failed"

        self.task_history.append(coordinated_task)
        return coordinated_task

    def _aggregate_results(self, subtasks: List[SubTask]) -> str:
        """Aggregate results from all subtasks."""
        aggregated = "FINAL RESULTS\n" + "=" * 40 + "\n\n"

        for st in subtasks:
            if st.result:
                aggregated += f"## {st.id.upper()} ({st.assigned_agent})\n"
                aggregated += st.result.output[:500] + "\n\n"

        return aggregated

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of shared memory."""
        return {
            "facts": len(self.memory.facts),
            "context_keys": list(self.memory.context.keys()),
            "messages": len(self.memory.conversation)
        }


# Example usage
if __name__ == "__main__":
    print("Agent Coordinator Demo")
    print("=" * 60)

    # Create coordinator (disable human approval for demo)
    coordinator = AgentCoordinator(require_human_approval=False)

    # Execute a complex task
    task = "Research best practices for LLM fine-tuning and implement a training script"

    result = coordinator.execute_task(task)

    print("\n" + "=" * 60)
    print("TASK SUMMARY")
    print("=" * 60)
    print(f"Status: {result.status.value}")
    print(f"Subtasks completed: {sum(1 for st in result.subtasks if st.status == TaskStatus.COMPLETED)}/{len(result.subtasks)}")

    print("\nShared Memory:")
    summary = coordinator.get_memory_summary()
    print(f"  Facts stored: {summary['facts']}")
    print(f"  Messages: {summary['messages']}")
