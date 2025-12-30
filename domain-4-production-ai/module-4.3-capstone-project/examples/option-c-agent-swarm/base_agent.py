#!/usr/bin/env python3
"""
Base Agent Implementation

Foundation class for all specialized agents.
This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
from datetime import datetime


class MessageType(Enum):
    """Types of messages agents can send."""
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    receiver: str
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from an agent task."""
    success: bool
    output: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class SharedMemory:
    """
    Shared memory for agent communication.

    In production, use vector store for semantic search.
    """

    def __init__(self):
        self.facts: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.conversation: List[AgentMessage] = []

    def add_fact(self, key: str, value: Any, source: str):
        """Add a fact to shared memory."""
        self.facts.append({
            "key": key,
            "value": value,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })

    def get_facts(self, query: str = None, limit: int = 5) -> List[Dict]:
        """Retrieve relevant facts."""
        if query is None:
            return self.facts[-limit:]

        # Simple keyword matching - use vector search in production
        query_lower = query.lower()
        relevant = [
            f for f in self.facts
            if query_lower in str(f.get("key", "")).lower()
            or query_lower in str(f.get("value", "")).lower()
        ]
        return relevant[-limit:]

    def set_context(self, key: str, value: Any):
        """Set shared context value."""
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get shared context value."""
        return self.context.get(key, default)

    def add_message(self, message: AgentMessage):
        """Add message to conversation history."""
        self.conversation.append(message)

    def get_conversation(self, limit: int = 10) -> List[AgentMessage]:
        """Get recent conversation messages."""
        return self.conversation[-limit:]


class BaseAgent(ABC):
    """
    Base class for all agents.

    Provides:
    - Standardized interface
    - Tool registration
    - Memory access
    - Message handling
    """

    def __init__(
        self,
        name: str,
        description: str,
        shared_memory: SharedMemory = None
    ):
        self.name = name
        self.description = description
        self.memory = shared_memory or SharedMemory()
        self.tools: Dict[str, Callable] = {}

        # Model (loaded lazily)
        self._model = None
        self._tokenizer = None

    @property
    def system_prompt(self) -> str:
        """Generate agent-specific system prompt."""
        tools_desc = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])

        return f"""You are {self.name}, {self.description}.

Your capabilities:
{tools_desc if tools_desc else 'No specific tools available'}

You are part of a multi-agent system. Be concise and focused.
When you need to use a tool, respond with:
TOOL: <tool_name>
ARGS: <json_args>

Always explain your reasoning briefly."""

    def register_tool(self, name: str, func: Callable):
        """Register a tool for this agent."""
        self.tools[name] = func
        print(f"[{self.name}] Registered tool: {name}")

    @abstractmethod
    def process(self, task: str) -> AgentResult:
        """
        Process a task (implemented by subclasses).

        Args:
            task: The task to process

        Returns:
            AgentResult with output
        """
        pass

    def send_message(self, receiver: str, content: str, msg_type: MessageType = MessageType.REQUEST):
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            receiver=receiver,
            content=content,
            message_type=msg_type
        )
        self.memory.add_message(message)
        return message

    def receive_messages(self, limit: int = 5) -> List[AgentMessage]:
        """Get messages addressed to this agent."""
        all_messages = self.memory.get_conversation(limit * 2)
        return [m for m in all_messages if m.receiver == self.name][-limit:]

    def add_to_memory(self, key: str, value: Any):
        """Add information to shared memory."""
        self.memory.add_fact(key, value, source=self.name)

    def query_memory(self, query: str) -> List[Dict]:
        """Query shared memory for relevant information."""
        return self.memory.get_facts(query)

    def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM (mock for demo)."""
        # In production, use loaded model
        return f"[{self.name}] Processing: {prompt[:100]}..."

    def _handle_tool_call(self, response: str) -> str:
        """Handle tool calls in response."""
        import re

        tool_match = re.search(r"TOOL:\s*(\w+)", response)
        args_match = re.search(r"ARGS:\s*({.*?})", response, re.DOTALL)

        if not tool_match:
            return response

        tool_name = tool_match.group(1)

        if tool_name not in self.tools:
            return response + f"\n[Tool '{tool_name}' not available]"

        try:
            args = json.loads(args_match.group(1)) if args_match else {}
            result = self.tools[tool_name](**args)
            return response + f"\n[Tool Result: {result}]"
        except Exception as e:
            return response + f"\n[Tool Error: {e}]"


# Example usage
if __name__ == "__main__":
    print("Base Agent Demo")
    print("=" * 50)

    # Create shared memory
    memory = SharedMemory()

    # Add some facts
    memory.add_fact("project", "AI Assistant", "system")
    memory.add_fact("deadline", "2 weeks", "user")

    print(f"Facts in memory: {len(memory.facts)}")
    print(f"Context: {memory.context}")
