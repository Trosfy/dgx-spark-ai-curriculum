"""
Agent Utilities for AI Agents & Agentic Systems

This module provides utilities for building, managing, and orchestrating AI agents,
including agent memory, conversation management, and multi-agent coordination.

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 3.6: AI Agents & Agentic Systems
"""

from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
from abc import ABC, abstractmethod


# ============================================================================
# MESSAGE AND CONVERSATION TYPES
# ============================================================================

class Role(Enum):
    """Message role types."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class Message:
    """
    Represents a message in a conversation.

    Attributes:
        role: The role of the message sender
        content: The message content
        metadata: Optional additional metadata
        timestamp: When the message was created
        message_id: Unique identifier for the message
    """
    role: Role
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLM APIs."""
        return {
            "role": self.role.value,
            "content": self.content
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ToolCall:
    """
    Represents a tool/function call by an agent.

    Attributes:
        tool_name: Name of the tool being called
        arguments: Arguments passed to the tool
        call_id: Unique identifier for this call
        result: The result of the tool call (if completed)
    """
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result: Optional[str] = None


# ============================================================================
# MEMORY SYSTEMS
# ============================================================================

class ConversationMemory:
    """
    Manages conversation history with optional summarization.

    This class maintains a sliding window of recent messages while
    optionally summarizing older messages to preserve context without
    consuming too many tokens.

    Example:
        >>> memory = ConversationMemory(max_messages=10)
        >>> memory.add_message(Role.USER, "Hello!")
        >>> memory.add_message(Role.ASSISTANT, "Hi there!")
        >>> print(len(memory.messages))
        2
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: Optional[int] = None,
        summarize_after: int = 30
    ):
        """
        Initialize conversation memory.

        Args:
            max_messages: Maximum messages to keep in memory
            max_tokens: Optional token limit for conversation
            summarize_after: Number of messages after which to summarize
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_after = summarize_after

        self.messages: List[Message] = []
        self.summary: Optional[str] = None
        self.system_message: Optional[Message] = None

    def set_system_message(self, content: str) -> None:
        """Set the system message for the conversation."""
        self.system_message = Message(role=Role.SYSTEM, content=content)

    def add_message(
        self,
        role: Role,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a message to the conversation.

        Args:
            role: The role of the sender
            content: The message content
            metadata: Optional metadata

        Returns:
            The created Message object
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Trim if exceeds max
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        return message

    def add_user_message(self, content: str) -> Message:
        """Add a user message."""
        return self.add_message(Role.USER, content)

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message."""
        return self.add_message(Role.ASSISTANT, content)

    def add_tool_result(self, tool_name: str, result: str) -> Message:
        """Add a tool result message."""
        return self.add_message(
            Role.TOOL,
            result,
            metadata={"tool_name": tool_name}
        )

    def get_messages(
        self,
        include_system: bool = True,
        include_summary: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get messages in API-compatible format.

        Args:
            include_system: Whether to include system message
            include_summary: Whether to include conversation summary

        Returns:
            List of message dictionaries
        """
        result = []

        # Add system message
        if include_system and self.system_message:
            content = self.system_message.content
            if include_summary and self.summary:
                content += f"\n\nPrevious conversation summary:\n{self.summary}"
            result.append({"role": "system", "content": content})

        # Add conversation messages
        for msg in self.messages:
            result.append(msg.to_dict())

        return result

    def clear(self, keep_system: bool = True) -> None:
        """Clear conversation history."""
        self.messages = []
        self.summary = None
        if not keep_system:
            self.system_message = None

    def get_last_n_messages(self, n: int) -> List[Message]:
        """Get the last n messages."""
        return self.messages[-n:]

    def search_messages(self, query: str) -> List[Message]:
        """Search messages for a query string."""
        query_lower = query.lower()
        return [m for m in self.messages if query_lower in m.content.lower()]

    def to_json(self) -> str:
        """Serialize memory to JSON."""
        data = {
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "metadata": m.metadata,
                    "timestamp": m.timestamp.isoformat(),
                    "message_id": m.message_id
                }
                for m in self.messages
            ],
            "summary": self.summary,
            "system_message": self.system_message.content if self.system_message else None
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ConversationMemory":
        """Load memory from JSON."""
        data = json.loads(json_str)
        memory = cls()

        if data.get("system_message"):
            memory.set_system_message(data["system_message"])

        memory.summary = data.get("summary")

        for msg_data in data.get("messages", []):
            memory.add_message(
                role=Role(msg_data["role"]),
                content=msg_data["content"],
                metadata=msg_data.get("metadata", {})
            )

        return memory


class EntityMemory:
    """
    Track entities mentioned in conversations.

    This is useful for maintaining context about people, places,
    and things discussed in the conversation.

    Example:
        >>> entity_memory = EntityMemory()
        >>> entity_memory.add_entity("John", "person", {"role": "CEO"})
        >>> info = entity_memory.get_entity("John")
    """

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add or update an entity."""
        if name not in self.entities:
            self.entities[name] = {
                "type": entity_type,
                "attributes": attributes or {},
                "mentions": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
        else:
            self.entities[name]["mentions"] += 1
            self.entities[name]["last_seen"] = datetime.now().isoformat()
            if attributes:
                self.entities[name]["attributes"].update(attributes)

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about an entity."""
        return self.entities.get(name)

    def get_all_entities(self, entity_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get all entities, optionally filtered by type."""
        if entity_type is None:
            return self.entities

        return {
            name: info
            for name, info in self.entities.items()
            if info["type"] == entity_type
        }

    def get_context_string(self) -> str:
        """Get entities as a context string for the LLM."""
        if not self.entities:
            return ""

        lines = ["Known entities:"]
        for name, info in self.entities.items():
            attrs = ", ".join(f"{k}: {v}" for k, v in info["attributes"].items())
            lines.append(f"- {name} ({info['type']}): {attrs}")

        return "\n".join(lines)


# ============================================================================
# AGENT BASE CLASSES
# ============================================================================

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    temperature: float = 0.7
    model: str = "llama3.1:70b"


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Subclasses should implement the `run` method to define
    the agent's behavior.

    Example:
        >>> class MyAgent(BaseAgent):
        ...     def run(self, input_text: str) -> str:
        ...         # Agent logic here
        ...         return "response"
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: Any = None,
        tools: Optional[List[Callable]] = None
    ):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
            llm: Language model to use
            tools: List of tool functions
        """
        self.config = config
        self.llm = llm
        self.tools = tools or []
        self.memory = ConversationMemory()
        self.memory.set_system_message(config.system_prompt)

        self.tool_map: Dict[str, Callable] = {}
        for tool in self.tools:
            name = getattr(tool, 'name', tool.__name__)
            self.tool_map[name] = tool

        self.iteration_count = 0
        self.is_running = False

    @abstractmethod
    def run(self, input_text: str) -> str:
        """
        Run the agent with the given input.

        Args:
            input_text: The user's input

        Returns:
            The agent's response
        """
        pass

    def reset(self) -> None:
        """Reset the agent's state."""
        self.memory.clear()
        self.memory.set_system_message(self.config.system_prompt)
        self.iteration_count = 0
        self.is_running = False

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            The tool's result as a string
        """
        if tool_name not in self.tool_map:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            tool = self.tool_map[tool_name]
            result = tool(**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"


class SimpleReActAgent(BaseAgent):
    """
    A simple ReAct (Reasoning + Acting) agent implementation.

    This agent follows the ReAct pattern:
    1. Think about what to do
    2. Decide on an action (tool use)
    3. Observe the result
    4. Repeat until task is complete

    Example:
        >>> agent = SimpleReActAgent(config, llm, tools)
        >>> response = agent.run("What is 25 * 48?")
    """

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response for thoughts, actions, and final answers."""
        result = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }

        # Look for Thought
        if "Thought:" in response:
            thought_match = response.split("Thought:")[-1].split("\n")[0].strip()
            result["thought"] = thought_match

        # Look for Action
        if "Action:" in response:
            action_match = response.split("Action:")[-1].split("\n")[0].strip()
            result["action"] = action_match

        # Look for Action Input
        if "Action Input:" in response:
            input_match = response.split("Action Input:")[-1].split("\n")[0].strip()
            result["action_input"] = input_match

        # Look for Final Answer
        if "Final Answer:" in response:
            answer_match = response.split("Final Answer:")[-1].strip()
            result["final_answer"] = answer_match

        return result

    def run(self, input_text: str) -> str:
        """
        Run the agent on the given input.

        Args:
            input_text: The user's input

        Returns:
            The agent's final response
        """
        self.is_running = True
        self.iteration_count = 0

        # Add user message
        self.memory.add_user_message(input_text)

        # Build tool descriptions
        tool_descriptions = []
        for name, tool in self.tool_map.items():
            doc = getattr(tool, 'description', tool.__doc__) or "No description"
            tool_descriptions.append(f"- {name}: {doc.split(chr(10))[0]}")

        tools_str = "\n".join(tool_descriptions)

        # Create the ReAct prompt
        react_prompt = f"""You are a helpful AI assistant with access to these tools:

{tools_str}

To use a tool, respond in this format:
Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: [input to the tool]

After getting the tool result, you can continue thinking and acting.

When you have the final answer, respond with:
Thought: [your final reasoning]
Final Answer: [your complete answer to the user]

User's request: {input_text}"""

        current_messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": react_prompt}
        ]

        # Agent loop
        while self.iteration_count < self.config.max_iterations:
            self.iteration_count += 1

            # Get LLM response
            if self.llm is None:
                # Simulated response for testing
                response = "Thought: I need to calculate this.\nFinal Answer: This is a test response."
            else:
                response = self.llm.invoke(current_messages)
                if hasattr(response, 'content'):
                    response = response.content

            # Parse the response
            parsed = self._parse_response(response)

            # Check for final answer
            if parsed["final_answer"]:
                self.memory.add_assistant_message(parsed["final_answer"])
                self.is_running = False
                return parsed["final_answer"]

            # Execute action if present
            if parsed["action"] and parsed["action_input"]:
                tool_result = self.execute_tool(
                    parsed["action"],
                    {"expression": parsed["action_input"]}  # Simplified for demo
                )

                # Add to messages
                current_messages.append({"role": "assistant", "content": response})
                current_messages.append({
                    "role": "user",
                    "content": f"Observation: {tool_result}"
                })

            else:
                # No action, treat response as final
                self.memory.add_assistant_message(response)
                self.is_running = False
                return response

        # Max iterations reached
        self.is_running = False
        return "I reached the maximum number of iterations without finding a complete answer."


# ============================================================================
# MULTI-AGENT COORDINATION
# ============================================================================

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str = "task"  # task, response, handoff, query
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AgentOrchestrator:
    """
    Orchestrates multiple agents working together.

    This class manages communication between agents and
    routes tasks to appropriate agents based on their capabilities.

    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> orchestrator.register_agent("researcher", research_agent)
        >>> orchestrator.register_agent("writer", writer_agent)
        >>> result = orchestrator.run_task("Write a blog about AI")
    """

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.conversation_log: List[AgentMessage] = []

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[name] = agent

    def unregister_agent(self, name: str) -> None:
        """Remove an agent from the orchestrator."""
        if name in self.agents:
            del self.agents[name]

    def send_message(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "task"
    ) -> None:
        """Send a message from one agent to another."""
        message = AgentMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type
        )
        self.message_queue.append(message)
        self.conversation_log.append(message)

    def process_messages(self) -> List[str]:
        """Process all messages in the queue."""
        results = []

        while self.message_queue:
            message = self.message_queue.pop(0)

            if message.recipient not in self.agents:
                results.append(f"Error: Agent '{message.recipient}' not found")
                continue

            agent = self.agents[message.recipient]
            response = agent.run(message.content)
            results.append(response)

            # Log the response
            response_msg = AgentMessage(
                sender=message.recipient,
                recipient=message.sender,
                content=response,
                message_type="response"
            )
            self.conversation_log.append(response_msg)

        return results

    def run_task(
        self,
        task: str,
        initial_agent: Optional[str] = None
    ) -> str:
        """
        Run a task through the agent system.

        Args:
            task: The task to complete
            initial_agent: Agent to start with (or first registered)

        Returns:
            The final result

        Example:
            >>> orchestrator = AgentOrchestrator()
            >>> orchestrator.register_agent("researcher", research_agent)
            >>> orchestrator.register_agent("writer", writer_agent)
            >>> result = orchestrator.run_task(
            ...     "Write a blog post about AI",
            ...     initial_agent="researcher"
            ... )
            >>> print(result)
            'Here is your blog post about AI...'
        """
        if not self.agents:
            return "Error: No agents registered"

        # Select initial agent
        if initial_agent and initial_agent in self.agents:
            agent_name = initial_agent
        else:
            agent_name = list(self.agents.keys())[0]

        # Send initial task
        self.send_message("orchestrator", agent_name, task)

        # Process until complete
        results = self.process_messages()

        return results[-1] if results else "No response generated"

    def get_conversation_log(self) -> List[Dict[str, Any]]:
        """Get the full conversation log."""
        return [
            {
                "sender": m.sender,
                "recipient": m.recipient,
                "content": m.content,
                "type": m.message_type,
                "timestamp": m.timestamp.isoformat()
            }
            for m in self.conversation_log
        ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_tool_result(
    tool_name: str,
    result: str,
    success: bool = True
) -> str:
    """Format a tool result for display."""
    status = "SUCCESS" if success else "FAILURE"
    return f"[Tool: {tool_name}] [{status}]\n{result}"


def create_system_prompt(
    role: str,
    personality: str,
    capabilities: List[str],
    constraints: Optional[List[str]] = None
) -> str:
    """
    Create a system prompt for an agent.

    Args:
        role: The agent's role
        personality: Description of personality traits
        capabilities: List of what the agent can do
        constraints: Optional list of limitations

    Returns:
        Formatted system prompt
    """
    prompt_parts = [
        f"You are {role}.",
        "",
        f"Personality: {personality}",
        "",
        "Capabilities:",
    ]

    for cap in capabilities:
        prompt_parts.append(f"- {cap}")

    if constraints:
        prompt_parts.append("")
        prompt_parts.append("Constraints:")
        for con in constraints:
            prompt_parts.append(f"- {con}")

    return "\n".join(prompt_parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    This is a rough approximation. For accurate counts,
    use the model's tokenizer.

    Args:
        text: The text to estimate

    Returns:
        Estimated token count
    """
    # Rough approximation: 1 token ≈ 4 characters for English
    return len(text) // 4


if __name__ == "__main__":
    print("Agent Utilities Demo")
    print("=" * 50)

    # Create memory
    memory = ConversationMemory()
    memory.set_system_message("You are a helpful AI assistant.")
    memory.add_user_message("Hello!")
    memory.add_assistant_message("Hi there! How can I help you today?")

    print("\n1. Conversation Memory:")
    for msg in memory.get_messages():
        print(f"  {msg['role']}: {msg['content'][:50]}...")

    # Create entity memory
    entity_mem = EntityMemory()
    entity_mem.add_entity("DGX Spark", "hardware", {
        "memory": "128GB",
        "gpu": "Blackwell GB10"
    })
    entity_mem.add_entity("LangChain", "framework", {
        "language": "Python",
        "purpose": "LLM applications"
    })

    print("\n2. Entity Memory:")
    print(entity_mem.get_context_string())

    # Create agent config
    config = AgentConfig(
        name="ResearchAgent",
        description="An agent that researches topics",
        system_prompt="You are a research assistant that finds accurate information.",
        tools=["web_search", "calculate"],
        temperature=0.3
    )

    print("\n3. Agent Config:")
    print(f"  Name: {config.name}")
    print(f"  Tools: {config.tools}")

    # Create system prompt
    prompt = create_system_prompt(
        role="a research assistant",
        personality="Thorough, accurate, and helpful",
        capabilities=[
            "Search the web for information",
            "Analyze documents",
            "Summarize findings"
        ],
        constraints=[
            "Always cite sources",
            "Admit when unsure"
        ]
    )

    print("\n4. Generated System Prompt:")
    print(prompt)
