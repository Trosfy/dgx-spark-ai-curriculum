#!/usr/bin/env python3
"""
Simple Domain-Specific AI Assistant

A minimal implementation demonstrating the core components
of a domain-specific AI assistant.

This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
import json


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # For tool messages


@dataclass
class Tool:
    """A tool the assistant can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: callable


class SimpleRAG:
    """
    Simple RAG implementation (mock for demo).

    In your capstone, replace this with a real vector-based RAG system.
    """

    def __init__(self):
        # Mock knowledge base
        self.documents = [
            {
                "content": "AWS S3 is a scalable object storage service. Use 'aws s3 mb' to create buckets.",
                "source": "AWS Documentation"
            },
            {
                "content": "AWS Lambda is a serverless compute service. Functions timeout after 15 minutes max.",
                "source": "AWS Lambda Guide"
            },
            {
                "content": "AWS EC2 provides virtual servers. Instance types range from t2.micro to x1e.32xlarge.",
                "source": "EC2 User Guide"
            },
        ]

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Retrieve relevant documents (mock implementation).

        In production, use vector similarity search.
        """
        # Simple keyword matching for demo
        query_lower = query.lower()

        scored = []
        for doc in self.documents:
            # Count keyword matches
            score = sum(1 for word in query_lower.split()
                       if word in doc["content"].lower())
            if score > 0:
                scored.append((score, doc))

        # Return top k
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k]]


class SimpleAssistant:
    """
    A simple domain-specific AI assistant.

    Components:
    - RAG for knowledge retrieval
    - Tool calling capability
    - Conversation memory
    """

    def __init__(
        self,
        domain: str = "AWS",
        model_name: str = "meta-llama/Llama-3.3-8B-Instruct",
        load_model: bool = True,
    ):
        self.domain = domain
        self.model_name = model_name
        self.rag = SimpleRAG()
        self.tools: Dict[str, Tool] = {}
        self.conversation: List[Message] = []

        self._model = None
        self._tokenizer = None

        if load_model:
            self._load_model()

    @property
    def system_prompt(self) -> str:
        """Generate system prompt."""
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools.values()
        ])

        return f"""You are an expert {self.domain} assistant. Your role is to:
1. Answer questions accurately using your knowledge
2. Use tools when they can help provide better answers
3. Be concise but thorough
4. Cite sources when using retrieved information

Available tools:
{tools_desc if tools_desc else "No tools available"}

When you need to use a tool, respond with:
TOOL: <tool_name>
ARGS: <json_args>

Always explain your reasoning."""

    def _load_model(self):
        """Load the language model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            print(f"Loading model: {self.model_name}")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            print("✅ Model loaded")

        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("Running in demo mode without LLM")

    def register_tool(self, tool: Tool):
        """Register a tool for the assistant to use."""
        self.tools[tool.name] = tool
        print(f"✅ Registered tool: {tool.name}")

    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.

        Args:
            user_message: The user's input

        Returns:
            Assistant's response
        """
        # Add to conversation
        self.conversation.append(Message(role="user", content=user_message))

        # Retrieve relevant context
        context = self.rag.retrieve(user_message)
        context_str = "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['content']}"
            for doc in context
        ]) if context else ""

        # Generate response
        if self._model is not None:
            response = self._generate_with_model(user_message, context_str)
        else:
            # Demo mode without model
            response = self._demo_response(user_message, context_str)

        # Check for tool calls
        if "TOOL:" in response:
            response = self._handle_tool_call(response)

        # Add to conversation
        self.conversation.append(Message(role="assistant", content=response))

        return response

    def _generate_with_model(self, user_message: str, context: str) -> str:
        """Generate response using the loaded model."""
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context:\n{context}"
            })

        # Add recent conversation
        for msg in self.conversation[-4:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        # Generate
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _demo_response(self, user_message: str, context: str) -> str:
        """Generate demo response without model."""
        if context:
            return f"""Based on the available information:

{context}

This is a demo response. In the full implementation, the LLM would
generate a more helpful answer based on this context."""
        else:
            return f"""I understand you're asking about: {user_message}

This is a demo response. In the full implementation with the LLM loaded,
I would provide a detailed, helpful answer."""

    def _handle_tool_call(self, response: str) -> str:
        """Handle tool calls in the response."""
        import re

        # Parse tool call
        tool_match = re.search(r"TOOL:\s*(\w+)", response)
        args_match = re.search(r"ARGS:\s*({.*?})", response, re.DOTALL)

        if not tool_match:
            return response

        tool_name = tool_match.group(1)

        if tool_name not in self.tools:
            return response + f"\n\n(Tool '{tool_name}' not found)"

        try:
            args = json.loads(args_match.group(1)) if args_match else {}
            tool = self.tools[tool_name]
            result = tool.function(**args)

            return response + f"\n\n**Tool Result ({tool_name}):**\n{result}"
        except Exception as e:
            return response + f"\n\n(Tool error: {e})"

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation = []
        print("✅ Conversation cleared")


# Example usage
if __name__ == "__main__":
    # Create assistant (without loading model for quick demo)
    assistant = SimpleAssistant(domain="AWS", load_model=False)

    # Add a simple tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)  # Note: Use a safer eval in production!
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error: {e}"

    assistant.register_tool(Tool(
        name="calculate",
        description="Evaluate a mathematical expression",
        parameters={"expression": {"type": "string"}},
        function=calculate,
    ))

    # Test conversation
    print("\n" + "="*50)
    print("SIMPLE ASSISTANT DEMO")
    print("="*50)

    questions = [
        "How do I create an S3 bucket?",
        "What is the Lambda timeout limit?",
    ]

    for q in questions:
        print(f"\n👤 User: {q}")
        response = assistant.chat(q)
        print(f"\n🤖 Assistant: {response}")
        print("-"*50)
