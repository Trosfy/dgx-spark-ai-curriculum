"""
AI Agents Module Scripts

This package provides utilities for RAG pipelines, custom tools,
agent management, and benchmarking for the DGX Spark AI Curriculum.

Modules:
    rag_utils: Document loading, chunking, embedding, and RAG pipeline
    custom_tools: Safe calculator, code execution, file reading, API calls
    agent_utils: Agent memory, conversation management, multi-agent coordination
    benchmark_utils: Agent evaluation metrics and benchmarking framework

Example:
    >>> from scripts.rag_utils import RAGPipeline, DocumentLoader
    >>> from scripts.custom_tools import calculate, get_all_tools
    >>> from scripts.agent_utils import ConversationMemory, SimpleReActAgent
    >>> from scripts.benchmark_utils import AgentEvaluator, TestCase

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 13
"""

# RAG utilities
from .rag_utils import (
    Document,
    Chunk,
    DocumentLoader,
    TextChunker,
    SimpleVectorStore,
    RAGPipeline,
    compute_retrieval_metrics,
)

# Custom tools for agents
from .custom_tools import (
    calculate,
    web_search,
    execute_python,
    read_file,
    call_api,
    get_current_datetime,
    json_parser,
    text_analyzer,
    get_all_tools,
    get_tool_descriptions,
    SafeCalculator,
    SafePythonExecutor,
    CalculatorResult,
)

# Agent utilities
from .agent_utils import (
    Role,
    Message,
    ToolCall,
    ConversationMemory,
    EntityMemory,
    AgentConfig,
    BaseAgent,
    SimpleReActAgent,
    AgentOrchestrator,
    AgentMessage,
    create_system_prompt,
    format_tool_result,
    estimate_tokens,
)

# Benchmarking utilities
from .benchmark_utils import (
    TestCategory,
    Difficulty,
    TestCase,
    TestResult,
    BenchmarkResults,
    AgentEvaluator,
    keyword_match_score,
    exact_match_score,
    contains_match_score,
    f1_score,
    rouge_l_score,
    semantic_similarity_score,
    tool_use_score,
    generate_report,
    save_results_json,
    load_test_cases_from_json,
    compare_agents,
)

__all__ = [
    # rag_utils
    "Document",
    "Chunk",
    "DocumentLoader",
    "TextChunker",
    "SimpleVectorStore",
    "RAGPipeline",
    "compute_retrieval_metrics",
    # custom_tools
    "calculate",
    "web_search",
    "execute_python",
    "read_file",
    "call_api",
    "get_current_datetime",
    "json_parser",
    "text_analyzer",
    "get_all_tools",
    "get_tool_descriptions",
    "SafeCalculator",
    "SafePythonExecutor",
    "CalculatorResult",
    # agent_utils
    "Role",
    "Message",
    "ToolCall",
    "ConversationMemory",
    "EntityMemory",
    "AgentConfig",
    "BaseAgent",
    "SimpleReActAgent",
    "AgentOrchestrator",
    "AgentMessage",
    "create_system_prompt",
    "format_tool_result",
    "estimate_tokens",
    # benchmark_utils
    "TestCategory",
    "Difficulty",
    "TestCase",
    "TestResult",
    "BenchmarkResults",
    "AgentEvaluator",
    "keyword_match_score",
    "exact_match_score",
    "contains_match_score",
    "f1_score",
    "rouge_l_score",
    "semantic_similarity_score",
    "tool_use_score",
    "generate_report",
    "save_results_json",
    "load_test_cases_from_json",
    "compare_agents",
]
