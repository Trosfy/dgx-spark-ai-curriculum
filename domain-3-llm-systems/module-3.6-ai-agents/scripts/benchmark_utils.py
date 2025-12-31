"""
Agent Benchmarking Utilities

This module provides tools for evaluating and benchmarking AI agents,
including metrics for retrieval quality, generation accuracy, and tool use.

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 3.6: AI Agents & Agentic Systems
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import statistics
from pathlib import Path
from enum import Enum
import re


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TestCategory(Enum):
    """Categories of agent test cases."""
    FACTUAL_RETRIEVAL = "factual_retrieval"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    TOOL_USE = "tool_use"
    CALCULATION = "calculation"
    CODE_GENERATION = "code_generation"
    SYNTHESIS = "synthesis"


class Difficulty(Enum):
    """Difficulty levels for test cases."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TestCase:
    """
    A single test case for agent evaluation.

    Attributes:
        id: Unique identifier
        query: The input query to the agent
        expected_answer: The expected response (for comparison)
        category: Type of test (retrieval, reasoning, etc.)
        difficulty: Easy, medium, or hard
        keywords: Keywords that should appear in the response
        source_documents: Documents containing the answer
        requires_tool: Name of tool required (if any)
    """
    id: str
    query: str
    expected_answer: str
    category: TestCategory
    difficulty: Difficulty = Difficulty.MEDIUM
    keywords: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    requires_tool: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """
    Result of running a single test case.

    Attributes:
        test_case: The original test case
        agent_response: The agent's response
        passed: Whether the test passed
        score: Numerical score (0.0 to 1.0)
        latency_ms: Response time in milliseconds
        metrics: Detailed metrics (precision, recall, etc.)
        error: Error message if test failed
    """
    test_case: TestCase
    agent_response: str
    passed: bool
    score: float
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResults:
    """
    Aggregated results from a benchmark run.

    Attributes:
        name: Name of the benchmark
        results: Individual test results
        overall_score: Average score across all tests
        category_scores: Scores broken down by category
        timing_stats: Latency statistics
    """
    name: str
    results: List[TestResult]
    overall_score: float
    category_scores: Dict[str, float]
    timing_stats: Dict[str, float]
    started_at: datetime
    completed_at: datetime


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def keyword_match_score(
    response: str,
    keywords: List[str],
    case_sensitive: bool = False
) -> float:
    """
    Calculate what fraction of keywords appear in the response.

    Args:
        response: The agent's response
        keywords: List of keywords to check
        case_sensitive: Whether matching is case-sensitive

    Returns:
        Score from 0.0 to 1.0

    Example:
        >>> score = keyword_match_score("The DGX Spark has 128GB memory", ["128GB", "DGX"])
        >>> print(score)
        1.0
    """
    if not keywords:
        return 1.0

    if not case_sensitive:
        response = response.lower()
        keywords = [k.lower() for k in keywords]

    matches = sum(1 for kw in keywords if kw in response)
    return matches / len(keywords)


def exact_match_score(response: str, expected: str) -> float:
    """
    Check if response exactly matches expected answer.

    Args:
        response: The agent's response
        expected: Expected answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Normalize whitespace
    response_norm = " ".join(response.strip().split())
    expected_norm = " ".join(expected.strip().split())

    return 1.0 if response_norm == expected_norm else 0.0


def contains_match_score(response: str, expected: str) -> float:
    """
    Check if response contains the expected answer.

    Args:
        response: The agent's response
        expected: Expected answer (substring)

    Returns:
        1.0 if expected is in response, 0.0 otherwise
    """
    return 1.0 if expected.lower() in response.lower() else 0.0


def f1_score(
    response_tokens: List[str],
    expected_tokens: List[str]
) -> float:
    """
    Calculate F1 score between response and expected tokens.

    Args:
        response_tokens: Tokens from the response
        expected_tokens: Tokens from expected answer

    Returns:
        F1 score from 0.0 to 1.0
    """
    if not response_tokens or not expected_tokens:
        return 0.0

    response_set = set(t.lower() for t in response_tokens)
    expected_set = set(t.lower() for t in expected_tokens)

    intersection = response_set & expected_set

    if not intersection:
        return 0.0

    precision = len(intersection) / len(response_set)
    recall = len(intersection) / len(expected_set)

    return 2 * precision * recall / (precision + recall)


def rouge_l_score(response: str, expected: str) -> float:
    """
    Calculate ROUGE-L score (longest common subsequence).

    Args:
        response: The agent's response
        expected: Expected answer

    Returns:
        ROUGE-L F1 score from 0.0 to 1.0
    """
    def lcs_length(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    response_tokens = response.lower().split()
    expected_tokens = expected.lower().split()

    if not response_tokens or not expected_tokens:
        return 0.0

    lcs = lcs_length(response.lower(), expected.lower())

    if lcs == 0:
        return 0.0

    precision = lcs / len(response)
    recall = lcs / len(expected)

    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def semantic_similarity_score(
    response: str,
    expected: str,
    embedding_model: Any = None
) -> float:
    """
    Calculate semantic similarity using embeddings.

    Args:
        response: The agent's response
        expected: Expected answer
        embedding_model: Model to generate embeddings

    Returns:
        Cosine similarity from 0.0 to 1.0
    """
    if embedding_model is None:
        # Fallback to word overlap if no embedding model
        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())
        intersection = response_words & expected_words
        union = response_words | expected_words
        return len(intersection) / len(union) if union else 0.0

    try:
        import numpy as np

        resp_emb = embedding_model.embed_query(response)
        exp_emb = embedding_model.embed_query(expected)

        resp_emb = np.array(resp_emb)
        exp_emb = np.array(exp_emb)

        # Cosine similarity
        similarity = np.dot(resp_emb, exp_emb) / (
            np.linalg.norm(resp_emb) * np.linalg.norm(exp_emb)
        )

        return float(max(0.0, similarity))  # Clamp to [0, 1]

    except Exception:
        return 0.0


def tool_use_score(
    response: str,
    required_tool: str,
    tool_calls: Optional[List[str]] = None
) -> float:
    """
    Evaluate if the agent correctly used the required tool.

    Args:
        response: The agent's response
        required_tool: Name of the required tool
        tool_calls: List of tools the agent called

    Returns:
        Score from 0.0 to 1.0
    """
    if tool_calls:
        return 1.0 if required_tool in tool_calls else 0.0

    # Check if tool name appears in response (fallback)
    return 1.0 if required_tool.lower() in response.lower() else 0.0


# ============================================================================
# EVALUATOR CLASS
# ============================================================================

class AgentEvaluator:
    """
    Comprehensive evaluator for AI agents.

    This class runs test cases against an agent and computes
    various metrics to assess performance.

    Example:
        >>> evaluator = AgentEvaluator(agent_func, embedding_model)
        >>> results = evaluator.run_benchmark(test_cases)
        >>> print(f"Overall score: {results.overall_score:.2%}")
    """

    def __init__(
        self,
        agent_func: Callable[[str], str],
        embedding_model: Any = None,
        verbose: bool = True,
        pass_threshold: float = 0.6
    ):
        """
        Initialize the evaluator.

        Args:
            agent_func: Function that takes a query and returns response
            embedding_model: Optional model for semantic similarity
            verbose: Whether to print progress
            pass_threshold: Minimum score to consider a test passed (default: 0.6)
        """
        self.agent_func = agent_func
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.pass_threshold = pass_threshold

    def evaluate_single(self, test_case: TestCase) -> TestResult:
        """
        Evaluate a single test case.

        Args:
            test_case: The test case to evaluate

        Returns:
            TestResult with scores and metrics
        """
        start_time = time.time()
        error = None

        try:
            response = self.agent_func(test_case.query)
        except Exception as e:
            response = ""
            error = str(e)

        latency_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        metrics = {}

        # Keyword matching
        if test_case.keywords:
            metrics["keyword_match"] = keyword_match_score(
                response, test_case.keywords
            )

        # Contains expected
        metrics["contains_expected"] = contains_match_score(
            response, test_case.expected_answer
        )

        # F1 score on tokens
        response_tokens = response.split()
        expected_tokens = test_case.expected_answer.split()
        metrics["f1"] = f1_score(response_tokens, expected_tokens)

        # Semantic similarity (if model available)
        if self.embedding_model:
            metrics["semantic_similarity"] = semantic_similarity_score(
                response, test_case.expected_answer, self.embedding_model
            )

        # Tool use (if required)
        if test_case.requires_tool:
            metrics["tool_use"] = tool_use_score(
                response, test_case.requires_tool
            )

        # Calculate overall score based on category
        if test_case.category == TestCategory.FACTUAL_RETRIEVAL:
            score = (
                metrics.get("keyword_match", 0) * 0.4 +
                metrics.get("contains_expected", 0) * 0.4 +
                metrics.get("f1", 0) * 0.2
            )
        elif test_case.category == TestCategory.TOOL_USE:
            score = (
                metrics.get("tool_use", 0) * 0.5 +
                metrics.get("keyword_match", 0) * 0.3 +
                metrics.get("f1", 0) * 0.2
            )
        elif test_case.category == TestCategory.CALCULATION:
            # For calculations, exact answer is important
            score = metrics.get("contains_expected", 0)
        else:
            # Default scoring
            score = sum(metrics.values()) / len(metrics) if metrics else 0.0

        # Determine pass/fail
        passed = score >= self.pass_threshold

        return TestResult(
            test_case=test_case,
            agent_response=response,
            passed=passed,
            score=score,
            latency_ms=latency_ms,
            metrics=metrics,
            error=error
        )

    def run_benchmark(
        self,
        test_cases: List[TestCase],
        name: str = "Agent Benchmark"
    ) -> BenchmarkResults:
        """
        Run a full benchmark across all test cases.

        Args:
            test_cases: List of test cases to run
            name: Name for this benchmark run

        Returns:
            BenchmarkResults with aggregated metrics
        """
        started_at = datetime.now()
        results = []

        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"Running test {i+1}/{len(test_cases)}: {test_case.id}")

            result = self.evaluate_single(test_case)
            results.append(result)

            if self.verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"  {status} (score: {result.score:.2f}, latency: {result.latency_ms:.0f}ms)")

        completed_at = datetime.now()

        # Calculate category scores
        category_scores = {}
        for category in TestCategory:
            category_results = [r for r in results if r.test_case.category == category]
            if category_results:
                category_scores[category.value] = statistics.mean(
                    r.score for r in category_results
                )

        # Calculate timing stats
        latencies = [r.latency_ms for r in results]
        timing_stats = {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }

        # Overall score
        overall_score = statistics.mean(r.score for r in results) if results else 0.0

        return BenchmarkResults(
            name=name,
            results=results,
            overall_score=overall_score,
            category_scores=category_scores,
            timing_stats=timing_stats,
            started_at=started_at,
            completed_at=completed_at
        )


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: BenchmarkResults) -> str:
    """
    Generate a human-readable report from benchmark results.

    Args:
        results: BenchmarkResults object

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        f"AGENT BENCHMARK REPORT: {results.name}",
        "=" * 60,
        "",
        f"Run Time: {results.started_at.strftime('%Y-%m-%d %H:%M:%S')} - "
        f"{results.completed_at.strftime('%H:%M:%S')}",
        f"Total Tests: {len(results.results)}",
        "",
        "-" * 40,
        "OVERALL RESULTS",
        "-" * 40,
        f"Overall Score: {results.overall_score:.1%}",
        f"Tests Passed: {sum(1 for r in results.results if r.passed)}/{len(results.results)}",
        "",
        "-" * 40,
        "SCORES BY CATEGORY",
        "-" * 40,
    ]

    for category, score in sorted(results.category_scores.items()):
        lines.append(f"  {category}: {score:.1%}")

    lines.extend([
        "",
        "-" * 40,
        "LATENCY STATISTICS",
        "-" * 40,
        f"  Mean: {results.timing_stats['mean_latency_ms']:.0f}ms",
        f"  Median: {results.timing_stats['median_latency_ms']:.0f}ms",
        f"  Min: {results.timing_stats['min_latency_ms']:.0f}ms",
        f"  Max: {results.timing_stats['max_latency_ms']:.0f}ms",
        f"  Std Dev: {results.timing_stats['std_latency_ms']:.0f}ms",
        "",
        "-" * 40,
        "FAILED TESTS",
        "-" * 40,
    ])

    failed = [r for r in results.results if not r.passed]
    if not failed:
        lines.append("  All tests passed!")
    else:
        for r in failed:
            lines.append(f"  - {r.test_case.id}: score={r.score:.2f}")
            if r.error:
                lines.append(f"    Error: {r.error}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def save_results_json(results: BenchmarkResults, path: str) -> None:
    """
    Save benchmark results to a JSON file.

    Args:
        results: BenchmarkResults object
        path: Path to save the JSON file
    """
    data = {
        "name": results.name,
        "overall_score": results.overall_score,
        "category_scores": results.category_scores,
        "timing_stats": results.timing_stats,
        "started_at": results.started_at.isoformat(),
        "completed_at": results.completed_at.isoformat(),
        "results": [
            {
                "test_id": r.test_case.id,
                "query": r.test_case.query,
                "expected": r.test_case.expected_answer,
                "response": r.agent_response,
                "passed": r.passed,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "metrics": r.metrics,
                "error": r.error
            }
            for r in results.results
        ]
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# TEST CASE LOADING
# ============================================================================

def load_test_cases_from_json(path: str) -> List[TestCase]:
    """
    Load test cases from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        List of TestCase objects

    Expected JSON format:
    {
        "test_cases": [
            {
                "id": "test_001",
                "query": "What is X?",
                "expected_answer": "X is Y",
                "category": "factual_retrieval",
                "difficulty": "easy",
                "keywords": ["X", "Y"]
            }
        ]
    }
    """
    with open(path, 'r') as f:
        data = json.load(f)

    test_cases = []
    for tc in data.get("test_cases", []):
        test_cases.append(TestCase(
            id=tc["id"],
            query=tc["query"],
            expected_answer=tc.get("expected_answer", ""),
            category=TestCategory(tc.get("category", "factual_retrieval")),
            difficulty=Difficulty(tc.get("difficulty", "medium")),
            keywords=tc.get("keywords", []),
            source_documents=tc.get("source_documents", tc.get("source_document", [])),
            requires_tool=tc.get("requires_tool"),
            metadata=tc.get("metadata", {})
        ))

    return test_cases


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_agents(
    results_list: List[BenchmarkResults],
    agent_names: List[str]
) -> str:
    """
    Generate a comparison report for multiple agents.

    Args:
        results_list: List of BenchmarkResults from different agents
        agent_names: Names of the agents being compared

    Returns:
        Formatted comparison report
    """
    if len(results_list) != len(agent_names):
        raise ValueError("Number of results must match number of agent names")

    lines = [
        "=" * 70,
        "AGENT COMPARISON REPORT",
        "=" * 70,
        "",
        "-" * 50,
        "OVERALL SCORES",
        "-" * 50,
    ]

    # Sort by score
    ranked = sorted(
        zip(agent_names, results_list),
        key=lambda x: x[1].overall_score,
        reverse=True
    )

    for i, (name, results) in enumerate(ranked, 1):
        lines.append(f"  {i}. {name}: {results.overall_score:.1%}")

    lines.extend([
        "",
        "-" * 50,
        "CATEGORY COMPARISON",
        "-" * 50,
    ])

    # Get all categories
    all_categories = set()
    for results in results_list:
        all_categories.update(results.category_scores.keys())

    for category in sorted(all_categories):
        lines.append(f"\n  {category}:")
        cat_scores = []
        for name, results in zip(agent_names, results_list):
            score = results.category_scores.get(category, 0)
            cat_scores.append((name, score))

        for name, score in sorted(cat_scores, key=lambda x: x[1], reverse=True):
            lines.append(f"    {name}: {score:.1%}")

    lines.extend([
        "",
        "-" * 50,
        "LATENCY COMPARISON",
        "-" * 50,
    ])

    latency_data = [
        (name, results.timing_stats['mean_latency_ms'])
        for name, results in zip(agent_names, results_list)
    ]

    for name, latency in sorted(latency_data, key=lambda x: x[1]):
        lines.append(f"  {name}: {latency:.0f}ms (mean)")

    lines.extend([
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print("Benchmark Utilities Demo")
    print("=" * 50)

    # Create sample test cases
    test_cases = [
        TestCase(
            id="test_001",
            query="What is the memory capacity of DGX Spark?",
            expected_answer="128GB unified memory",
            category=TestCategory.FACTUAL_RETRIEVAL,
            difficulty=Difficulty.EASY,
            keywords=["128GB", "unified", "memory"]
        ),
        TestCase(
            id="test_002",
            query="Calculate 25 * 48",
            expected_answer="1200",
            category=TestCategory.CALCULATION,
            difficulty=Difficulty.EASY,
            requires_tool="calculator"
        ),
    ]

    # Create a simple mock agent
    def mock_agent(query: str) -> str:
        if "memory" in query.lower():
            return "The DGX Spark has 128GB of unified LPDDR5X memory."
        elif "calculate" in query.lower():
            return "25 * 48 = 1200"
        return "I don't know."

    # Run evaluation
    print("\nRunning benchmark...")
    evaluator = AgentEvaluator(mock_agent, verbose=True)
    results = evaluator.run_benchmark(test_cases, "Demo Benchmark")

    # Generate report
    print("\n" + generate_report(results))
