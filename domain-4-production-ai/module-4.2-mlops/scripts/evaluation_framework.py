"""
Custom Evaluation Framework for LLMs.

This module provides a flexible framework for creating custom
evaluation suites with support for multiple metric types including
LLM-as-a-Judge evaluation.

Example usage:
    from scripts.evaluation_framework import CustomEvaluator, EvalSample

    # Create evaluator with your model
    evaluator = CustomEvaluator(
        model_fn=lambda prompt: your_model.generate(prompt),
        name="My Evaluation"
    )

    # Define test samples
    samples = [
        EvalSample(
            input="What is 2+2?",
            expected="4",
            category="math"
        ),
    ]

    # Run evaluation
    results = evaluator.evaluate_dataset(samples, metric_type=MetricType.CONTAINS)
    evaluator.print_summary()
"""

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MetricType(Enum):
    """Types of evaluation metrics."""

    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    NUMERIC = "numeric"
    F1 = "f1"
    LLM_JUDGE = "llm_judge"
    CUSTOM = "custom"


@dataclass
class EvalSample:
    """
    A single evaluation example.

    Attributes:
        input: The prompt or question to evaluate
        expected: The expected output (for reference-based evaluation)
        metadata: Additional information (patterns, tolerances, etc.)
        category: Category for grouping results

    Example:
        >>> sample = EvalSample(
        ...     input="What is the capital of France?",
        ...     expected="Paris",
        ...     category="geography"
        ... )
    """

    input: str
    expected: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    category: str = "default"


@dataclass
class EvalResult:
    """
    Result of evaluating a single sample.

    Attributes:
        sample: The original sample
        output: The model's output
        score: Numeric score (0-1)
        passed: Whether the sample passed (score >= threshold)
        details: Additional evaluation details
        latency_ms: Response time in milliseconds
    """

    sample: EvalSample
    output: str
    score: float
    passed: bool
    details: Dict = field(default_factory=dict)
    latency_ms: float = 0.0


class EvaluationMetrics:
    """Collection of evaluation metric functions."""

    @staticmethod
    def exact_match(output: str, expected: str, case_sensitive: bool = False) -> float:
        """
        Check if output exactly matches expected.

        Args:
            output: Model output
            expected: Expected string
            case_sensitive: Whether comparison is case-sensitive

        Returns:
            1.0 if match, 0.0 otherwise
        """
        if not case_sensitive:
            output = output.lower().strip()
            expected = expected.lower().strip()
        else:
            output = output.strip()
            expected = expected.strip()
        return 1.0 if output == expected else 0.0

    @staticmethod
    def contains(output: str, expected: str, case_sensitive: bool = False) -> float:
        """
        Check if output contains expected string.

        Args:
            output: Model output
            expected: String to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            1.0 if found, 0.0 otherwise
        """
        if not case_sensitive:
            output = output.lower()
            expected = expected.lower()
        return 1.0 if expected in output else 0.0

    @staticmethod
    def regex_match(output: str, pattern: str) -> float:
        """
        Check if output matches regex pattern.

        Args:
            output: Model output
            pattern: Regex pattern

        Returns:
            1.0 if match found, 0.0 otherwise
        """
        try:
            return 1.0 if re.search(pattern, output) else 0.0
        except re.error:
            return 0.0

    @staticmethod
    def numeric_close(
        output: str,
        expected: float,
        tolerance: float = 0.01
    ) -> float:
        """
        Check if extracted number is close to expected.

        Args:
            output: Model output (will extract numbers)
            expected: Expected numeric value
            tolerance: Allowed absolute difference

        Returns:
            1.0 if any extracted number is within tolerance, 0.0 otherwise
        """
        numbers = re.findall(r'-?\d+\.?\d*', output)
        if not numbers:
            return 0.0

        for num_str in numbers:
            try:
                num = float(num_str)
                if abs(num - expected) <= tolerance:
                    return 1.0
            except ValueError:
                continue
        return 0.0

    @staticmethod
    def f1_score(output: str, expected: str) -> float:
        """
        Calculate token-level F1 score.

        Args:
            output: Model output
            expected: Expected output

        Returns:
            F1 score between 0 and 1
        """
        output_tokens = set(output.lower().split())
        expected_tokens = set(expected.lower().split())

        if not output_tokens or not expected_tokens:
            return 0.0

        overlap = output_tokens & expected_tokens
        precision = len(overlap) / len(output_tokens)
        recall = len(overlap) / len(expected_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class CustomEvaluator:
    """
    Main evaluation framework for custom LLM evaluation.

    Args:
        model_fn: Function that takes a prompt and returns model output
        name: Name of this evaluation run

    Example:
        >>> def my_model(prompt):
        ...     return "Some response"
        >>> evaluator = CustomEvaluator(model_fn=my_model, name="Test Eval")
        >>> sample = EvalSample(input="Hello", expected="world")
        >>> result = evaluator.evaluate_sample(sample, MetricType.CONTAINS)
    """

    def __init__(self, model_fn: Callable[[str], str], name: str = "default"):
        self.model_fn = model_fn
        self.name = name
        self.results: List[EvalResult] = []
        self.metrics = EvaluationMetrics()

    def evaluate_sample(
        self,
        sample: EvalSample,
        metric_type: MetricType,
        **metric_kwargs
    ) -> EvalResult:
        """
        Evaluate a single sample.

        Args:
            sample: The evaluation sample
            metric_type: Type of metric to use
            **metric_kwargs: Additional arguments for the metric

        Returns:
            EvalResult with evaluation details
        """
        # Generate output
        start_time = time.time()
        try:
            output = self.model_fn(sample.input)
        except Exception as e:
            output = f"ERROR: {str(e)}"
        latency_ms = (time.time() - start_time) * 1000

        # Calculate score based on metric type
        score = 0.0
        details = {"metric_type": metric_type.value}

        if sample.expected is None and metric_type != MetricType.LLM_JUDGE:
            score = 0.0
            details["error"] = "No expected value provided"
        elif metric_type == MetricType.EXACT_MATCH:
            score = self.metrics.exact_match(output, sample.expected, **metric_kwargs)
        elif metric_type == MetricType.CONTAINS:
            score = self.metrics.contains(output, sample.expected, **metric_kwargs)
        elif metric_type == MetricType.REGEX:
            pattern = sample.metadata.get("pattern", sample.expected)
            score = self.metrics.regex_match(output, pattern)
        elif metric_type == MetricType.NUMERIC:
            expected_num = float(sample.expected)
            tolerance = metric_kwargs.get("tolerance", sample.metadata.get("tolerance", 0.01))
            score = self.metrics.numeric_close(output, expected_num, tolerance)
        elif metric_type == MetricType.F1:
            score = self.metrics.f1_score(output, sample.expected)
        elif metric_type == MetricType.CUSTOM:
            custom_fn = metric_kwargs.get("custom_fn")
            if custom_fn:
                score = custom_fn(output, sample.expected)

        result = EvalResult(
            sample=sample,
            output=output,
            score=score,
            passed=score >= metric_kwargs.get("threshold", 0.5),
            details=details,
            latency_ms=latency_ms,
        )

        self.results.append(result)
        return result

    def evaluate_dataset(
        self,
        samples: List[EvalSample],
        metric_type: MetricType,
        verbose: bool = True,
        **metric_kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a full dataset.

        Args:
            samples: List of evaluation samples
            metric_type: Type of metric to use
            verbose: Print progress
            **metric_kwargs: Additional arguments for the metric

        Returns:
            Summary dictionary with evaluation results
        """
        if verbose:
            print(f"\nEvaluating {len(samples)} samples...")

        for i, sample in enumerate(samples):
            self.evaluate_sample(sample, metric_type, **metric_kwargs)
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(samples)}")

        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of evaluation.

        Returns:
            Dictionary with summary metrics
        """
        if not self.results:
            return {"error": "No results to summarize"}

        scores = [r.score for r in self.results]
        latencies = [r.latency_ms for r in self.results]

        # Group by category
        category_scores: Dict[str, List[float]] = {}
        for r in self.results:
            cat = r.sample.category
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(r.score)

        return {
            "total_samples": len(self.results),
            "mean_score": sum(scores) / len(scores),
            "pass_rate": sum(1 for r in self.results if r.passed) / len(self.results),
            "mean_latency_ms": sum(latencies) / len(latencies),
            "min_score": min(scores),
            "max_score": max(scores),
            "category_scores": {
                cat: sum(s) / len(s) for cat, s in category_scores.items()
            },
        }

    def print_summary(self):
        """Print a formatted summary."""
        summary = self.get_summary()

        print(f"\n{'='*50}")
        print(f"Evaluation Summary: {self.name}")
        print(f"{'='*50}")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Mean Score: {summary['mean_score']:.2%}")
        print(f"Pass Rate: {summary['pass_rate']:.2%}")
        print(f"Mean Latency: {summary['mean_latency_ms']:.1f}ms")

        if len(summary['category_scores']) > 1:
            print(f"\nScores by Category:")
            for cat, score in summary['category_scores'].items():
                print(f"  {cat}: {score:.2%}")

    def get_failed_samples(self) -> List[EvalResult]:
        """Get list of failed samples for debugging."""
        return [r for r in self.results if not r.passed]

    def clear_results(self):
        """Clear all results for a new evaluation run."""
        self.results = []


class LLMJudge:
    """
    Use an LLM to evaluate responses.

    Args:
        judge_fn: Function to call the judge LLM
        prompt_template: Custom prompt template (optional)

    Example:
        >>> judge = LLMJudge(judge_fn=my_judge_model)
        >>> evaluation = judge.evaluate(
        ...     question="What is Python?",
        ...     response="Python is a programming language."
        ... )
        >>> print(f"Score: {evaluation['score']}")
    """

    DEFAULT_PROMPT = """You are an expert evaluator. Rate the following response on a scale of 1-10.

Question: {question}

Response to evaluate: {response}

Evaluation criteria:
- Accuracy: Is the information correct?
- Helpfulness: Does it address the question?
- Clarity: Is it easy to understand?
- Completeness: Does it cover the topic adequately?

Provide your evaluation in this exact JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<brief explanation>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
}}

JSON evaluation:"""

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        prompt_template: Optional[str] = None
    ):
        self.judge_fn = judge_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT

    def evaluate(self, question: str, response: str) -> Dict[str, Any]:
        """
        Evaluate a single response.

        Args:
            question: The original question
            response: The response to evaluate

        Returns:
            Dictionary with score, reasoning, and details
        """
        prompt = self.prompt_template.format(
            question=question,
            response=response
        )

        judge_output = self.judge_fn(prompt)

        # Try to parse JSON response
        try:
            json_match = re.search(r'\{[^{}]*\}', judge_output, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return {
                    "success": True,
                    "score": evaluation.get("score", 5) / 10.0,
                    "reasoning": evaluation.get("reasoning", ""),
                    "strengths": evaluation.get("strengths", []),
                    "weaknesses": evaluation.get("weaknesses", []),
                    "raw_output": judge_output,
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: try to extract just a number
        numbers = re.findall(r'\b([1-9]|10)\b', judge_output)
        if numbers:
            return {
                "success": True,
                "score": int(numbers[0]) / 10.0,
                "reasoning": "Score extracted from response",
                "strengths": [],
                "weaknesses": [],
                "raw_output": judge_output,
            }

        return {
            "success": False,
            "score": 0.5,
            "reasoning": "Could not parse judge response",
            "raw_output": judge_output,
        }


class PairwiseJudge:
    """
    Compare two responses and pick a winner.

    Args:
        judge_fn: Function to call the judge LLM

    Example:
        >>> pairwise = PairwiseJudge(judge_fn=my_judge_model)
        >>> winner = pairwise.compare(
        ...     question="How do I make coffee?",
        ...     response_a="Boil water and add coffee.",
        ...     response_b="Use fresh beans, grind them, brew at 200F."
        ... )
        >>> print(f"Winner: Response {winner}")
    """

    COMPARISON_PROMPT = """You are comparing two AI responses to the same question.

Question: {question}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider:
- Accuracy of information
- Helpfulness and completeness
- Clarity and organization

Reply with ONLY one of these options:
- "A" if Response A is better
- "B" if Response B is better
- "TIE" if they are roughly equal

Your choice:"""

    def __init__(self, judge_fn: Callable[[str], str]):
        self.judge_fn = judge_fn
        self.results: List[Dict] = []

    def compare(
        self,
        question: str,
        response_a: str,
        response_b: str
    ) -> str:
        """
        Compare two responses and return the winner.

        Args:
            question: The original question
            response_a: First response
            response_b: Second response

        Returns:
            "A", "B", or "TIE"
        """
        prompt = self.COMPARISON_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b,
        )

        result = self.judge_fn(prompt).strip().upper()

        # Parse result
        if "A" in result and "B" not in result:
            winner = "A"
        elif "B" in result and "A" not in result:
            winner = "B"
        else:
            winner = "TIE"

        self.results.append({
            "question": question,
            "response_a": response_a,
            "response_b": response_b,
            "winner": winner,
        })

        return winner

    def get_win_rates(self) -> Dict[str, float]:
        """Calculate win rates for A, B, and ties."""
        if not self.results:
            return {}

        total = len(self.results)
        a_wins = sum(1 for r in self.results if r["winner"] == "A")
        b_wins = sum(1 for r in self.results if r["winner"] == "B")
        ties = sum(1 for r in self.results if r["winner"] == "TIE")

        return {
            "A_wins": a_wins / total,
            "B_wins": b_wins / total,
            "ties": ties / total,
        }


if __name__ == "__main__":
    # Example usage
    print("Evaluation Framework Example")
    print("=" * 40)

    # Mock model function
    def mock_model(prompt: str) -> str:
        return "The answer is 4"

    # Create evaluator
    evaluator = CustomEvaluator(model_fn=mock_model, name="Demo")

    # Create samples
    samples = [
        EvalSample(input="What is 2+2?", expected="4", category="math"),
        EvalSample(input="What is 3+3?", expected="6", category="math"),
    ]

    # Evaluate
    evaluator.evaluate_dataset(samples, MetricType.CONTAINS)
    evaluator.print_summary()
