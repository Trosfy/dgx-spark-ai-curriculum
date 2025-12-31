"""
Evaluation utilities for RAG systems.

This module provides metrics and evaluation tools for RAG pipelines.

Example Usage:
    from scripts.evaluation import RAGEvaluator, EvaluationSample

    evaluator = RAGEvaluator()
    result = evaluator.evaluate(sample)
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EvaluationSample:
    """A single sample for RAG evaluation."""
    question: str
    ground_truth: str
    contexts: List[str] = None
    answer: str = None
    expected_source: str = None


@dataclass
class EvaluationResult:
    """Evaluation results for a sample."""
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def average(self) -> float:
        """Calculate average score across all metrics."""
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4


def calculate_retrieval_metrics(
    results: List[Dict],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Calculate retrieval metrics (Recall@K, MRR).

    Args:
        results: List of dicts with 'retrieved' and 'relevant' keys
        k_values: K values for Recall@K

    Returns:
        Dictionary of metrics

    Example:
        >>> results = [
        ...     {"retrieved": ["doc1", "doc2"], "relevant": ["doc1"]},
        ...     {"retrieved": ["doc3", "doc1"], "relevant": ["doc1"]}
        ... ]
        >>> metrics = calculate_retrieval_metrics(results)
        >>> print(f"Recall@5: {metrics['recall@5']:.2f}")
    """
    metrics = {}

    # Recall@K
    for k in k_values:
        recalls = []
        for r in results:
            retrieved_k = set(r["retrieved"][:k])
            relevant = set(r["relevant"])
            if relevant:
                recall = len(retrieved_k & relevant) / len(relevant)
                recalls.append(recall)
        metrics[f"recall@{k}"] = np.mean(recalls) if recalls else 0.0

    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for r in results:
        relevant = set(r["relevant"])
        for i, doc in enumerate(r["retrieved"]):
            if doc in relevant:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)
    metrics["mrr"] = np.mean(reciprocal_ranks)

    return metrics


class RAGEvaluator:
    """
    Evaluator for RAG systems using LLM-as-judge.

    Example:
        >>> evaluator = RAGEvaluator(llm_model="llama3.1:8b")
        >>> result = evaluator.evaluate(sample)
        >>> print(f"Faithfulness: {result.faithfulness}")
    """

    def __init__(self, llm_model: str = "llama3.1:8b"):
        """
        Initialize evaluator.

        Args:
            llm_model: Ollama model name for evaluation
        """
        self.llm_model = llm_model

    def _llm_judge(self, prompt: str) -> str:
        """Get LLM judgment."""
        try:
            import ollama
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return "0.5"

    def _parse_score(self, response: str) -> float:
        """Parse score from LLM response."""
        for val in ["1.0", "0.5", "0.0", "1", "0"]:
            if val in response:
                return float(val)
        return 0.5

    def evaluate_faithfulness(self, sample: EvaluationSample) -> float:
        """Evaluate if answer is grounded in context."""
        if not sample.contexts:
            return 0.0

        context_str = "\n".join(sample.contexts[:3])
        prompt = f"""Determine if the answer is faithfully grounded in the context.

CONTEXT:
{context_str}

ANSWER:
{sample.answer}

Score 1.0 if all claims are supported, 0.5 if some are, 0.0 if not.
Respond with ONLY: 0.0, 0.5, or 1.0"""

        return self._parse_score(self._llm_judge(prompt))

    def evaluate_answer_relevancy(self, sample: EvaluationSample) -> float:
        """Evaluate if answer addresses the question."""
        prompt = f"""Does the answer directly address the question?

QUESTION: {sample.question}
ANSWER: {sample.answer}

Score 1.0 if complete, 0.5 if partial, 0.0 if not.
Respond with ONLY: 0.0, 0.5, or 1.0"""

        return self._parse_score(self._llm_judge(prompt))

    def evaluate_context_precision(self, sample: EvaluationSample) -> float:
        """Evaluate if retrieved contexts are relevant."""
        if not sample.contexts:
            return 0.0

        relevant_count = 0
        for context in sample.contexts[:5]:
            prompt = f"""Is this context relevant to the question?

QUESTION: {sample.question}
CONTEXT: {context[:500]}

Respond with ONLY: YES or NO"""

            response = self._llm_judge(prompt).upper()
            if "YES" in response:
                relevant_count += 1

        return relevant_count / min(5, len(sample.contexts))

    def evaluate_context_recall(self, sample: EvaluationSample) -> float:
        """Evaluate if contexts contain information for ground truth."""
        if not sample.contexts or not sample.ground_truth:
            return 0.0

        context_str = "\n".join(sample.contexts)
        prompt = f"""Does the context contain info for this ground truth?

GROUND TRUTH: {sample.ground_truth}
CONTEXT: {context_str[:2000]}

Score 1.0 if all info present, 0.5 if some, 0.0 if none.
Respond with ONLY: 0.0, 0.5, or 1.0"""

        return self._parse_score(self._llm_judge(prompt))

    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Run all evaluations on a sample."""
        return EvaluationResult(
            question=sample.question,
            faithfulness=self.evaluate_faithfulness(sample),
            answer_relevancy=self.evaluate_answer_relevancy(sample),
            context_precision=self.evaluate_context_precision(sample),
            context_recall=self.evaluate_context_recall(sample)
        )

    def evaluate_batch(
        self,
        samples: List[EvaluationSample],
        progress_callback: Optional[Callable] = None
    ) -> List[EvaluationResult]:
        """Evaluate multiple samples."""
        results = []
        for i, sample in enumerate(samples):
            result = self.evaluate(sample)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(samples), result)
        return results


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, float]:
    """
    Aggregate evaluation results.

    Args:
        results: List of EvaluationResult objects

    Returns:
        Dictionary with aggregated metrics

    Example:
        >>> metrics = aggregate_results(results)
        >>> print(f"Avg faithfulness: {metrics['faithfulness']:.2f}")
    """
    return {
        "faithfulness": np.mean([r.faithfulness for r in results]),
        "answer_relevancy": np.mean([r.answer_relevancy for r in results]),
        "context_precision": np.mean([r.context_precision for r in results]),
        "context_recall": np.mean([r.context_recall for r in results]),
        "average": np.mean([r.average for r in results])
    }
