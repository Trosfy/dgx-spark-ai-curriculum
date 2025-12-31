"""
Module 3.4: Test-Time Compute & Reasoning - Utility Scripts

This module provides utilities for implementing test-time compute strategies:
- reasoning_utils: Chain-of-Thought, self-consistency, and reasoning helpers
- reward_models: Reward model loading and scoring utilities
- evaluation_utils: Benchmark evaluation and accuracy measurement

Example:
    >>> from scripts.reasoning_utils import chain_of_thought, self_consistency
    >>> from scripts.reward_models import load_reward_model, score_response
    >>> from scripts.evaluation_utils import evaluate_gsm8k, calculate_accuracy
"""

from .reasoning_utils import (
    chain_of_thought,
    few_shot_cot,
    self_consistency,
    extract_answer,
    parse_thinking_tokens,
    TreeOfThought,
)

from .reward_models import (
    load_reward_model,
    score_response,
    best_of_n,
    BestOfNSampler,
)

from .evaluation_utils import (
    load_gsm8k_sample,
    load_test_problems,
    evaluate_accuracy,
    compare_models,
    format_results_table,
    ReasoningEvaluator,
)

__all__ = [
    # Reasoning utilities
    "chain_of_thought",
    "few_shot_cot",
    "self_consistency",
    "extract_answer",
    "parse_thinking_tokens",
    "TreeOfThought",
    # Reward models
    "load_reward_model",
    "score_response",
    "best_of_n",
    "BestOfNSampler",
    # Evaluation
    "load_gsm8k_sample",
    "load_test_problems",
    "evaluate_accuracy",
    "compare_models",
    "format_results_table",
    "ReasoningEvaluator",
]

__version__ = "1.0.0"
