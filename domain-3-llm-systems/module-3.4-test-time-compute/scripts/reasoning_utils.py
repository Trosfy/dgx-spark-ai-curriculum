"""
Reasoning Utilities for Test-Time Compute

This module provides utilities for implementing various reasoning strategies:
- Chain-of-Thought (CoT) prompting
- Self-consistency with majority voting
- Tree-of-Thought exploration
- Answer extraction and parsing

Optimized for DGX Spark's 128GB unified memory.

Example:
    >>> from scripts.reasoning_utils import chain_of_thought, self_consistency
    >>>
    >>> # Zero-shot Chain-of-Thought
    >>> response = chain_of_thought(client, model, "What is 17 * 23?")
    >>> print(response)

    >>> # Self-consistency with majority voting
    >>> answer, confidence, all_answers = self_consistency(
    ...     client, model, "If I have 5 apples and eat 2, how many left?",
    ...     n_samples=5, temperature=0.7
    ... )
    >>> print(f"Answer: {answer} (confidence: {confidence:.0%})")
"""

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Type alias for LLM client (works with ollama, openai, etc.)
LLMClient = Any


# =============================================================================
# Chain-of-Thought Prompting
# =============================================================================

def chain_of_thought(
    client: LLMClient,
    model: str,
    question: str,
    system_prompt: Optional[str] = None,
    cot_trigger: str = "Let's think step by step:",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """
    Apply zero-shot Chain-of-Thought prompting to a question.

    Chain-of-Thought prompting encourages the model to show its reasoning
    process before arriving at an answer, leading to better accuracy on
    complex reasoning tasks.

    Args:
        client: LLM client (ollama, openai, etc.)
        model: Model name/identifier
        question: The question to answer
        system_prompt: Optional system prompt for context
        cot_trigger: The phrase that triggers step-by-step reasoning
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0 for deterministic)

    Returns:
        The model's response including reasoning steps

    Example:
        >>> import ollama
        >>> response = chain_of_thought(
        ...     ollama, "llama3.1:8b",
        ...     "A store sells apples for $2 each. If I buy 5 apples, how much do I spend?"
        ... )
        >>> print(response)
        Let's think step by step:
        1. Each apple costs $2
        2. I want to buy 5 apples
        3. Total cost = 5 x $2 = $10

        The answer is $10.
    """
    # Build the prompt with CoT trigger
    full_prompt = f"{question}\n\n{cot_trigger}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": full_prompt})

    # Handle different client types
    if hasattr(client, 'chat'):
        # Ollama-style client
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response['message']['content']
    elif hasattr(client, 'ChatCompletion'):
        # OpenAI-style client
        response = client.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")


def few_shot_cot(
    client: LLMClient,
    model: str,
    question: str,
    examples: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """
    Apply few-shot Chain-of-Thought prompting with examples.

    Few-shot CoT provides worked examples before the question, showing
    the model the expected reasoning format.

    Args:
        client: LLM client
        model: Model name/identifier
        question: The question to answer
        examples: List of dicts with 'question' and 'answer' keys
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        The model's response including reasoning steps

    Example:
        >>> examples = [
        ...     {
        ...         "question": "If there are 3 cars and 2 more arrive, how many cars?",
        ...         "answer": "Let's think step by step.\\n1. Start with 3 cars\\n2. 2 more arrive\\n3. Total = 3 + 2 = 5 cars\\nThe answer is 5."
        ...     },
        ...     {
        ...         "question": "If I have 5 apples and eat 2, how many left?",
        ...         "answer": "Let's think step by step.\\n1. Start with 5 apples\\n2. Eat 2 apples\\n3. Remaining = 5 - 2 = 3 apples\\nThe answer is 3."
        ...     }
        ... ]
        >>> response = few_shot_cot(client, model, "What is 7 + 8?", examples)
    """
    # Build few-shot prompt
    prompt_parts = []

    for ex in examples:
        prompt_parts.append(f"Q: {ex['question']}")
        prompt_parts.append(f"A: {ex['answer']}")
        prompt_parts.append("")  # Empty line between examples

    prompt_parts.append(f"Q: {question}")
    prompt_parts.append("A: Let's think step by step.")

    full_prompt = "\n".join(prompt_parts)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": full_prompt})

    # Handle different client types
    if hasattr(client, 'chat'):
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response['message']['content']
    elif hasattr(client, 'ChatCompletion'):
        response = client.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported client type: {type(client)}")


# =============================================================================
# Self-Consistency
# =============================================================================

def extract_answer(
    response: str,
    patterns: Optional[List[str]] = None,
    extract_number: bool = True,
) -> Optional[str]:
    """
    Extract the final answer from a model response.

    Uses multiple strategies to find the answer:
    1. Look for explicit patterns like "The answer is X"
    2. Look for patterns like "= X" or "equals X"
    3. Extract the last number if extract_number=True

    Args:
        response: The model's full response
        patterns: Custom regex patterns to try
        extract_number: Whether to fall back to extracting last number

    Returns:
        The extracted answer as a string, or None if not found

    Example:
        >>> response = "Let's solve this. 5 + 3 = 8. The answer is 8."
        >>> extract_answer(response)
        '8'
    """
    # Default patterns to look for
    default_patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n.]+)",
        r"[Aa]nswer[:\s]+([^\n.]+)",
        r"=\s*([^\n,]+?)(?:\s*$|\s*\.|\s*,)",
        r"[Tt]herefore[,\s]+([^\n.]+)",
        r"[Ss]o\s+(?:the\s+answer\s+is\s+)?([^\n.]+?)(?:\s*$|\s*\.)",
    ]

    all_patterns = (patterns or []) + default_patterns

    for pattern in all_patterns:
        matches = re.findall(pattern, response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common artifacts
            answer = answer.strip('.')
            answer = answer.strip()
            if answer:
                return answer

    # Fallback: extract last number
    if extract_number:
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

    return None


def self_consistency(
    client: LLMClient,
    model: str,
    question: str,
    n_samples: int = 5,
    temperature: float = 0.7,
    cot_trigger: str = "Let's think step by step:",
    max_tokens: int = 2048,
    extract_fn: Optional[Callable[[str], Optional[str]]] = None,
    verbose: bool = False,
) -> Tuple[Optional[str], float, List[str]]:
    """
    Apply self-consistency: generate multiple reasoning paths and vote.

    Self-consistency generates diverse reasoning paths by sampling with
    temperature > 0, then takes a majority vote on the final answer.
    This often outperforms single-sample greedy decoding.

    Args:
        client: LLM client
        model: Model name/identifier
        question: The question to answer
        n_samples: Number of reasoning paths to generate
        temperature: Sampling temperature (>0 for diversity)
        cot_trigger: The phrase that triggers step-by-step reasoning
        max_tokens: Maximum tokens per response
        extract_fn: Custom function to extract answer from response
        verbose: Whether to print progress

    Returns:
        Tuple of (best_answer, confidence, all_answers)
        - best_answer: The majority vote answer
        - confidence: Fraction of samples agreeing with majority
        - all_answers: List of all extracted answers

    Example:
        >>> answer, conf, all_ans = self_consistency(
        ...     client, model, "What is 15% of 80?",
        ...     n_samples=5, temperature=0.7
        ... )
        >>> print(f"Answer: {answer} (confidence: {conf:.0%})")
        Answer: 12 (confidence: 80%)
    """
    extract_fn = extract_fn or extract_answer
    answers = []
    responses = []

    full_prompt = f"{question}\n\n{cot_trigger}"

    for i in range(n_samples):
        if verbose:
            print(f"  Generating sample {i+1}/{n_samples}...", end=" ", flush=True)

        start_time = time.time()

        messages = [{"role": "user", "content": full_prompt}]

        if hasattr(client, 'chat'):
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            response_text = response['message']['content']
        elif hasattr(client, 'ChatCompletion'):
            response = client.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response_text = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported client type: {type(client)}")

        responses.append(response_text)
        answer = extract_fn(response_text)
        answers.append(answer)

        if verbose:
            elapsed = time.time() - start_time
            print(f"Answer: {answer} ({elapsed:.1f}s)")

    # Filter out None values for voting
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None, 0.0, answers

    # Majority vote
    counter = Counter(valid_answers)
    best_answer, count = counter.most_common(1)[0]
    confidence = count / len(valid_answers)

    return best_answer, confidence, answers


# =============================================================================
# DeepSeek-R1 Thinking Token Parsing
# =============================================================================

@dataclass
class ThinkingResult:
    """Result from parsing R1-style thinking tokens."""
    thinking: str  # Content within <think>...</think>
    answer: str    # Final answer after thinking
    thinking_tokens: int  # Approximate token count for thinking
    answer_tokens: int    # Approximate token count for answer


def parse_thinking_tokens(response: str) -> ThinkingResult:
    """
    Parse DeepSeek-R1 style thinking tokens from a response.

    R1-style models wrap their reasoning in <think>...</think> tags,
    followed by the final answer.

    Args:
        response: The full model response

    Returns:
        ThinkingResult with separated thinking and answer

    Example:
        >>> response = '''<think>
        ... Let me solve this step by step.
        ... 17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391
        ... </think>
        ...
        ... The answer is 391.'''
        >>> result = parse_thinking_tokens(response)
        >>> print(f"Thinking: {result.thinking[:50]}...")
        >>> print(f"Answer: {result.answer}")
    """
    # Pattern for <think>...</think> blocks
    think_pattern = r'<think>(.*?)</think>'

    # Find all thinking blocks
    thinking_matches = re.findall(think_pattern, response, re.DOTALL)
    thinking_content = "\n".join(thinking_matches)

    # Remove thinking blocks from response to get the answer
    answer_content = re.sub(think_pattern, '', response, flags=re.DOTALL)
    answer_content = answer_content.strip()

    # Rough token count (1 token ~ 4 characters on average)
    thinking_tokens = len(thinking_content) // 4
    answer_tokens = len(answer_content) // 4

    return ThinkingResult(
        thinking=thinking_content.strip(),
        answer=answer_content,
        thinking_tokens=thinking_tokens,
        answer_tokens=answer_tokens,
    )


# =============================================================================
# Tree-of-Thought
# =============================================================================

@dataclass
class ThoughtNode:
    """A node in the Tree-of-Thought."""
    thought: str
    value: float = 0.0
    children: List['ThoughtNode'] = None
    parent: Optional['ThoughtNode'] = None
    depth: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TreeOfThought:
    """
    Tree-of-Thought implementation for systematic reasoning exploration.

    Tree-of-Thought (ToT) explores multiple reasoning branches,
    evaluating each intermediate step and backtracking when needed.

    This is particularly useful for:
    - Math problems with multiple solution paths
    - Puzzles requiring exploration
    - Planning tasks

    Example:
        >>> tot = TreeOfThought(client, model)
        >>> solution = tot.solve(
        ...     "Find a sequence of moves to solve this puzzle...",
        ...     n_branches=3,
        ...     max_depth=4
        ... )
        >>> print(solution['best_path'])
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        evaluator_model: Optional[str] = None,
    ):
        """
        Initialize Tree-of-Thought solver.

        Args:
            client: LLM client
            model: Model for generating thoughts
            evaluator_model: Model for evaluating thoughts (defaults to same model)
        """
        self.client = client
        self.model = model
        self.evaluator_model = evaluator_model or model

    def generate_thoughts(
        self,
        problem: str,
        current_path: List[str],
        n_thoughts: int = 3,
    ) -> List[str]:
        """Generate possible next thoughts from current state."""
        path_str = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(current_path)])

        prompt = f"""Problem: {problem}

Current reasoning path:
{path_str if path_str else "(Starting fresh)"}

Generate {n_thoughts} different possible next steps to explore.
Each step should be distinct and potentially lead to the solution.
Format: List each step on a new line, numbered 1, 2, 3, etc."""

        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.client, 'chat'):
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.8, "num_predict": 512}
            )
            content = response['message']['content']
        else:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.8
            )
            content = response.choices[0].message.content

        # Parse numbered thoughts
        thoughts = []
        lines = content.strip().split('\n')
        for line in lines:
            # Remove numbering like "1.", "1)", "Step 1:", etc.
            cleaned = re.sub(r'^[\d]+[.)\]:]?\s*', '', line).strip()
            if cleaned and len(cleaned) > 10:  # Filter out too-short lines
                thoughts.append(cleaned)

        return thoughts[:n_thoughts]

    def evaluate_thought(
        self,
        problem: str,
        path: List[str],
    ) -> float:
        """Evaluate how promising a reasoning path is (0-1 scale)."""
        path_str = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(path)])

        prompt = f"""Problem: {problem}

Reasoning path so far:
{path_str}

Rate this reasoning path on a scale of 0-10:
- 0: Completely wrong or stuck
- 5: Making progress but unclear
- 10: Excellent, likely leading to solution

Just respond with a single number."""

        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.client, 'chat'):
            response = self.client.chat(
                model=self.evaluator_model,
                messages=messages,
                options={"temperature": 0.0, "num_predict": 16}
            )
            content = response['message']['content']
        else:
            response = self.client.ChatCompletion.create(
                model=self.evaluator_model,
                messages=messages,
                max_tokens=16,
                temperature=0.0
            )
            content = response.choices[0].message.content

        # Extract number
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        if numbers:
            score = float(numbers[0])
            return min(max(score / 10.0, 0.0), 1.0)
        return 0.5  # Default middle score

    def solve(
        self,
        problem: str,
        n_branches: int = 3,
        max_depth: int = 5,
        beam_width: int = 2,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve a problem using Tree-of-Thought search.

        Uses beam search to explore the most promising paths.

        Args:
            problem: The problem to solve
            n_branches: Number of thoughts to generate at each step
            max_depth: Maximum reasoning depth
            beam_width: Number of paths to keep at each level
            verbose: Whether to print progress

        Returns:
            Dictionary with 'best_path', 'score', and 'all_paths'
        """
        # Initialize with empty paths
        paths = [[]]
        path_scores = [1.0]

        for depth in range(max_depth):
            if verbose:
                print(f"Depth {depth + 1}/{max_depth}: Exploring {len(paths)} paths")

            new_paths = []
            new_scores = []

            for path, score in zip(paths, path_scores):
                # Generate new thoughts
                thoughts = self.generate_thoughts(problem, path, n_branches)

                for thought in thoughts:
                    new_path = path + [thought]

                    # Evaluate the new path
                    new_score = self.evaluate_thought(problem, new_path)

                    new_paths.append(new_path)
                    new_scores.append(new_score)

                    if verbose:
                        print(f"  Path score: {new_score:.2f} - {thought[:50]}...")

            if not new_paths:
                break

            # Beam search: keep top paths
            sorted_indices = sorted(
                range(len(new_scores)),
                key=lambda i: new_scores[i],
                reverse=True
            )[:beam_width]

            paths = [new_paths[i] for i in sorted_indices]
            path_scores = [new_scores[i] for i in sorted_indices]

            # Early stopping if we found a very good path
            if path_scores[0] > 0.9:
                if verbose:
                    print("  Found high-confidence path, stopping early")
                break

        # Return best path
        best_idx = path_scores.index(max(path_scores))

        return {
            'best_path': paths[best_idx],
            'score': path_scores[best_idx],
            'all_paths': list(zip(paths, path_scores)),
        }


# =============================================================================
# Utility Functions
# =============================================================================

def format_cot_response(response: str, show_steps: bool = True) -> str:
    """
    Format a Chain-of-Thought response for display.

    Args:
        response: Raw model response
        show_steps: Whether to show numbered steps

    Returns:
        Formatted response string
    """
    lines = response.strip().split('\n')

    if not show_steps:
        # Just return the final answer
        for line in reversed(lines):
            if 'answer' in line.lower():
                return line.strip()
        return lines[-1].strip() if lines else ""

    # Format with clear step markers
    formatted_lines = []
    step_num = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if already numbered
        if re.match(r'^\d+[.)]', line):
            formatted_lines.append(line)
        elif 'step' in line.lower() and ':' in line:
            formatted_lines.append(line)
        else:
            # Add step number if it looks like a reasoning step
            if any(word in line.lower() for word in ['first', 'then', 'next', 'finally', 'therefore', 'so']):
                formatted_lines.append(f"{step_num}. {line}")
                step_num += 1
            else:
                formatted_lines.append(line)

    return '\n'.join(formatted_lines)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses a rough heuristic of ~4 characters per token on average.
    For accurate counts, use the model's tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 chars per token on average
    return len(text) // 4


def calculate_thinking_overhead(
    thinking_tokens: int,
    answer_tokens: int,
) -> Dict[str, float]:
    """
    Calculate the token overhead from thinking.

    Args:
        thinking_tokens: Tokens used for thinking/reasoning
        answer_tokens: Tokens used for final answer

    Returns:
        Dict with overhead statistics
    """
    total = thinking_tokens + answer_tokens
    overhead_ratio = thinking_tokens / max(answer_tokens, 1)
    thinking_pct = thinking_tokens / max(total, 1) * 100

    return {
        'thinking_tokens': thinking_tokens,
        'answer_tokens': answer_tokens,
        'total_tokens': total,
        'overhead_ratio': overhead_ratio,
        'thinking_percentage': thinking_pct,
    }
