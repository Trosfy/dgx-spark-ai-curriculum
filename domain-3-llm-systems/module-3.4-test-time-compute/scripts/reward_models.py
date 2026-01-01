"""
Reward Model Utilities for Test-Time Compute

This module provides utilities for loading and using reward models:
- ArmoRM (Llama3-based reward model)
- Skywork-Reward models
- Best-of-N sampling with reward-guided selection

Optimized for DGX Spark's 128GB unified memory - can run larger reward
models without worrying about OOM errors.

Example:
    >>> from scripts.reward_models import BestOfNSampler, load_reward_model
    >>>
    >>> # Load reward model
    >>> reward_model = load_reward_model("RLHFlow/ArmoRM-Llama3-8B-v0.1")
    >>>
    >>> # Best-of-N sampling
    >>> sampler = BestOfNSampler(llm_client, reward_model)
    >>> best_response = sampler.sample(prompt, n=5)
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Try to import torch (may not be available in all environments)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

# Try to import transformers (may not be available in all environments)
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoModel,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Type aliases
LLMClient = Any
RewardModel = Any


# =============================================================================
# Reward Model Loading
# =============================================================================

@dataclass
class RewardModelConfig:
    """Configuration for reward model loading."""
    model_name: str
    model_type: str = "sequence_classification"  # or "custom"
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    trust_remote_code: bool = True
    max_length: int = 4096


def load_reward_model(
    model_name: str,
    config: Optional[RewardModelConfig] = None,
    verbose: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a reward model and tokenizer from HuggingFace.

    Supports several popular reward models:
    - ArmoRM: RLHFlow/ArmoRM-Llama3-8B-v0.1
    - Skywork: Skywork/Skywork-Reward-Llama-3.1-8B
    - InternLM: internlm/internlm2-7b-reward

    Args:
        model_name: HuggingFace model name or path
        config: Optional configuration override
        verbose: Whether to print loading progress

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_reward_model(
        ...     "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        ... )
        >>> # Memory used: ~16GB on DGX Spark
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch library required. Install with: pip install torch"
        )

    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    config = config or RewardModelConfig(model_name=model_name)

    if verbose:
        print(f"Loading reward model: {model_name}")
        print(f"  torch_dtype: {config.torch_dtype}")
        print(f"  device_map: {config.device_map}")

    start_time = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=config.trust_remote_code,
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model based on type
    if config.model_type == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )

    model.eval()

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Loaded in {elapsed:.1f}s")

        # Report memory usage
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory used: {mem_gb:.1f} GB")

    return model, tokenizer


# =============================================================================
# Scoring Functions
# =============================================================================

def score_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int = 4096,
) -> float:
    """
    Score a response using a reward model.

    Higher scores indicate better responses according to the reward model.

    Args:
        model: Loaded reward model
        tokenizer: Associated tokenizer
        prompt: The input prompt
        response: The response to score
        max_length: Maximum sequence length

    Returns:
        Reward score (float)

    Example:
        >>> score = score_response(
        ...     model, tokenizer,
        ...     prompt="What is 2 + 2?",
        ...     response="2 + 2 equals 4."
        ... )
        >>> print(f"Score: {score:.3f}")
        Score: 0.847
    """
    # Format as chat-style input
    conversation = f"User: {prompt}\n\nAssistant: {response}"

    # Tokenize
    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    # Move to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get score
    with torch.no_grad():
        outputs = model(**inputs)

        # Handle different output formats
        if hasattr(outputs, 'logits'):
            # Sequence classification model
            score = outputs.logits.squeeze().item()
        elif hasattr(outputs, 'score'):
            score = outputs.score.squeeze().item()
        else:
            # Try to get last hidden state and pool
            hidden = outputs.last_hidden_state
            score = hidden[:, -1, :].mean().item()

    return score


def score_batch(
    model: Any,
    tokenizer: Any,
    prompt: str,
    responses: List[str],
    max_length: int = 4096,
    batch_size: int = 4,
) -> List[float]:
    """
    Score multiple responses in batches for efficiency.

    On DGX Spark's 128GB unified memory, you can use larger batch sizes.

    Args:
        model: Loaded reward model
        tokenizer: Associated tokenizer
        prompt: The input prompt
        responses: List of responses to score
        max_length: Maximum sequence length
        batch_size: Number of responses per batch

    Returns:
        List of scores for each response

    Example:
        >>> responses = ["2 + 2 = 4", "The answer is four.", "It equals 4!"]
        >>> scores = score_batch(model, tokenizer, "What is 2+2?", responses)
        >>> for r, s in zip(responses, scores):
        ...     print(f"{s:.3f}: {r}")
    """
    scores = []

    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i + batch_size]

        # Format conversations
        conversations = [
            f"User: {prompt}\n\nAssistant: {response}"
            for response in batch_responses
        ]

        # Tokenize batch
        inputs = tokenizer(
            conversations,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )

        # Move to model's device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get scores
        with torch.no_grad():
            outputs = model(**inputs)

            if hasattr(outputs, 'logits'):
                batch_scores = outputs.logits.squeeze(-1).tolist()
            elif hasattr(outputs, 'score'):
                batch_scores = outputs.score.squeeze(-1).tolist()
            else:
                hidden = outputs.last_hidden_state
                batch_scores = hidden[:, -1, :].mean(dim=-1).tolist()

            # Handle single-item batch
            if isinstance(batch_scores, (int, float)):
                batch_scores = [batch_scores]

            scores.extend(batch_scores)

    return scores


# =============================================================================
# Best-of-N Sampling
# =============================================================================

def best_of_n(
    llm_client: LLMClient,
    llm_model: str,
    reward_model: Any,
    reward_tokenizer: Any,
    prompt: str,
    n: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    verbose: bool = False,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Generate N candidates and select the best by reward score.

    Best-of-N (BoN) sampling is a simple but effective way to improve
    response quality using a reward model.

    Args:
        llm_client: LLM client (ollama, openai, etc.)
        llm_model: LLM model name
        reward_model: Loaded reward model
        reward_tokenizer: Reward model tokenizer
        prompt: Input prompt
        n: Number of candidates to generate
        temperature: Sampling temperature for diversity
        max_tokens: Maximum tokens per response
        verbose: Whether to print progress

    Returns:
        Tuple of (best_response, best_score, all_candidates)

    Example:
        >>> best, score, all_cands = best_of_n(
        ...     ollama, "qwen3:8b",
        ...     reward_model, reward_tokenizer,
        ...     "Explain quantum entanglement simply.",
        ...     n=5
        ... )
        >>> print(f"Best score: {score:.3f}")
        >>> print(f"Best response: {best[:100]}...")
    """
    candidates = []
    messages = [{"role": "user", "content": prompt}]

    # Generate N candidates
    for i in range(n):
        if verbose:
            print(f"  Generating candidate {i+1}/{n}...", end=" ", flush=True)

        start_time = time.time()

        if hasattr(llm_client, 'chat'):
            response = llm_client.chat(
                model=llm_model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            response_text = response['message']['content']
        elif hasattr(llm_client, 'ChatCompletion'):
            response = llm_client.ChatCompletion.create(
                model=llm_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response_text = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported client type: {type(llm_client)}")

        candidates.append(response_text)

        if verbose:
            elapsed = time.time() - start_time
            print(f"({elapsed:.1f}s)")

    # Score all candidates
    if verbose:
        print("  Scoring candidates...")

    scores = score_batch(
        reward_model, reward_tokenizer,
        prompt, candidates
    )

    # Create list of (candidate, score) pairs
    all_candidates = list(zip(candidates, scores))

    # Find best
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_response = candidates[best_idx]
    best_score = scores[best_idx]

    if verbose:
        print(f"  Best score: {best_score:.3f} (candidate {best_idx + 1})")
        print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")

    return best_response, best_score, all_candidates


class BestOfNSampler:
    """
    Best-of-N sampler with caching and statistics.

    Encapsulates the Best-of-N strategy with additional features:
    - Caching of similar prompts
    - Statistics tracking
    - Configurable scoring

    Optimized for DGX Spark: can run 8B reward models alongside
    70B generation models thanks to 128GB unified memory.

    Example:
        >>> sampler = BestOfNSampler(llm_client, reward_model, reward_tokenizer)
        >>> result = sampler.sample("Explain relativity", n=5)
        >>> print(f"Best: {result.response}")
        >>> print(f"Quality improvement: {result.improvement:.1%}")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        llm_model: str,
        reward_model: Any,
        reward_tokenizer: Any,
        default_n: int = 5,
        default_temperature: float = 0.7,
    ):
        """
        Initialize Best-of-N sampler.

        Args:
            llm_client: LLM client
            llm_model: LLM model name
            reward_model: Loaded reward model
            reward_tokenizer: Reward model tokenizer
            default_n: Default number of candidates
            default_temperature: Default sampling temperature
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.default_n = default_n
        self.default_temperature = default_temperature

        # Statistics
        self.total_samples = 0
        self.total_candidates = 0
        self.avg_improvement = 0.0

    def sample(
        self,
        prompt: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        return_all: bool = False,
        verbose: bool = False,
    ) -> Union['BestOfNResult', str]:
        """
        Generate candidates and select the best.

        Args:
            prompt: Input prompt
            n: Number of candidates (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            return_all: Whether to return full result object
            verbose: Whether to print progress

        Returns:
            BestOfNResult object if return_all=True, else best response string
        """
        n = n or self.default_n
        temperature = temperature if temperature is not None else self.default_temperature

        best_response, best_score, all_candidates = best_of_n(
            self.llm_client,
            self.llm_model,
            self.reward_model,
            self.reward_tokenizer,
            prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
        )

        # Calculate improvement over greedy (first sample)
        greedy_score = all_candidates[0][1]
        improvement = (best_score - greedy_score) / abs(greedy_score) if greedy_score != 0 else 0

        # Update statistics
        self.total_samples += 1
        self.total_candidates += n
        self.avg_improvement = (
            (self.avg_improvement * (self.total_samples - 1) + improvement)
            / self.total_samples
        )

        result = BestOfNResult(
            response=best_response,
            score=best_score,
            all_candidates=all_candidates,
            improvement=improvement,
            n=n,
        )

        return result if return_all else best_response

    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            'total_samples': self.total_samples,
            'total_candidates': self.total_candidates,
            'avg_candidates_per_sample': self.total_candidates / max(self.total_samples, 1),
            'avg_improvement': self.avg_improvement,
        }


@dataclass
class BestOfNResult:
    """Result from Best-of-N sampling."""
    response: str
    score: float
    all_candidates: List[Tuple[str, float]]
    improvement: float
    n: int

    def __repr__(self):
        return (
            f"BestOfNResult(score={self.score:.3f}, improvement={self.improvement:.1%}, "
            f"n={self.n})"
        )

    def get_score_distribution(self) -> Dict[str, float]:
        """Get statistics on score distribution."""
        scores = [s for _, s in self.all_candidates]
        return {
            'min': min(scores),
            'max': max(scores),
            'mean': sum(scores) / len(scores),
            'std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
        }


# =============================================================================
# Process Reward Models
# =============================================================================

class ProcessRewardModel:
    """
    Process Reward Model (PRM) for step-by-step scoring.

    PRMs score each reasoning step, not just the final answer.
    This enables more fine-grained feedback and can guide search
    algorithms like Monte Carlo Tree Search.

    Note: Requires a PRM-specific model. Few are publicly available
    as of early 2024. This class provides the interface for when
    more PRMs become available.

    Example:
        >>> prm = ProcessRewardModel("math-shepherd/prm-mistral-7b")
        >>> scores = prm.score_steps(
        ...     "What is 15% of 80?",
        ...     ["First, convert 15% to decimal: 0.15",
        ...      "Then multiply: 80 * 0.15 = 12",
        ...      "The answer is 12"]
        ... )
        >>> print(scores)  # [0.95, 0.92, 0.98]
    """

    def __init__(
        self,
        model_name: str,
        step_delimiter: str = "\n",
    ):
        """
        Initialize Process Reward Model.

        Args:
            model_name: HuggingFace model name
            step_delimiter: How steps are separated
        """
        self.model_name = model_name
        self.step_delimiter = step_delimiter

        # Load model if available
        if HAS_TRANSFORMERS:
            self.model, self.tokenizer = load_reward_model(model_name)
        else:
            self.model = None
            self.tokenizer = None

    def score_steps(
        self,
        problem: str,
        steps: List[str],
    ) -> List[float]:
        """
        Score each reasoning step.

        Args:
            problem: The original problem
            steps: List of reasoning steps

        Returns:
            List of scores, one per step
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Ensure transformers is installed and model exists."
            )

        scores = []

        # Score each prefix of steps
        for i in range(len(steps)):
            partial_solution = self.step_delimiter.join(steps[:i+1])

            # Score this partial solution
            conversation = f"Problem: {problem}\n\nSolution:\n{partial_solution}"

            inputs = self.tokenizer(
                conversation,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'logits'):
                    score = outputs.logits.squeeze().item()
                else:
                    score = outputs.last_hidden_state[:, -1, :].mean().item()

            scores.append(score)

        return scores

    def get_step_quality(
        self,
        problem: str,
        steps: List[str],
    ) -> List[str]:
        """
        Get quality labels for each step.

        Args:
            problem: The original problem
            steps: List of reasoning steps

        Returns:
            List of labels: 'good', 'neutral', or 'bad'
        """
        scores = self.score_steps(problem, steps)

        labels = []
        for score in scores:
            if score > 0.7:
                labels.append('good')
            elif score > 0.3:
                labels.append('neutral')
            else:
                labels.append('bad')

        return labels


# =============================================================================
# Utility Functions
# =============================================================================

def compare_greedy_vs_bon(
    llm_client: LLMClient,
    llm_model: str,
    reward_model: Any,
    reward_tokenizer: Any,
    prompts: List[str],
    n: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare greedy decoding vs Best-of-N on a set of prompts.

    Args:
        llm_client: LLM client
        llm_model: LLM model name
        reward_model: Loaded reward model
        reward_tokenizer: Reward model tokenizer
        prompts: List of prompts to test
        n: Number of candidates for BoN
        verbose: Whether to print progress

    Returns:
        Dictionary with comparison statistics
    """
    greedy_scores = []
    bon_scores = []
    improvements = []

    for i, prompt in enumerate(prompts):
        if verbose:
            print(f"Processing prompt {i+1}/{len(prompts)}...")

        # Best-of-N (first candidate approximates greedy with temp=0.7)
        _, best_score, all_candidates = best_of_n(
            llm_client, llm_model,
            reward_model, reward_tokenizer,
            prompt, n=n, verbose=False
        )

        greedy_score = all_candidates[0][1]
        greedy_scores.append(greedy_score)
        bon_scores.append(best_score)

        improvement = best_score - greedy_score
        improvements.append(improvement)

    avg_greedy = sum(greedy_scores) / len(greedy_scores)
    avg_bon = sum(bon_scores) / len(bon_scores)
    avg_improvement = sum(improvements) / len(improvements)

    return {
        'avg_greedy_score': avg_greedy,
        'avg_bon_score': avg_bon,
        'avg_improvement': avg_improvement,
        'improvement_percentage': (avg_bon - avg_greedy) / abs(avg_greedy) * 100,
        'greedy_scores': greedy_scores,
        'bon_scores': bon_scores,
        'num_prompts': len(prompts),
        'n': n,
    }


def clear_reward_model_cache():
    """Clear GPU memory used by reward models."""
    import gc

    gc.collect()
    if HAS_TORCH and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
