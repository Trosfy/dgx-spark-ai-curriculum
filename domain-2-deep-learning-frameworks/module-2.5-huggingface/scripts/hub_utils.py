"""
Hugging Face Hub Utilities

This module provides helper functions for interacting with the Hugging Face Hub,
including model discovery, documentation, and testing.

Example usage:
    from scripts.hub_utils import search_models, document_model, test_model_locally

    # Search for sentiment analysis models
    models = search_models(task="text-classification", limit=10)

    # Document a model
    doc = document_model("distilbert-base-uncased-finetuned-sst-2-english")

    # Test a model locally
    result = test_model_locally(
        "distilbert-base-uncased-finetuned-sst-2-english",
        task="classification",
        test_input="This is great!"
    )
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import torch
import time
import json
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification
)


@dataclass
class ModelDocumentation:
    """Documentation for a Hugging Face model."""
    model_id: str
    author: str
    task: Optional[str]
    downloads: int
    likes: int
    library: Optional[str]
    tags: List[str]
    documented_at: str
    notes: str = ""
    tested_locally: bool = False
    local_test_results: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TestResult:
    """Results from testing a model locally."""
    model_id: str
    task: str
    test_input: str
    success: bool
    output: Optional[str]
    load_time_seconds: float
    inference_time_ms: float
    memory_used_gb: float
    error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Returns:
        Dictionary with allocated and reserved memory in GB.
    """
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9
        }
    return {"allocated_gb": 0, "reserved_gb": 0}


def search_models(
    task: Optional[str] = None,
    search: Optional[str] = None,
    author: Optional[str] = None,
    limit: int = 10,
    sort: str = "downloads",
    direction: int = -1
) -> List[Dict[str, Any]]:
    """
    Search for models on the Hugging Face Hub.

    Args:
        task: Task type filter (e.g., "text-classification")
        search: Search query string
        author: Filter by author/organization
        limit: Maximum number of results
        sort: Sort field (downloads, likes, created_at)
        direction: Sort direction (-1 for descending, 1 for ascending)

    Returns:
        List of model dictionaries with id, downloads, likes, etc.

    Example:
        >>> models = search_models(task="text-generation", limit=5)
        >>> for m in models:
        ...     print(f"{m['id']}: {m['downloads']} downloads")
    """
    api = HfApi()

    kwargs = {
        "sort": sort,
        "direction": direction,
        "limit": limit
    }

    if task:
        kwargs["filter"] = task
    if search:
        kwargs["search"] = search
    if author:
        kwargs["author"] = author

    models = list(api.list_models(**kwargs))

    return [
        {
            "id": m.id,
            "author": m.author,
            "downloads": m.downloads if hasattr(m, 'downloads') else 0,
            "likes": m.likes if hasattr(m, 'likes') else 0,
            "pipeline_tag": m.pipeline_tag if hasattr(m, 'pipeline_tag') else None,
            "tags": m.tags[:10] if hasattr(m, 'tags') and m.tags else []
        }
        for m in models
    ]


def document_model(model_id: str) -> ModelDocumentation:
    """
    Create documentation for a Hugging Face model.

    Args:
        model_id: Model identifier (e.g., "bert-base-uncased")

    Returns:
        ModelDocumentation dataclass with model information.

    Example:
        >>> doc = document_model("distilbert-base-uncased-finetuned-sst-2-english")
        >>> print(f"Task: {doc.task}")
        >>> print(f"Downloads: {doc.downloads:,}")
    """
    api = HfApi()

    try:
        info = api.model_info(model_id)

        return ModelDocumentation(
            model_id=model_id,
            author=info.author or "unknown",
            task=info.pipeline_tag,
            downloads=info.downloads or 0,
            likes=info.likes or 0,
            library=info.library_name,
            tags=info.tags[:10] if info.tags else [],
            documented_at=datetime.now().isoformat()
        )
    except Exception as e:
        return ModelDocumentation(
            model_id=model_id,
            author="error",
            task=None,
            downloads=0,
            likes=0,
            library=None,
            tags=[],
            documented_at=datetime.now().isoformat(),
            notes=f"Error: {str(e)}"
        )


def get_model_readme(model_id: str) -> Optional[str]:
    """
    Download and return a model's README content.

    Args:
        model_id: Model identifier

    Returns:
        README.md content as string, or None if not found.
    """
    try:
        readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def test_model_locally(
    model_id: str,
    task: str,
    test_input: str,
    context: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16
) -> TestResult:
    """
    Test a model locally and return results.

    Args:
        model_id: HuggingFace model identifier
        task: Type of task (classification, generation, qa, ner)
        test_input: Sample input to test with
        context: Context for QA tasks
        device: Device to use (defaults to best available)
        dtype: Data type (default: bfloat16 for DGX Spark)

    Returns:
        TestResult dataclass with test results.

    Example:
        >>> result = test_model_locally(
        ...     "distilbert-base-uncased-finetuned-sst-2-english",
        ...     "classification",
        ...     "This movie was amazing!"
        ... )
        >>> print(f"Output: {result.output}")
        >>> print(f"Time: {result.inference_time_ms:.2f}ms")
    """
    if device is None:
        device = get_device()

    result = TestResult(
        model_id=model_id,
        task=task,
        test_input=test_input,
        success=False,
        output=None,
        load_time_seconds=0,
        inference_time_ms=0,
        memory_used_gb=0,
        error=None
    )

    try:
        # Clear memory and track initial state
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Time model loading
        start_load = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load appropriate model class
        if task == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id, torch_dtype=dtype
            )
        elif task == "generation":
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif task == "qa":
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_id, torch_dtype=dtype
            )
        elif task == "ner":
            model = AutoModelForTokenClassification.from_pretrained(
                model_id, torch_dtype=dtype
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        model = model.to(device).eval()
        result.load_time_seconds = time.time() - start_load
        result.memory_used_gb = (torch.cuda.memory_allocated() / 1e9) - initial_memory if torch.cuda.is_available() else 0

        # Time inference
        start_inference = time.time()

        if task == "qa" and context:
            inputs = tokenizer(test_input, context, return_tensors="pt", truncation=True)
        else:
            inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=512)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if task == "classification":
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item()
                result.output = f"Class {pred} (confidence: {conf:.2%})"

            elif task == "generation":
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                result.output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            elif task == "qa":
                outputs = model(**inputs)
                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits) + 1
                answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
                result.output = answer

            elif task == "ner":
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                entities = []
                for token, pred in zip(tokens, predictions[0]):
                    if pred.item() != 0:  # Non-O tag
                        entities.append(f"{token}:{pred.item()}")
                result.output = ", ".join(entities) if entities else "No entities found"

        result.inference_time_ms = (time.time() - start_inference) * 1000
        result.success = True

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        result.error = str(e)

    return result


def compare_models(
    model_ids: List[str],
    task: str,
    test_inputs: List[str],
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Compare multiple models on the same inputs.

    Args:
        model_ids: List of model identifiers to compare
        task: Task type (classification, generation, qa, ner)
        test_inputs: List of test inputs
        device: Device to use

    Returns:
        List of comparison results for each model.

    Example:
        >>> results = compare_models(
        ...     ["distilbert-base-uncased-finetuned-sst-2-english", "bert-base-uncased-finetuned-sst-2-english"],
        ...     "classification",
        ...     ["Great movie!", "Terrible film."]
        ... )
    """
    results = []

    for model_id in model_ids:
        model_results = {
            "model_id": model_id,
            "tests": [],
            "avg_inference_ms": 0,
            "memory_gb": 0
        }

        total_time = 0
        for test_input in test_inputs:
            result = test_model_locally(model_id, task, test_input, device=device)
            model_results["tests"].append({
                "input": test_input[:50] + "..." if len(test_input) > 50 else test_input,
                "output": result.output,
                "time_ms": result.inference_time_ms,
                "success": result.success
            })
            total_time += result.inference_time_ms
            model_results["memory_gb"] = result.memory_used_gb

        model_results["avg_inference_ms"] = total_time / len(test_inputs)
        results.append(model_results)

    return results


def print_model_comparison(results: List[Dict[str, Any]]) -> None:
    """
    Print a formatted comparison of model results.

    Args:
        results: Output from compare_models()
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    for r in results:
        print(f"\n{r['model_id']}")
        print(f"  Memory: {r['memory_gb']:.2f} GB")
        print(f"  Avg inference: {r['avg_inference_ms']:.2f} ms")
        print("  Results:")
        for t in r["tests"]:
            status = "✓" if t["success"] else "✗"
            print(f"    {status} '{t['input']}' → {t['output']}")


if __name__ == "__main__":
    # Example usage
    print("Hub Utilities Demo")
    print("=" * 50)

    # Search for models
    print("\nSearching for sentiment models...")
    models = search_models(task="text-classification", limit=3)
    for m in models:
        print(f"  {m['id']}: {m['downloads']:,} downloads")

    # Document a model
    print("\nDocumenting a model...")
    doc = document_model("distilbert-base-uncased-finetuned-sst-2-english")
    print(f"  Task: {doc.task}")
    print(f"  Downloads: {doc.downloads:,}")
    print(f"  Likes: {doc.likes}")

    # Test model locally
    if torch.cuda.is_available():
        print("\nTesting model locally...")
        result = test_model_locally(
            "distilbert-base-uncased-finetuned-sst-2-english",
            "classification",
            "This is an amazing product!"
        )
        print(f"  Success: {result.success}")
        print(f"  Output: {result.output}")
        print(f"  Time: {result.inference_time_ms:.2f} ms")
