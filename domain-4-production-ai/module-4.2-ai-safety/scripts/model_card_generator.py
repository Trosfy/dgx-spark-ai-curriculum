"""
Model Card Generator for AI Safety Documentation

This module provides utilities for creating comprehensive model cards
following Hugging Face and industry best practices.

Example usage:
    >>> from model_card_generator import ModelCardGenerator
    >>> generator = ModelCardGenerator()
    >>> generator.set_model_info(name="my-model", version="1.0.0")
    >>> generator.add_safety_results(truthfulqa=0.52, bbq_accuracy=0.73)
    >>> markdown = generator.generate()
    >>> generator.save("model_card.md")
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class SafetyEvaluation:
    """Safety evaluation results for a model."""
    truthfulqa_score: Optional[float] = None
    bbq_accuracy: Optional[float] = None
    bbq_bias_score: Optional[float] = None
    red_team_pass_rate: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BiasAnalysis:
    """Bias analysis results for a model."""
    dimensions_tested: List[str] = field(default_factory=list)
    disparities_found: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mitigations_applied: List[str] = field(default_factory=list)


@dataclass
class TrainingDetails:
    """Training details for a model."""
    base_model: str = ""
    training_method: str = ""
    training_data: str = ""
    hyperparameters: Dict[str, str] = field(default_factory=dict)
    hardware: str = ""
    training_time: str = ""
    compute_cost: str = ""


class ModelCardGenerator:
    """
    Generate comprehensive model cards for AI models.

    This generator creates Hugging Face compatible model cards with
    emphasis on safety documentation, bias analysis, and responsible
    AI practices.

    Example:
        >>> gen = ModelCardGenerator()
        >>> gen.set_model_info(
        ...     name="tech-assistant-llama3-8b",
        ...     version="1.0.0",
        ...     model_type="Text Generation",
        ...     base_model="meta-llama/Llama-3.1-8B-Instruct"
        ... )
        >>> gen.set_description(
        ...     summary="A helpful technical assistant",
        ...     developed_by="AI Team"
        ... )
        >>> gen.add_intended_use("Technical Q&A")
        >>> gen.add_limitation("May hallucinate facts")
        >>> markdown = gen.generate()
    """

    def __init__(self):
        """Initialize the model card generator."""
        # Model Information
        self.model_name: str = ""
        self.model_version: str = ""
        self.model_type: str = ""
        self.base_model: str = ""
        self.license: str = ""

        # Description
        self.summary: str = ""
        self.developed_by: str = ""
        self.language: str = "en"

        # Uses
        self.intended_uses: List[str] = []
        self.out_of_scope_uses: List[str] = []

        # Training
        self.training: TrainingDetails = TrainingDetails()

        # Evaluation
        self.general_benchmarks: Dict[str, float] = {}
        self.safety: SafetyEvaluation = SafetyEvaluation()
        self.bias: BiasAnalysis = BiasAnalysis()

        # Limitations and Risks
        self.limitations: List[str] = []
        self.risks: List[str] = []
        self.known_biases: List[str] = []

        # Recommendations
        self.recommendations: List[str] = []

        # Metadata
        self.tags: List[str] = []
        self.created_date: str = datetime.now().strftime("%Y-%m-%d")

    def set_model_info(
        self,
        name: str,
        version: str = "1.0.0",
        model_type: str = "Text Generation",
        base_model: str = "",
        license: str = "apache-2.0"
    ) -> "ModelCardGenerator":
        """
        Set basic model information.

        Args:
            name: Model name
            version: Model version
            model_type: Type of model (e.g., "Text Generation")
            base_model: Base model used for fine-tuning
            license: Model license

        Returns:
            Self for method chaining
        """
        self.model_name = name
        self.model_version = version
        self.model_type = model_type
        self.base_model = base_model
        self.license = license
        return self

    def set_description(
        self,
        summary: str,
        developed_by: str = "",
        language: str = "en"
    ) -> "ModelCardGenerator":
        """
        Set model description.

        Args:
            summary: Brief summary of the model
            developed_by: Organization/person who developed the model
            language: Primary language

        Returns:
            Self for method chaining
        """
        self.summary = summary
        self.developed_by = developed_by
        self.language = language
        return self

    def add_intended_use(self, use: str) -> "ModelCardGenerator":
        """Add an intended use case."""
        self.intended_uses.append(use)
        return self

    def add_out_of_scope_use(self, use: str) -> "ModelCardGenerator":
        """Add an out-of-scope use case."""
        self.out_of_scope_uses.append(use)
        return self

    def add_limitation(self, limitation: str) -> "ModelCardGenerator":
        """Add a known limitation."""
        self.limitations.append(limitation)
        return self

    def add_risk(self, risk: str) -> "ModelCardGenerator":
        """Add a known risk."""
        self.risks.append(risk)
        return self

    def add_known_bias(self, bias: str) -> "ModelCardGenerator":
        """Add a known bias."""
        self.known_biases.append(bias)
        return self

    def add_recommendation(self, recommendation: str) -> "ModelCardGenerator":
        """Add a usage recommendation."""
        self.recommendations.append(recommendation)
        return self

    def add_tag(self, tag: str) -> "ModelCardGenerator":
        """Add a metadata tag."""
        self.tags.append(tag)
        return self

    def set_training_details(
        self,
        base_model: str = "",
        training_method: str = "",
        training_data: str = "",
        hyperparameters: Optional[Dict[str, str]] = None,
        hardware: str = "",
        training_time: str = "",
        compute_cost: str = ""
    ) -> "ModelCardGenerator":
        """
        Set training details.

        Args:
            base_model: Base model used
            training_method: Training method (e.g., "QLoRA")
            training_data: Description of training data
            hyperparameters: Training hyperparameters
            hardware: Training hardware used
            training_time: Time taken to train
            compute_cost: Estimated compute cost

        Returns:
            Self for method chaining
        """
        self.training = TrainingDetails(
            base_model=base_model or self.base_model,
            training_method=training_method,
            training_data=training_data,
            hyperparameters=hyperparameters or {},
            hardware=hardware,
            training_time=training_time,
            compute_cost=compute_cost
        )
        return self

    def add_benchmark_result(
        self,
        benchmark: str,
        score: float
    ) -> "ModelCardGenerator":
        """Add a general benchmark result."""
        self.general_benchmarks[benchmark] = score
        return self

    def add_safety_results(
        self,
        truthfulqa: Optional[float] = None,
        bbq_accuracy: Optional[float] = None,
        bbq_bias_score: Optional[float] = None,
        red_team_pass_rate: Optional[float] = None,
        **custom_metrics
    ) -> "ModelCardGenerator":
        """
        Add safety evaluation results.

        Args:
            truthfulqa: TruthfulQA MC2 score
            bbq_accuracy: BBQ accuracy score
            bbq_bias_score: BBQ bias score (lower is better)
            red_team_pass_rate: Percentage of red team tests passed
            **custom_metrics: Additional custom metrics

        Returns:
            Self for method chaining
        """
        self.safety = SafetyEvaluation(
            truthfulqa_score=truthfulqa,
            bbq_accuracy=bbq_accuracy,
            bbq_bias_score=bbq_bias_score,
            red_team_pass_rate=red_team_pass_rate,
            custom_metrics=custom_metrics
        )
        return self

    def add_bias_analysis(
        self,
        dimensions: List[str],
        disparities: Optional[Dict[str, Dict[str, float]]] = None,
        mitigations: Optional[List[str]] = None
    ) -> "ModelCardGenerator":
        """
        Add bias analysis results.

        Args:
            dimensions: Demographic dimensions tested
            disparities: Disparities found (by dimension)
            mitigations: Mitigations applied

        Returns:
            Self for method chaining
        """
        self.bias = BiasAnalysis(
            dimensions_tested=dimensions,
            disparities_found=disparities or {},
            mitigations_applied=mitigations or []
        )
        return self

    def generate(self) -> str:
        """
        Generate the complete model card markdown.

        Returns:
            Complete model card as markdown string
        """
        sections = []

        # YAML Frontmatter
        sections.append(self._generate_frontmatter())

        # Title
        sections.append(f"# Model Card: {self.model_name}\n")

        # Model Details
        sections.append(self._generate_model_details())

        # Description
        sections.append(self._generate_description())

        # Intended Uses
        sections.append(self._generate_uses())

        # Training Details
        if self.training.training_method:
            sections.append(self._generate_training())

        # Evaluation
        sections.append(self._generate_evaluation())

        # Safety Evaluation
        sections.append(self._generate_safety())

        # Bias Analysis
        if self.bias.dimensions_tested:
            sections.append(self._generate_bias())

        # Limitations and Risks
        sections.append(self._generate_limitations())

        # Recommendations
        if self.recommendations:
            sections.append(self._generate_recommendations())

        # Usage Examples
        sections.append(self._generate_usage())

        # Citation
        sections.append(self._generate_citation())

        return "\n".join(sections)

    def _generate_frontmatter(self) -> str:
        """Generate YAML frontmatter."""
        tags_str = "\n".join(f"  - {tag}" for tag in self.tags)
        return f"""---
language: {self.language}
license: {self.license}
base_model: {self.base_model}
tags:
{tags_str}
library_name: transformers
pipeline_tag: text-generation
---
"""

    def _generate_model_details(self) -> str:
        """Generate model details section."""
        return f"""## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | {self.model_name} |
| **Version** | {self.model_version} |
| **Type** | {self.model_type} |
| **Base Model** | [{self.base_model}](https://huggingface.co/{self.base_model}) |
| **License** | {self.license} |
| **Developed By** | {self.developed_by} |
| **Release Date** | {self.created_date} |
"""

    def _generate_description(self) -> str:
        """Generate description section."""
        return f"""## Model Description

{self.summary}
"""

    def _generate_uses(self) -> str:
        """Generate uses section."""
        intended = "\n".join(f"- {use}" for use in self.intended_uses)
        out_of_scope = "\n".join(f"- {use}" for use in self.out_of_scope_uses)

        return f"""## Intended Uses

### Primary Use Cases

{intended if intended else "- General text generation"}

### Out-of-Scope Uses

The following uses are **NOT recommended**:

{out_of_scope if out_of_scope else "- Uses not aligned with the model's intended purpose"}
"""

    def _generate_training(self) -> str:
        """Generate training details section."""
        hyperparam_str = "\n".join(
            f"- **{k}**: {v}"
            for k, v in self.training.hyperparameters.items()
        )

        return f"""## Training Details

### Training Data

{self.training.training_data if self.training.training_data else "Not specified"}

### Training Procedure

- **Method**: {self.training.training_method}
- **Hardware**: {self.training.hardware}
- **Training Time**: {self.training.training_time}

### Hyperparameters

{hyperparam_str if hyperparam_str else "Not specified"}
"""

    def _generate_evaluation(self) -> str:
        """Generate evaluation section."""
        if not self.general_benchmarks:
            return ""

        benchmarks = "\n".join(
            f"| {name} | {score:.2%} |"
            for name, score in self.general_benchmarks.items()
        )

        return f"""## Evaluation

### General Benchmarks

| Benchmark | Score |
|-----------|-------|
{benchmarks}
"""

    def _generate_safety(self) -> str:
        """Generate safety evaluation section."""
        metrics = []

        if self.safety.truthfulqa_score is not None:
            metrics.append(
                f"| TruthfulQA MC2 | {self.safety.truthfulqa_score:.2%} | "
                f"Measures truthfulness (higher is better) |"
            )

        if self.safety.bbq_accuracy is not None:
            metrics.append(
                f"| BBQ Accuracy | {self.safety.bbq_accuracy:.2%} | "
                f"Bias benchmark accuracy |"
            )

        if self.safety.bbq_bias_score is not None:
            metrics.append(
                f"| BBQ Bias Score | {self.safety.bbq_bias_score:.2%} | "
                f"Stereotype reliance (lower is better) |"
            )

        if self.safety.red_team_pass_rate is not None:
            metrics.append(
                f"| Red Team Pass Rate | {self.safety.red_team_pass_rate:.2%} | "
                f"Adversarial test success |"
            )

        for name, score in self.safety.custom_metrics.items():
            metrics.append(f"| {name} | {score:.2%} | |")

        metrics_str = "\n".join(metrics) if metrics else "| No safety metrics available | - | - |"

        return f"""## Safety Evaluation

| Metric | Score | Notes |
|--------|-------|-------|
{metrics_str}

### Safety Testing Methodology

This model was evaluated using:
- **TruthfulQA**: Measures tendency to generate truthful vs. false information
- **BBQ (Bias Benchmark for QA)**: Measures social biases across 9 categories
- **Red Team Testing**: Adversarial testing for jailbreaks and prompt injection
"""

    def _generate_bias(self) -> str:
        """Generate bias analysis section."""
        dims = ", ".join(self.bias.dimensions_tested)

        disparities = []
        for dim, metrics in self.bias.disparities_found.items():
            for metric, value in metrics.items():
                flag = " (significant)" if value > 0.1 else ""
                disparities.append(f"| {dim} | {metric} | {value:.3f}{flag} |")

        disparities_str = "\n".join(disparities) if disparities else "| No significant disparities found | - | - |"

        mitigations = "\n".join(
            f"- {m}" for m in self.bias.mitigations_applied
        ) if self.bias.mitigations_applied else "- None applied"

        return f"""## Bias Analysis

### Dimensions Tested

{dims}

### Disparities Found

| Dimension | Metric | Gap |
|-----------|--------|-----|
{disparities_str}

### Mitigations Applied

{mitigations}
"""

    def _generate_limitations(self) -> str:
        """Generate limitations and risks section."""
        limitations = "\n".join(
            f"- {lim}" for lim in self.limitations
        ) if self.limitations else "- No specific limitations documented"

        risks = "\n".join(
            f"- {risk}" for risk in self.risks
        ) if self.risks else "- No specific risks documented"

        biases = "\n".join(
            f"- {bias}" for bias in self.known_biases
        ) if self.known_biases else "- No specific biases documented"

        return f"""## Limitations

{limitations}

## Risks

{risks}

## Known Biases

{biases}
"""

    def _generate_recommendations(self) -> str:
        """Generate recommendations section."""
        recs = "\n".join(f"- {rec}" for rec in self.recommendations)

        return f"""## Recommendations for Safe Use

{recs}
"""

    def _generate_usage(self) -> str:
        """Generate usage examples section."""
        return f"""## How to Use

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "{self.base_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{self.base_model}")

# Generate response
messages = [{{"role": "user", "content": "Hello, how can you help me?"}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Recommended Generation Settings

```python
generation_config = {{
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1
}}
```
"""

    def _generate_citation(self) -> str:
        """Generate citation section."""
        safe_name = self.model_name.replace("-", "_").replace(" ", "_")

        return f"""## Citation

```bibtex
@misc{{{safe_name},
  author = {{{self.developed_by}}},
  title = {{{self.model_name}}},
  year = {{{self.created_date[:4]}}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/your-username/{self.model_name}}}}}
}}
```

## Model Card Authors

{self.developed_by}

---

*This model card was generated using the Model Card Generator from the DGX Spark AI Curriculum.*
"""

    def save(self, filepath: str) -> None:
        """
        Save the model card to a file.

        Args:
            filepath: Path to save the model card
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.generate())
        print(f"Model card saved to: {filepath}")

    def to_dict(self) -> Dict:
        """Convert model card data to dictionary."""
        return {
            "model_info": {
                "name": self.model_name,
                "version": self.model_version,
                "type": self.model_type,
                "base_model": self.base_model,
                "license": self.license,
            },
            "description": {
                "summary": self.summary,
                "developed_by": self.developed_by,
                "language": self.language,
            },
            "uses": {
                "intended": self.intended_uses,
                "out_of_scope": self.out_of_scope_uses,
            },
            "safety": {
                "truthfulqa": self.safety.truthfulqa_score,
                "bbq_accuracy": self.safety.bbq_accuracy,
                "bbq_bias": self.safety.bbq_bias_score,
                "red_team_pass_rate": self.safety.red_team_pass_rate,
            },
            "limitations": self.limitations,
            "risks": self.risks,
            "biases": self.known_biases,
            "recommendations": self.recommendations,
            "created_date": self.created_date,
        }

    def to_json(self, filepath: str) -> None:
        """Save model card data as JSON."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def create_example_model_card() -> ModelCardGenerator:
    """
    Create an example model card demonstrating all features.

    Returns:
        Configured ModelCardGenerator instance
    """
    gen = ModelCardGenerator()

    # Basic info
    gen.set_model_info(
        name="tech-assistant-llama3-8b-lora",
        version="1.0.0",
        model_type="Text Generation (Conversational)",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        license="llama3.1"
    )

    # Description
    gen.set_description(
        summary="""A LoRA fine-tuned version of Llama 3.1 8B optimized for technical assistance.
Trained on DGX Spark using QLoRA with 4-bit quantization.
Designed for programming help, tech explanations, and general knowledge Q&A.""",
        developed_by="DGX Spark AI Curriculum Team"
    )

    # Intended uses
    gen.add_intended_use("Technical support and programming assistance")
    gen.add_intended_use("Answering general knowledge questions")
    gen.add_intended_use("Code review and debugging help")
    gen.add_intended_use("Educational tutoring in STEM subjects")

    # Out of scope uses
    gen.add_out_of_scope_use("Medical diagnosis or health advice")
    gen.add_out_of_scope_use("Legal counsel or advice")
    gen.add_out_of_scope_use("Financial investment recommendations")
    gen.add_out_of_scope_use("Generating malicious code")
    gen.add_out_of_scope_use("Critical safety systems without human oversight")

    # Training
    gen.set_training_details(
        training_method="QLoRA (4-bit quantization + LoRA adapters)",
        training_data="Curated technical Q&A pairs from Stack Overflow and documentation",
        hyperparameters={
            "LoRA Rank": "64",
            "LoRA Alpha": "128",
            "Learning Rate": "2e-4",
            "Epochs": "3",
            "Batch Size": "4 (with gradient accumulation)"
        },
        hardware="NVIDIA DGX Spark (128GB unified memory)",
        training_time="~4 hours"
    )

    # Benchmarks
    gen.add_benchmark_result("MMLU", 0.65)
    gen.add_benchmark_result("HellaSwag", 0.78)
    gen.add_benchmark_result("ARC-Challenge", 0.52)

    # Safety results
    gen.add_safety_results(
        truthfulqa=0.52,
        bbq_accuracy=0.73,
        bbq_bias_score=0.08,
        red_team_pass_rate=0.85
    )

    # Bias analysis
    gen.add_bias_analysis(
        dimensions=["gender", "age", "profession"],
        disparities={
            "gender": {"sentiment_gap": 0.05, "helpfulness_gap": 0.03},
            "age": {"sentiment_gap": 0.08, "helpfulness_gap": 0.06}
        },
        mitigations=["Debiasing system prompt", "Balanced training data"]
    )

    # Limitations
    gen.add_limitation("Knowledge cutoff: Training data up to early 2024")
    gen.add_limitation("May hallucinate facts, especially for recent events")
    gen.add_limitation("Complex multi-step reasoning may be unreliable")
    gen.add_limitation("Code generation should be reviewed before production use")

    # Risks
    gen.add_risk("May generate plausible but incorrect information")
    gen.add_risk("Could be manipulated through adversarial prompts")
    gen.add_risk("Outputs may perpetuate biases from training data")

    # Known biases
    gen.add_known_bias("Slight preference for Western cultural contexts")
    gen.add_known_bias("Technical examples skew toward Python and JavaScript")

    # Recommendations
    gen.add_recommendation("Always verify important information from authoritative sources")
    gen.add_recommendation("Review generated code before execution")
    gen.add_recommendation("Implement guardrails for production deployment")
    gen.add_recommendation("Monitor outputs for bias and quality")
    gen.add_recommendation("Keep a human in the loop for important decisions")

    # Tags
    gen.add_tag("llama")
    gen.add_tag("llama-3.1")
    gen.add_tag("lora")
    gen.add_tag("qlora")
    gen.add_tag("technical-assistant")
    gen.add_tag("safety-evaluated")
    gen.add_tag("dgx-spark")

    return gen


if __name__ == "__main__":
    print("Model Card Generator Demo")
    print("=" * 50)

    # Create example model card
    generator = create_example_model_card()

    # Generate markdown
    markdown = generator.generate()

    # Save to file
    generator.save("example_model_card.md")

    print("\nPreview (first 1000 chars):")
    print("-" * 50)
    print(markdown[:1000])
    print("...")

    print("\nModel card saved to: example_model_card.md")
