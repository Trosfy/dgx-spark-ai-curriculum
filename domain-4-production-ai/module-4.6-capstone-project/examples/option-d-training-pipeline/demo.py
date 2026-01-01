#!/usr/bin/env python3
"""
Training Pipeline Demo

Quick demonstration of the complete training pipeline.
"""

from data_pipeline import DataCollector, DataCurator, DataFormatter, DataSample
from training_loop import TrainingConfig, QLoRATrainer
from model_registry import ModelRegistry
import json


def demo_data_pipeline():
    """Demonstrate data pipeline."""
    print("=" * 60)
    print("PART 1: DATA PIPELINE")
    print("=" * 60)

    # Collect
    print("\n1.1 Data Collection")
    print("-" * 40)
    collector = DataCollector()

    # Add mock samples directly
    samples = [
        DataSample("Explain neural networks", "", "Neural networks are...", "manual"),
        DataSample("Write a sorting algorithm", "Python", "def sort(arr): ...", "manual"),
        DataSample("What is backpropagation?", "", "Backpropagation is...", "manual"),
        DataSample("", "", "", "invalid"),  # Will be filtered
        DataSample("Explain", "", "I cannot help with that", "invalid"),  # Will be filtered
    ] * 5  # Replicate for demo

    print(f"Collected {len(samples)} samples")

    # Curate
    print("\n1.2 Data Curation")
    print("-" * 40)
    curator = DataCurator()
    curator.add_filter("length", curator.length_filter(5, 1000))
    curator.add_filter("quality", curator.quality_filter())

    curated = curator.curate(samples)
    print(f"Curated to {len(curated)} samples")
    print(f"Stats: {curator.stats}")

    # Format
    print("\n1.3 Data Formatting")
    print("-" * 40)
    formatter = DataFormatter()

    alpaca_data = formatter.to_alpaca(curated)
    chat_data = formatter.to_chat(curated)

    print(f"Alpaca format samples: {len(alpaca_data)}")
    print(f"Chat format samples: {len(chat_data)}")

    print("\nSample (Chat format):")
    print(json.dumps(chat_data[0], indent=2))

    return chat_data


def demo_training(train_data):
    """Demonstrate training loop."""
    print("\n" + "=" * 60)
    print("PART 2: TRAINING LOOP")
    print("=" * 60)

    # Config
    print("\n2.1 Training Configuration")
    print("-" * 40)
    config = TrainingConfig(
        model_name="Qwen/Qwen3-8B-Instruct",
        lora_r=16,
        lora_alpha=32,
        num_epochs=2,
        batch_size=4,
        learning_rate=2e-4,
    )

    print(f"Model: {config.model_name}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Training: {config.num_epochs} epochs, bs={config.batch_size}")

    # Train (demo mode)
    print("\n2.2 SFT Training")
    print("-" * 40)
    trainer = QLoRATrainer(config, load_model=False)
    result = trainer.train_sft(train_data)

    print(f"\nResult: {result['status']}")

    # Metrics
    print("\n2.3 Training Metrics")
    print("-" * 40)
    summary = trainer.get_metrics_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return summary


def demo_registry(metrics):
    """Demonstrate model registry."""
    print("\n" + "=" * 60)
    print("PART 3: MODEL REGISTRY")
    print("=" * 60)

    # Create registry
    print("\n3.1 Register Model")
    print("-" * 40)
    registry = ModelRegistry("./demo_registry")

    model = registry.register(
        name="demo-assistant",
        path="./outputs/demo-checkpoint",
        base_model="Llama-3.3-8B-Instruct",
        training_method="SFT",
        metrics={"final_loss": metrics.get("final_loss", 0.5)},
        tags=["demo", "sft"]
    )

    print(f"Registered: {model.full_name}")

    # List
    print("\n3.2 Registry Contents")
    print("-" * 40)
    for name in registry.list_models():
        versions = registry.list_versions(name)
        print(f"  {name}: {len(versions)} version(s)")
        for v in versions:
            print(f"    - {v.version}: loss={v.metrics.get('final_loss', 'N/A')}")

    # Get latest
    print("\n3.3 Get Latest")
    print("-" * 40)
    latest = registry.get("demo-assistant", "latest")
    if latest:
        print(f"Latest: {latest.full_name}")
        print(f"Created: {latest.created_at}")
        print(f"Path: {latest.path}")


def main():
    print("=" * 60)
    print("CUSTOM TRAINING PIPELINE DEMO")
    print("=" * 60)
    print("""
This demo shows the complete training pipeline:

1. Data Pipeline - Collection, curation, formatting
2. Training Loop - QLoRA fine-tuning with SFT/DPO
3. Model Registry - Version tracking and management
""")

    # Run pipeline
    train_data = demo_data_pipeline()
    metrics = demo_training(train_data)
    demo_registry(metrics)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("""
For your full capstone:

1. Data Pipeline
   - Add real data sources (HuggingFace, APIs)
   - Implement semantic deduplication
   - Add data augmentation
   - Build preference pairs for DPO

2. Training Loop
   - Load actual model with QLoRA
   - Implement full SFT training
   - Add DPO training stage
   - Integrate with Weights & Biases

3. Model Registry
   - Connect to MLflow or custom backend
   - Add model evaluation on registration
   - Implement A/B testing support
   - Build deployment pipeline

4. Evaluation
   - LLM-as-judge evaluation
   - Benchmark suite
   - Human evaluation framework
""")


if __name__ == "__main__":
    main()
