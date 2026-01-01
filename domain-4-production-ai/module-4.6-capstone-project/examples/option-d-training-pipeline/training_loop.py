#!/usr/bin/env python3
"""
Training Loop Implementation

SFT and DPO training with QLoRA.
This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-8B-Instruct"
    use_4bit: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03

    # Output
    output_dir: str = "./outputs"
    save_steps: int = 100
    logging_steps: int = 10


@dataclass
class TrainingMetrics:
    """Metrics from training."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    samples_seen: int
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class QLoRATrainer:
    """
    QLoRA trainer for LLM fine-tuning.

    Supports:
    - Supervised Fine-Tuning (SFT)
    - Direct Preference Optimization (DPO)
    """

    def __init__(self, config: TrainingConfig, load_model: bool = False):
        self.config = config
        self.metrics_history: List[TrainingMetrics] = []
        self._model = None
        self._tokenizer = None

        if load_model:
            self._load_model()

    def _load_model(self):
        """Load model with QLoRA configuration."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            print(f"Loading model: {self.config.model_name}")

            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            # Load base model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Prepare for k-bit training
            self._model = prepare_model_for_kbit_training(self._model)

            # Add LoRA adapters
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self._model = get_peft_model(self._model, lora_config)

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Print trainable parameters
            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self._model.parameters())
            print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        except Exception as e:
            print(f"Could not load model: {e}")
            print("Running in demo mode")

    def train_sft(self, train_data: List[Dict], eval_data: Optional[List[Dict]] = None) -> Dict:
        """
        Supervised Fine-Tuning.

        Args:
            train_data: Training samples in chat format
            eval_data: Optional evaluation samples

        Returns:
            Training results
        """
        print("\n" + "=" * 50)
        print("SUPERVISED FINE-TUNING")
        print("=" * 50)

        if self._model is None:
            return self._demo_train(train_data, "SFT")

        # Real training with SFTTrainer
        try:
            from trl import SFTTrainer, SFTConfig

            training_args = SFTConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                max_seq_length=self.config.max_seq_length,
                bf16=True,
            )

            trainer = SFTTrainer(
                model=self._model,
                args=training_args,
                train_dataset=train_data,  # Would need proper Dataset
                processing_class=self._tokenizer,
            )

            trainer.train()
            return {"status": "completed", "method": "SFT"}

        except Exception as e:
            print(f"Training error: {e}")
            return self._demo_train(train_data, "SFT")

    def train_dpo(self, train_data: List[Dict], eval_data: Optional[List[Dict]] = None) -> Dict:
        """
        Direct Preference Optimization.

        Args:
            train_data: Preference pairs (prompt, chosen, rejected)
            eval_data: Optional evaluation samples

        Returns:
            Training results
        """
        print("\n" + "=" * 50)
        print("DIRECT PREFERENCE OPTIMIZATION")
        print("=" * 50)

        if self._model is None:
            return self._demo_train(train_data, "DPO")

        # Real training with DPOTrainer
        try:
            from trl import DPOTrainer, DPOConfig

            training_args = DPOConfig(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                max_length=self.config.max_seq_length,
                bf16=True,
                beta=0.1,  # DPO temperature
            )

            # Would need reference model
            trainer = DPOTrainer(
                model=self._model,
                ref_model=None,  # Use implicit reference
                args=training_args,
                train_dataset=train_data,
                processing_class=self._tokenizer,
            )

            trainer.train()
            return {"status": "completed", "method": "DPO"}

        except Exception as e:
            print(f"Training error: {e}")
            return self._demo_train(train_data, "DPO")

    def _demo_train(self, data: List[Dict], method: str) -> Dict:
        """Demo training without model."""
        print(f"\n[DEMO MODE] Simulating {method} training...")
        print(f"Dataset size: {len(data)} samples")
        print(f"Config: {self.config.num_epochs} epochs, batch={self.config.batch_size}")

        # Simulate training progress
        total_steps = len(data) * self.config.num_epochs // self.config.batch_size
        for epoch in range(self.config.num_epochs):
            for step in range(min(5, total_steps)):  # Limit for demo
                loss = 2.5 - (epoch * 0.3) - (step * 0.1)  # Simulated decreasing loss
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    step=step + 1,
                    loss=max(0.5, loss),
                    learning_rate=self.config.learning_rate,
                    samples_seen=(step + 1) * self.config.batch_size
                )
                self.metrics_history.append(metrics)
                print(f"  Epoch {epoch+1}, Step {step+1}: loss={metrics.loss:.4f}")

        return {
            "status": "completed (demo)",
            "method": method,
            "final_loss": self.metrics_history[-1].loss if self.metrics_history else None,
            "total_steps": len(self.metrics_history)
        }

    def save_model(self, path: str):
        """Save trained model."""
        if self._model is not None:
            self._model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
            print(f"Model saved to: {path}")
        else:
            print(f"[DEMO] Would save model to: {path}")

    def get_metrics_summary(self) -> Dict:
        """Get training metrics summary."""
        if not self.metrics_history:
            return {}

        losses = [m.loss for m in self.metrics_history]
        return {
            "total_steps": len(self.metrics_history),
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "avg_loss": sum(losses) / len(losses)
        }


# Example usage
if __name__ == "__main__":
    print("Training Loop Demo")
    print("=" * 50)

    # Create config
    config = TrainingConfig(
        model_name="Qwen/Qwen3-8B-Instruct",
        num_epochs=2,
        batch_size=4,
        learning_rate=2e-4,
    )

    print("\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")

    # Create trainer (demo mode)
    trainer = QLoRATrainer(config, load_model=False)

    # Mock training data
    train_data = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
        {"messages": [{"role": "user", "content": "Explain AI"}, {"role": "assistant", "content": "AI is..."}]},
    ] * 10  # Replicate for demo

    # Run SFT
    result = trainer.train_sft(train_data)
    print(f"\nSFT Result: {result}")

    # Get summary
    summary = trainer.get_metrics_summary()
    print(f"\nMetrics Summary: {json.dumps(summary, indent=2)}")
