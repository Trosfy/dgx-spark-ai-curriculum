#!/usr/bin/env python3
"""
Option E: LoRA Adapter Merging

Merges LoRA adapters into the base model in full precision (BF16).
CRITICAL: Must merge in full precision for quality preservation!

Usage:
    python option_e_merge_adapters.py \
        --base-model google/gemma-3-1b-it \
        --adapter-path ./models/matcha-lora \
        --output-path ./models/matcha-merged
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapters(
    base_model_name: str,
    adapter_path: Path,
    output_path: Path,
    device_map: str = "auto",
) -> None:
    """
    Merge LoRA adapters into base model.

    CRITICAL: Loads base model in BF16 (full precision) for quality!
    Do NOT merge while model is in 4-bit quantization.

    Args:
        base_model_name: HuggingFace model ID
        adapter_path: Path to LoRA adapters
        output_path: Where to save merged model
        device_map: Device mapping strategy
    """
    print("=" * 70)
    print("MERGING LORA ADAPTERS")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model in FULL PRECISION (BF16)
    # CRITICAL: Do NOT use quantization here!
    print(f"\nLoading base model in BF16 (full precision)...")
    print("  This is CRITICAL for quality - do NOT load in 4-bit!")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,  # Full precision!
        device_map=device_map,
        # NO quantization_config here!
    )

    print(f"  Model dtype: {base_model.dtype}")
    print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Load LoRA adapters
    print(f"\nLoading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch.bfloat16,
    )

    # Merge adapters into base model
    print("\nMerging adapters...")
    merged_model = model.merge_and_unload()

    print(f"  Merged model type: {type(merged_model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in merged_model.parameters()):,}")

    # Save merged model
    print(f"\nSaving to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(output_path))

    # Calculate size
    model_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors")) / 1e9

    print(f"\nMerge complete!")
    print(f"  Path: {output_path}")
    print(f"  Size: {model_size:.2f} GB")

    # List files
    print("\nSaved files:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1e6
            print(f"  {f.name}: {size:.1f} MB")


def verify_merge(merged_path: Path, tokenizer_path: Path = None) -> None:
    """Verify the merged model works correctly."""
    print("\nVerifying merged model...")

    tokenizer_path = tokenizer_path or merged_path

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Test generation
    messages = [
        {"role": "system", "content": "You are a matcha expert."},
        {"role": "user", "content": "What is ceremonial grade matcha?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    print("\nTest generation:")
    print(f"  Q: What is ceremonial grade matcha?")
    print(f"  A: {response[:200]}...")

    print("\nVerification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters")
    parser.add_argument("--base-model", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Verify after merging")
    args = parser.parse_args()

    merge_adapters(
        args.base_model,
        args.adapter_path,
        args.output_path,
    )

    if args.verify:
        verify_merge(args.output_path)
