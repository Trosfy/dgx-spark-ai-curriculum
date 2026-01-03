#!/usr/bin/env python3
"""
Option E: INT4 Quantization

Applies INT4 quantization to ONNX model for browser deployment.
CRITICAL: Browser ONLY supports INT4, not NF4 or FP4!

Usage:
    python option_e_quantize_int4.py \
        --input-path ./models/matcha-onnx \
        --output-path ./models/matcha-onnx-int4
"""

import argparse
from pathlib import Path
import shutil


def quantize_int4(
    input_path: Path,
    output_path: Path,
    per_channel: bool = True,
) -> None:
    """
    Apply INT4 quantization to ONNX model.

    CRITICAL: Browser deployment requires INT4 format!
    - NOT NF4 (training format)
    - NOT FP4 (not browser compatible)
    - ONLY INT4 works in Transformers.js

    Args:
        input_path: Path to ONNX model
        output_path: Where to save quantized model
        per_channel: Use per-channel quantization (better quality)
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("=" * 70)
    print("INT4 QUANTIZATION")
    print("=" * 70)
    print("\nCRITICAL: Browser ONLY supports INT4, not NF4 or FP4!")

    # Find ONNX model file
    onnx_files = list(input_path.glob("*.onnx"))
    if not onnx_files:
        print(f"\nError: No ONNX files found in {input_path}")
        return

    # Find main model file
    model_file = None
    for f in onnx_files:
        if "model" in f.name.lower():
            model_file = f
            break
    if model_file is None:
        model_file = onnx_files[0]

    print(f"\nSource: {model_file}")
    print(f"Output: {output_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Output file
    output_file = output_path / "model_quantized.onnx"

    # Apply INT4 quantization
    print("\nApplying INT4 quantization...")
    print(f"  Per-channel: {per_channel}")

    quantize_dynamic(
        model_input=str(model_file),
        model_output=str(output_file),
        weight_type=QuantType.QInt4,  # INT4 for browser!
        per_channel=per_channel,
        reduce_range=False,
    )

    # Calculate sizes
    original_size = model_file.stat().st_size / 1e6
    quantized_size = output_file.stat().st_size / 1e6
    compression = (1 - quantized_size / original_size) * 100

    print(f"\nQuantization complete!")
    print(f"  Original: {original_size:.1f} MB")
    print(f"  Quantized: {quantized_size:.1f} MB")
    print(f"  Compression: {compression:.1f}%")

    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ]

    for fname in tokenizer_files:
        src = input_path / fname
        if src.exists():
            dst = output_path / fname
            shutil.copy(src, dst)
            print(f"  Copied {fname}")

    # Create browser config
    import json
    browser_config = {
        "model_type": "gemma",
        "quantization": "int4",
        "framework": "onnx",
        "runtime": "transformers.js",
        "files": [f.name for f in output_path.iterdir() if f.is_file()],
    }
    with open(output_path / "browser_config.json", "w") as f:
        json.dump(browser_config, f, indent=2)

    print("\nFinal package:")
    total_size = 0
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1e6
            total_size += size
            print(f"  {f.name}: {size:.1f} MB")
    print(f"\n  Total: {total_size:.1f} MB")

    print("\nModel is now browser-ready!")


def verify_quantized(model_path: Path) -> bool:
    """Verify the quantized model loads correctly."""
    import onnxruntime as ort

    print("\nVerifying quantized model...")

    model_file = model_path / "model_quantized.onnx"
    if not model_file.exists():
        print(f"  Model not found: {model_file}")
        return False

    try:
        session = ort.InferenceSession(
            str(model_file),
            providers=["CPUExecutionProvider"],
        )
        print(f"  Model loaded successfully!")
        print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in session.get_outputs()]}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT4 Quantization")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    quantize_int4(args.input_path, args.output_path, args.per_channel)

    if args.verify:
        verify_quantized(args.output_path)
