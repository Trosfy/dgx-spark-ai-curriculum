#!/usr/bin/env python3
"""
Option E: ONNX Conversion

Exports merged model to ONNX format for browser deployment.

Usage:
    python option_e_convert_to_onnx.py \
        --model-path ./models/matcha-merged \
        --output-path ./models/matcha-onnx
"""

import argparse
from pathlib import Path
import torch


def export_to_onnx(
    model_path: Path,
    output_path: Path,
    task: str = "text-generation-with-past",
) -> None:
    """
    Export HuggingFace model to ONNX format.

    Uses optimum library for clean export with KV cache support.

    Args:
        model_path: Path to merged model
        output_path: Where to save ONNX model
        task: Export task type
    """
    from optimum.exporters.onnx import main_export

    print("=" * 70)
    print("ONNX EXPORT")
    print("=" * 70)

    print(f"\nSource: {model_path}")
    print(f"Output: {output_path}")
    print(f"Task: {task}")

    # Check model exists
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Export
    print("\nExporting to ONNX (this may take 5-10 minutes)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    main_export(
        model_name_or_path=str(model_path),
        output=str(output_path),
        task=task,
        device=device,
        fp16=False,  # Export as FP32 first, quantize separately
    )

    # Calculate size
    onnx_files = list(output_path.glob("*.onnx"))
    total_size = sum(f.stat().st_size for f in onnx_files) / 1e9

    print(f"\nExport complete!")
    print(f"  Path: {output_path}")
    print(f"  Size: {total_size:.2f} GB")

    # List files
    print("\nGenerated files:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size = f.stat().st_size / 1e6
            print(f"  {f.name}: {size:.1f} MB")


def verify_onnx(onnx_path: Path) -> bool:
    """Verify the ONNX model is valid."""
    import onnx

    print("\nVerifying ONNX model...")

    onnx_files = list(onnx_path.glob("*.onnx"))
    if not onnx_files:
        print("  No ONNX files found!")
        return False

    for onnx_file in onnx_files:
        print(f"  Checking {onnx_file.name}...")
        try:
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
            print(f"    Valid!")
        except Exception as e:
            print(f"    Error: {e}")
            return False

    print("\nAll ONNX models verified!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export to ONNX")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--task", default="text-generation-with-past")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_path, args.task)

    if args.verify:
        verify_onnx(args.output_path)
