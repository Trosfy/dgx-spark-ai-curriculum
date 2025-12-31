# Example: Custom Training Pipeline

This is a minimal working example for Option D: Custom Training Pipeline.

## What This Example Includes

- `data_pipeline.py` - Data collection and curation
- `training_loop.py` - Training implementation with SFT/DPO
- `model_registry.py` - Model versioning and tracking
- `demo.py` - Quick demonstration script

## Quick Start

```bash
# Navigate to this directory
cd examples/option-d-training-pipeline

# Run the demo
python demo.py
```

## Files Overview

### data_pipeline.py

Data handling with:
- Data collection from multiple sources
- Quality filtering and deduplication
- Format conversion to training format
- Dataset statistics

### training_loop.py

Training implementation with:
- QLoRA fine-tuning setup
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- Training metrics logging

### model_registry.py

Model management with:
- Version tracking
- Checkpoint management
- Metadata storage
- Model comparison

### demo.py

Interactive demo showing:
- Data pipeline execution
- Training configuration
- Model registration

## Extending This Example

To build your full capstone, you'll want to:

1. **Add real data sources** (APIs, scrapers, datasets)
2. **Implement full QLoRA** with PEFT library
3. **Add DPO training** for preference alignment
4. **Build evaluation suite** for model comparison
5. **Create experiment tracking** with MLflow/W&B
6. **Add distributed training** support

## Memory Requirements

This example is designed to run on DGX Spark:
- Llama 3.3 8B with QLoRA
- 4-bit quantization
- Estimated: ~12GB GPU memory

For 70B models, use 4-bit quantization (~35GB).

## Next Steps

1. Review the code in each file
2. Run the demo to see the pipeline
3. Add your specific training data
4. Follow the main Option D notebook for full implementation
