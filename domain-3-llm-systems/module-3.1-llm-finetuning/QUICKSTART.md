# Module 3.1: LLM Fine-Tuning - Quickstart

## ⏱️ Time: ~5 minutes

## 🎯 What You'll Build
Fine-tune an LLM with LoRA in under 5 minutes and see the difference immediately.

## ✅ Before You Start
- [ ] DGX Spark NGC container running
- [ ] HuggingFace account (for model access)

## 🚀 Let's Go!

### Step 1: Start the Container
```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Access at: http://localhost:8888
```

### Step 2: Install Dependencies
```bash
pip install peft transformers accelerate bitsandbytes
```

### Step 3: Load Model with LoRA
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load a small model for quick demo
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Add LoRA adapters
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### Step 4: See the Difference
```python
# Check trainable parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
```

**Expected output**:
```
trainable params: 524,288 || all params: 1,100,572,672 || trainable%: 0.0476
Total: 1,100,572,672 | Trainable: 524,288 (0.05%)
```

## 🎉 You Did It!

You just added LoRA adapters to an LLM! With only **0.05% trainable parameters**, you can now fine-tune the model on your custom data. This same technique scales to 70B models on DGX Spark.

In the full module, you'll learn:
- **QLoRA**: Fine-tune 70B models in 45GB memory
- **DoRA**: +3.7 points improvement on benchmarks
- **NEFTune**: 29.8% → 64.7% on AlpacaEval with 5 lines of code
- **Preference optimization**: DPO, SimPO, ORPO, KTO

## ▶️ Next Steps
1. **Understand LoRA math**: See [Lab 3.1.1](./labs/lab-3.1.1-lora-theory.ipynb)
2. **Try 8B model**: See [Lab 3.1.4](./labs/lab-3.1.4-8b-lora-finetuning.ipynb)
3. **Full setup**: Start with [LAB_PREP.md](./LAB_PREP.md)
