# Module 3.1: LLM Fine-Tuning - Quick Reference

## 🚀 Essential Commands

### NGC Container Setup
```bash
# Standard fine-tuning container
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Access at: http://localhost:8888
```

### Memory Clearing (Critical for 70B)
```bash
# Clear buffer cache before loading large models
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

```python
# Clear PyTorch cache between runs
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

## 📊 Key Values to Remember

### LoRA Rank Guidelines
| Rank | Use Case | Memory | Quality |
|------|----------|--------|---------|
| r=8 | Quick experiments | Minimal | Good |
| r=16 | Most tasks (default) | Low | Great |
| r=32 | Complex tasks | Medium | Excellent |
| r=64 | Maximum adaptation | Higher | Best |
| r=128+ | Near full fine-tuning | High | Near full FT |

### Model Memory Requirements
| Model | Full FT | LoRA (FP16) | QLoRA (4-bit) |
|-------|---------|-------------|---------------|
| 8B | ~64GB | ~16GB | ~6GB |
| 70B | ~280GB (OOM) | ~140GB (OOM) | ~45-55GB ✅ |

### Preference Methods Comparison
| Method | Memory | Reference Model? | Best For |
|--------|--------|-----------------|----------|
| DPO | High | Yes | Proven baseline |
| SimPO | Medium | No | Best quality (+6.4 pts) |
| ORPO | Low | No | Memory constrained |
| KTO | Medium | No | Binary feedback only |

## 🔧 Common Patterns

### Pattern: Standard LoRA Configuration
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # Rank: 8-64 typical
    lora_alpha=32,           # Scaling: usually 2*r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

### Pattern: QLoRA for 70B Models
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Expected memory: ~45-55GB
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

### Pattern: DoRA (Drop-in Improvement)
```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    use_dora=True  # +3.7 points improvement!
)
```

### Pattern: NEFTune with TRL
```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./output",
    neftune_noise_alpha=5,  # Recommended: 5-15
    # ... other config
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=config,
)
# 29.8% → 64.7% on AlpacaEval!
```

### Pattern: SimPO Training
```python
from trl import SimPOTrainer, SimPOConfig

config = SimPOConfig(
    output_dir="./simpo_output",
    beta=2.0,
    gamma_beta_ratio=0.5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
)

trainer = SimPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Pattern: Unsloth 2x Faster Training
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
# ~2x faster than standard PEFT
```

## 📝 Dataset Formats

### Alpaca Format
```json
{
    "instruction": "Summarize the following text",
    "input": "Long text here...",
    "output": "Summary here..."
}
```

### ChatML Format
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
Hi! How can I help?
<|im_end|>
```

### Llama 3 Chat Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Hello!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Hi! How can I help?<|eot_id|>
```

### DPO Preference Format
```json
{
    "prompt": "Write a haiku about coding",
    "chosen": "Lines of code flow bright\nDebugging through the long night\nSoftware takes its flight",
    "rejected": "coding is fun\ni like to code\ncode code code"
}
```

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| OOM loading 70B | Clear buffer cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `load_in_8bit` AND `load_in_4bit` | Use only ONE quantization option |
| Missing `device_map="auto"` | Always include for large models |
| Wrong dtype | Use `torch.bfloat16` on DGX Spark (Blackwell native) |
| LoRA alpha too low | Set `lora_alpha = 2 * r` as starting point |
| Pad token issues | Set `tokenizer.pad_token = tokenizer.eos_token` |
| Gradient explosion | Enable gradient checkpointing for stability |

## 🔗 Quick Links
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DoRA Paper](https://arxiv.org/abs/2402.09353)
- [NEFTune Paper](https://arxiv.org/abs/2310.05914)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [SimPO Paper](https://arxiv.org/abs/2405.14734)
- [ORPO Paper](https://arxiv.org/abs/2403.07691)
- [KTO Paper](https://arxiv.org/abs/2402.01306)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
