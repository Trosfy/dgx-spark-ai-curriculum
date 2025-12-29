# Module 10: Large Language Model Fine-Tuning

**Phase:** 3 - Advanced  
**Duration:** Weeks 15-17 (15-18 hours)  
**Prerequisites:** Phase 2 complete

---

## Overview

This is where DGX Spark truly shines. With 128GB unified memory, you can fine-tune models that would require cloud GPUs or multiple high-end consumer cards. You'll master LoRA, QLoRA, and full fine-tuning, and successfully fine-tune a **70B parameter model** on your desktop.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Fine-tune LLMs using LoRA, QLoRA, and full fine-tuning
- ✅ Prepare datasets for instruction tuning and chat formats
- ✅ Select appropriate fine-tuning strategies based on resources and goals
- ✅ Evaluate fine-tuned models for task performance

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 10.1 | Explain the mathematical foundations of LoRA | Understand |
| 10.2 | Configure and execute QLoRA fine-tuning for 70B models | Apply |
| 10.3 | Prepare datasets in instruction-following formats | Apply |
| 10.4 | Evaluate and compare fine-tuned models | Evaluate |

---

## Topics

### 10.1 Fine-Tuning Strategies

| Method | Trainable Params | Memory | DGX Spark Capability |
|--------|-----------------|--------|----------------------|
| Full Fine-tuning | 100% | Very High | Up to 8B models |
| LoRA | ~0.1-1% | Low | 8B easily |
| QLoRA | ~0.1-1% + 4-bit | Very Low | **70B+ models** ⭐ |

### 10.2 LoRA Deep Dive
- Low-rank decomposition theory (W = W₀ + BA)
- Rank selection (r parameter)
- Alpha scaling
- Target modules selection

### 10.3 Dataset Preparation
- Instruction format (Alpaca, ShareGPT)
- Chat templates (ChatML, Llama)
- Data quality considerations
- Synthetic data generation

### 10.4 Training Infrastructure
- Gradient checkpointing
- Memory optimization
- Unsloth acceleration
- LLaMA Factory GUI

### 10.5 Preference Optimization
- Reward modeling
- PPO training
- DPO (Direct Preference Optimization)
- ORPO, SimPO variants

---

## Tasks

### Task 10.1: LoRA Theory Notebook
**Time:** 3 hours

Understand LoRA mathematically and implement it.

**Instructions:**
1. Implement LoRA layer from scratch
2. Visualize weight updates during training
3. Experiment with different ranks (r = 4, 8, 16, 32, 64)
4. Plot performance vs rank tradeoff
5. Document connection to SVD

**Deliverable:** LoRA theory notebook with visualizations

---

### Task 10.2: 8B Model LoRA Fine-tuning
**Time:** 3 hours

Fine-tune Llama 3.1 8B with LoRA.

**Instructions:**
1. Load Llama 3.1 8B with 4-bit quantization
2. Apply LoRA to attention layers
3. Prepare instruction dataset
4. Train with Unsloth for acceleration
5. Evaluate on held-out set

**Deliverable:** Fine-tuned 8B model with evaluation

---

### Task 10.3: 70B Model QLoRA ⭐
**Time:** 4 hours

**This is the DGX Spark showcase task!**

**Instructions:**
1. Clear buffer cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
2. Load Llama 3.1 70B with QLoRA config
3. Document memory usage (~45-55GB expected)
4. Fine-tune on custom dataset
5. Compare with 8B results
6. Document the experience (what works on DGX Spark that wouldn't on RTX 4090)

**Deliverable:** 70B fine-tuning notebook with memory analysis

---

### Task 10.4: Dataset Preparation
**Time:** 2 hours

Create instruction datasets.

**Instructions:**
1. Convert raw data to Alpaca format
2. Implement ChatML template
3. Include system prompts
4. Implement data cleaning
5. Create train/val splits

**Deliverable:** Dataset preparation pipeline

---

### Task 10.5: DPO Training
**Time:** 3 hours

Implement preference optimization.

**Instructions:**
1. Create preference pairs dataset
2. Implement DPO loss
3. Train on preference data
4. Compare with SFT-only baseline
5. Evaluate response quality

**Deliverable:** DPO training notebook with comparison

---

### Task 10.6: LLaMA Factory Exploration
**Time:** 2 hours

Use GUI-based fine-tuning.

**Instructions:**
1. Launch LLaMA Factory web UI
2. Configure training through UI
3. Monitor training progress
4. Compare workflow with script-based approach
5. Document pros/cons

**Deliverable:** LLaMA Factory workflow documentation

---

### Task 10.7: Ollama Integration
**Time:** 2 hours

Deploy your fine-tuned model.

**Instructions:**
1. Merge LoRA weights with base model
2. Convert to GGUF format
3. Import to Ollama
4. Test in your custom Web UI
5. Benchmark performance

**Deliverable:** Fine-tuned model running in Ollama

---

## Guidance

### QLoRA Configuration for 70B on DGX Spark

```python
from transformers import BitsAndBytesConfig

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

### LoRA Configuration

```python
from peft import LoraConfig

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
```

### LoRA Rank Guidelines

| Rank | Use Case | Memory | Quality |
|------|----------|--------|---------|
| r=8 | Quick experiments | Minimal | Good |
| r=16 | Most tasks | Low | Great |
| r=32 | Complex tasks | Medium | Excellent |
| r=64 | Maximum adaptation | Higher | Best |
| r=128+ | Approaches full FT | High | Near full FT |

### Dataset Formats

**Alpaca Format:**
```json
{
    "instruction": "Summarize the following text",
    "input": "Long text here...",
    "output": "Summary here..."
}
```

**ChatML Format:**
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

### Memory Management

```python
# CRITICAL: Clear before loading large models
import subprocess
subprocess.run(["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])

# Clear PyTorch cache between runs
import torch, gc
torch.cuda.empty_cache()
gc.collect()
```

### Unsloth for 2x Faster Training

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
# ~2x faster training than standard PEFT
```

---

## Milestone Checklist

- [ ] LoRA theory notebook with from-scratch implementation
- [ ] 8B model successfully fine-tuned with LoRA
- [ ] **70B model fine-tuned with QLoRA** ⭐ (DGX Spark showcase!)
- [ ] Memory usage documented for 70B (~45-55GB)
- [ ] Custom instruction dataset created
- [ ] DPO preference optimization completed
- [ ] LLaMA Factory workflow documented
- [ ] Fine-tuned model running in Ollama via Web UI

---

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM loading 70B | Clear buffer cache first |
| Slow training | Use Unsloth or reduce batch size |
| GGUF conversion fails | Use llama.cpp convert script |
| Ollama won't load model | Check GGUF format and quantization |

---

## Next Steps

After completing this module:
1. ✅ Celebrate! You fine-tuned a 70B model on your desktop!
2. 📁 Save your fine-tuned models
3. ➡️ Proceed to [Module 11: Quantization & Optimization](../module-11-quantization/)

---

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
