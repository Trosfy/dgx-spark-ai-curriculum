# Module 3.1: Large Language Model Fine-Tuning

**Domain:** 3 - LLM Systems
**Duration:** Weeks 16-18 (18-22 hours)
**Prerequisites:** Module 2.6 (Diffusion Models)
**Priority:** P1 Expanded (DoRA, NEFTune, SimPO, ORPO, KTO)

---

## Overview

This is where DGX Spark truly shines. With 128GB unified memory, you can fine-tune models that would require cloud GPUs or multiple high-end consumer cards. You'll master LoRA, QLoRA, DoRA, and advanced preference optimization methods—and successfully fine-tune a **70B parameter model** on your desktop.

**What's New in v2.0:** This module now covers cutting-edge techniques like DoRA (+3.7 points on commonsense reasoning), NEFTune (29.8% → 64.7% on AlpacaEval!), SimPO, ORPO, and KTO for preference learning.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Fine-tune LLMs using LoRA, QLoRA, DoRA, and full fine-tuning
- ✅ Apply modern alignment techniques (DPO, SimPO, ORPO, KTO)
- ✅ Prepare datasets for instruction tuning and preference learning
- ✅ Fine-tune 70B+ models on DGX Spark

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 3.1.1 | Explain mathematical foundations of LoRA and DoRA | Understand |
| 3.1.2 | Configure and execute QLoRA fine-tuning for 70B models | Apply |
| 3.1.3 | Implement preference optimization with DPO, SimPO, ORPO, KTO | Apply |
| 3.1.4 | Apply NEFTune for improved fine-tuning | Apply |

---

## Topics

### 3.1.1 Fine-Tuning Strategies

| Method | Trainable Params | Memory | DGX Spark Capability |
|--------|-----------------|--------|----------------------|
| Full Fine-tuning | 100% | Very High | Up to 8B models |
| LoRA | ~0.1-1% | Low | 8B easily |
| QLoRA | ~0.1-1% + 4-bit | Very Low | **70B+ models** ⭐ |

### 3.1.2 LoRA Deep Dive
- Low-rank decomposition theory (W = W₀ + BA)
- Rank selection (r parameter)
- Alpha scaling
- Target modules selection

### 3.1.3 Advanced LoRA Variants [P1 Expansion]

- **DoRA (Weight-Decomposed LoRA)**
  - Decomposes weights into magnitude and direction
  - +3.7 points on commonsense reasoning benchmarks
  - Drop-in replacement for LoRA

- **rsLoRA (Rank-Stabilized LoRA)**
  - Better scaling at higher ranks
  - Improved training stability

- **QA-LoRA**
  - Quantization-aware LoRA training
  - Better quality when using quantized base

### 3.1.4 Training Enhancements [P1 Expansion]

- **NEFTune (Noisy Embedding Fine-Tuning)**
  - Add noise to input embeddings during training
  - 29.8% → 64.7% on AlpacaEval! (5 lines of code)
  - Works with any fine-tuning method

- **Gradient Checkpointing**
  - Trade compute for memory
  - Essential for 70B training

- **Flash Attention Integration**
  - 2-4x faster attention computation
  - Native in modern transformers

- **Unsloth 2x Speedup**
  - Optimized kernels for LoRA training
  - Automatic gradient checkpointing

### 3.1.5 Dataset Preparation
- Instruction format (Alpaca, ShareGPT, OpenAI)
- Chat templates (ChatML, Llama)
- Data quality filtering
- Synthetic data generation basics

### 3.1.6 Preference Optimization [P1 Expansion]

- **Reward Modeling Overview**
  - Bradley-Terry model
  - Training reward models

- **DPO (Direct Preference Optimization)**
  - No reward model needed
  - Proven, well-understood default

- **SimPO (Simple Preference Optimization)**
  - +6.4 points on AlpacaEval vs DPO
  - No reference model needed
  - Simpler implementation

- **ORPO (Odds Ratio Preference Optimization)**
  - 50% less memory than DPO
  - Single training stage
  - Great for memory-constrained setups

- **KTO (Kahneman-Tversky Optimization)** [P2]
  - Works with binary feedback (thumbs up/down)
  - No preference pairs required
  - Human-aligned loss function

- **Choosing the Right Method**
  - DPO: Proven default, good baseline
  - SimPO: Better quality, simpler
  - ORPO: Memory constrained? Use this
  - KTO: Only binary feedback? Use this

### 3.1.7 Training Infrastructure
- LLaMA Factory GUI
- Axolotl configuration
- TRL library usage

---

## Labs

### Lab 3.1.1: LoRA Theory Notebook
**Time:** 2 hours

Understand LoRA mathematically and implement it.

**Instructions:**
1. Implement LoRA layer from scratch
2. Visualize weight updates during training
3. Experiment with different ranks (r = 4, 8, 16, 32, 64)
4. Plot performance vs rank tradeoff
5. Document connection to SVD

**Deliverable:** LoRA theory notebook with visualizations

---

### Lab 3.1.2: DoRA Comparison [P1]
**Time:** 2 hours

Compare LoRA vs DoRA for improved fine-tuning.

**Instructions:**
1. Fine-tune same model with standard LoRA
2. Fine-tune with DoRA (weight-decomposed)
3. Use identical hyperparameters for fair comparison
4. Evaluate on benchmark (e.g., commonsense reasoning)
5. Document quality improvement
6. Compare training time and memory

**Deliverable:** DoRA comparison notebook showing improvement

---

### Lab 3.1.3: NEFTune Magic [P1]
**Time:** 1 hour

Add NEFTune for dramatic quality boost.

**Instructions:**
1. Implement NEFTune (5 lines of code!)
2. Train with and without NEFTune
3. Evaluate on AlpacaEval or similar
4. Measure the improvement
5. Document optimal noise level

**Deliverable:** NEFTune implementation with measured improvement

---

### Lab 3.1.4: 8B Model LoRA Fine-tuning
**Time:** 3 hours

Fine-tune Llama 3.1 8B with LoRA + NEFTune.

**Instructions:**
1. Load Llama 3.1 8B with 4-bit quantization
2. Apply LoRA to attention layers
3. Enable NEFTune for quality boost
4. Train with Unsloth for acceleration
5. Evaluate on held-out set

**Deliverable:** Fine-tuned 8B model with evaluation

---

### Lab 3.1.5: 70B Model QLoRA ⭐
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

### Lab 3.1.6: Dataset Preparation
**Time:** 2 hours

Create instruction and preference datasets.

**Instructions:**
1. Convert raw data to Alpaca format
2. Implement ChatML template
3. Include system prompts
4. Create preference pairs for DPO
5. Implement data cleaning and quality filtering
6. Create train/val splits

**Deliverable:** Dataset preparation pipeline

---

### Lab 3.1.7: DPO Training
**Time:** 2 hours

Implement Direct Preference Optimization.

**Instructions:**
1. Create preference pairs dataset
2. Configure DPO training with TRL
3. Train on preference data
4. Compare with SFT-only baseline
5. Evaluate response quality

**Deliverable:** DPO training notebook with comparison

---

### Lab 3.1.8: SimPO vs ORPO [P1]
**Time:** 2 hours

Compare modern preference optimization methods.

**Instructions:**
1. Train same model with SimPO
2. Train same model with ORPO
3. Compare quality on evaluation set
4. Compare memory usage (ORPO should use 50% less)
5. Document when to use each method

**Deliverable:** SimPO vs ORPO comparison with recommendations

---

### Lab 3.1.9: KTO for Binary Feedback [P2]
**Time:** 2 hours

Train with thumbs up/down data.

**Instructions:**
1. Create binary feedback dataset
2. Configure KTO training
3. Train model on binary signals
4. Compare with DPO baseline
5. Document use cases for KTO

**Deliverable:** KTO training notebook with binary feedback

---

### Lab 3.1.10: Ollama Integration
**Time:** 2 hours

Deploy your fine-tuned model.

**Instructions:**
1. Merge LoRA weights with base model
2. Convert to GGUF format
3. Import to Ollama
4. Test in JupyterLab
5. Benchmark performance

**Deliverable:** Fine-tuned model running in Ollama

---

## Guidance

### Model Access Requirements

**Important:** Llama 3.1 models require approval from Meta before download.

1. **Request Access:** Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and request access
2. **Login to HuggingFace:**
   ```bash
   huggingface-cli login
   ```
3. **Alternative Models (No Approval Required):**
   - `mistralai/Mistral-7B-Instruct-v0.2`
   - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (for quick testing)

### Launching NGC Container for This Module

```bash
# Recommended: NGC PyTorch container with all dependencies
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Access at: http://localhost:8888
# The NGC container includes: PyTorch, CUDA, cuDNN, bitsandbytes (ARM64-optimized)

# Additional packages (usually pre-installed, but install if missing):
# pip install psutil  # For memory monitoring in notebook 03

# NETWORK OPTIONS:
# If you need to access external services (HuggingFace Hub, Ollama, etc.):
#   Option 1: Port mapping (shown above with -p 8888:8888)
#   Option 2: Host networking (simpler, but less isolated):
#             docker run --gpus all -it --rm --network=host ...
#
# Use --network=host when:
#   - Connecting to Ollama running on the host
#   - Downloading models from HuggingFace Hub
#   - Running LLaMA Factory web UI that needs external access
```

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

### DoRA Configuration [P1]

```python
from peft import LoraConfig

# DoRA: Weight-Decomposed Low-Rank Adaptation
# Just add use_dora=True to your LoRA config!
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    use_dora=True  # Enable DoRA!
)

# DoRA decomposes weights into magnitude and direction
# Provides +3.7 points improvement on commonsense reasoning
```

### NEFTune Implementation [P1]

```python
# NEFTune: 5 lines for massive quality improvement!
def neftune_forward(self, input_ids):
    embeddings = self.original_forward(input_ids)
    if self.training:
        # Add noise scaled by sequence length
        noise = torch.randn_like(embeddings) * (self.neftune_alpha / embeddings.size(1)**0.5)
        embeddings = embeddings + noise
    return embeddings

# Or use TRL's built-in NEFTune:
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    neftune_noise_alpha=5,  # Recommended: 5-15
    ...
)
# 29.8% → 64.7% on AlpacaEval!
```

### Choosing Preference Optimization Method

| Method | Best For | Memory | Reference Model? |
|--------|----------|--------|-----------------|
| DPO | Proven baseline | High | Yes |
| SimPO | Better quality | Medium | No |
| ORPO | Memory-constrained | Low | No |
| KTO | Binary feedback only | Medium | No |

```python
# SimPO: Simpler and better than DPO
from trl import SimPOTrainer, SimPOConfig

config = SimPOConfig(
    beta=2.0,
    gamma_beta_ratio=0.5,
    ...
)

trainer = SimPOTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
)
```

---

## Milestone Checklist

- [ ] LoRA theory notebook with from-scratch implementation
- [ ] DoRA comparison showing improvement [P1]
- [ ] NEFTune improvement measured [P1]
- [ ] 8B model fine-tuned with LoRA + NEFTune
- [ ] **70B model fine-tuned with QLoRA** ⭐ (DGX Spark showcase!)
- [ ] Memory usage documented for 70B (~45-55GB)
- [ ] Custom instruction dataset created
- [ ] DPO preference optimization completed
- [ ] SimPO and ORPO compared [P1]
- [ ] KTO trained with binary feedback [P2]
- [ ] Fine-tuned model running in Ollama

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
3. ➡️ Proceed to [Module 3.2: Quantization & Optimization](../module-3.2-quantization/)

---

## Module Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Module 2.6: Diffusion Models](../../domain-2-deep-learning-frameworks/module-2.6-diffusion-models/) | **Module 3.1: LLM Fine-Tuning** | [Module 3.2: Quantization](../module-3.2-quantization/) |

---

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DoRA Paper](https://arxiv.org/abs/2402.09353) - Weight-Decomposed LoRA
- [NEFTune Paper](https://arxiv.org/abs/2310.05914) - Noisy Embeddings
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [SimPO Paper](https://arxiv.org/abs/2405.14734) - Simple Preference Optimization
- [ORPO Paper](https://arxiv.org/abs/2403.07691) - Odds Ratio Preference
- [KTO Paper](https://arxiv.org/abs/2402.01306) - Kahneman-Tversky Optimization
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
- [TRL Documentation](https://huggingface.co/docs/trl)
