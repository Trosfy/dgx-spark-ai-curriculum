# Module 2.6: Diffusion Models - Study Guide

## Learning Objectives

By the end of this module, you will be able to:

1. **Explain diffusion theory** - forward/reverse process, noise schedules
2. **Generate images** with SDXL and Flux
3. **Use ControlNet** for guided generation
4. **Train custom LoRAs** for personal styles
5. **Build production pipelines** with reproducibility and batch generation

---

## Module Roadmap

| # | Lab | Focus | Time | Key Outcome |
|---|-----|-------|------|-------------|
| 1 | lab-2.6.1-diffusion-theory.ipynb | DDPM from scratch | ~2 hr | Understand noise/denoise |
| 2 | lab-2.6.2-stable-diffusion.ipynb | SDXL generation | ~2 hr | Professional image generation |
| 3 | lab-2.6.3-controlnet.ipynb | Guided generation | ~2 hr | Edge/depth control |
| 4 | lab-2.6.4-flux-exploration.ipynb | Next-gen models | ~2 hr | Flux comparison |
| 5 | lab-2.6.5-lora-training.ipynb | Custom styles | ~2 hr | Trained LoRA adapter |
| 6 | lab-2.6.6-generation-pipeline.ipynb | Production system | ~2 hr | Complete pipeline |

**Total time**: ~10-12 hours

---

## Core Concepts

### Forward Diffusion
**What**: Gradually adding noise to images until they become pure noise
**Why it matters**: This is how we create training data - we know what noise was added
**First appears in**: Lab 2.6.1

### Reverse Diffusion (Denoising)
**What**: Training a model to predict and remove noise, step by step
**Why it matters**: Generation is just repeated denoising from pure noise to image
**First appears in**: Lab 2.6.1

### Noise Schedule
**What**: How much noise to add at each timestep (linear vs cosine)
**Why it matters**: Affects generation quality and speed
**First appears in**: Lab 2.6.1

### Classifier-Free Guidance
**What**: Combining conditional and unconditional predictions for stronger adherence to prompts
**Why it matters**: Higher guidance = more prompt-following; lower = more creativity
**First appears in**: Lab 2.6.2

### ControlNet
**What**: Additional conditioning (edges, depth, pose) for structural control
**Why it matters**: Maintain composition while changing style/content
**First appears in**: Lab 2.6.3

---

## How This Module Connects

```
Previous                    This Module                 Next
─────────────────────────────────────────────────────────────
Module 2.5              ►   Module 2.6              ►  Domain 3
HuggingFace                 Diffusion                   LLM Systems
(ecosystem)                 (image gen)                 (text)
```

**Builds on**:
- **HuggingFace ecosystem** from Module 2.5 (Diffusers is HF library)
- **U-Net architecture** from Module 2.2 (diffusion uses U-Net)
- **LoRA concepts** from Module 2.5 (same PEFT techniques)

**Prepares for**:
- **Module 4.1** (Multimodal) will combine image and text generation
- **Module 4.5** (Demo Building) will create Gradio interfaces for generation
- Understanding diffusion helps with video/audio generation (future)

---

## Recommended Approach

### Standard Path (10-12 hours)
1. **Start with Lab 2.6.1** - Theory is essential for understanding parameters
2. **Work through Lab 2.6.2** - Master SDXL prompt engineering
3. **Complete Lab 2.6.3** - ControlNet is powerful for practical applications
4. **Explore Lab 2.6.4** - Flux is the future of diffusion
5. **Apply Lab 2.6.5** - Train your own style
6. **Build Lab 2.6.6** - Create a reusable system

### Quick Path (6-8 hours)
1. Skim Lab 2.6.1 - Focus on intuition, not implementation
2. Focus on Lab 2.6.2 - Prompt engineering is the key skill
3. Complete Lab 2.6.5 - LoRA training for customization
4. Skip theory-heavy labs if time-constrained

### Deep-Dive Path (15+ hours)
1. Implement DDPM from scratch with multiple schedulers
2. Compare all SDXL variants and refiners
3. Train LoRAs on custom datasets
4. Build a complete generation service

---

## DGX Spark Performance

| Model | Resolution | Steps | Time | Memory |
|-------|------------|-------|------|--------|
| SDXL Base | 1024×1024 | 30 | ~5-8s | ~7 GB |
| SDXL + Refiner | 1024×1024 | 50 | ~12-15s | ~14 GB |
| Flux-schnell | 1024×1024 | 4 | ~3-4s | ~12 GB |
| Flux-dev | 1024×1024 | 50 | ~15-20s | ~12 GB |
| SDXL + ControlNet | 1024×1024 | 30 | ~8-10s | ~10 GB |

DGX Spark's 128GB easily handles all models at full precision, even with multiple ControlNets.

---

## Generation Parameters Guide

| Parameter | Low Value | High Value | Effect |
|-----------|-----------|------------|--------|
| `guidance_scale` | 1-3 | 10-15 | Prompt adherence vs creativity |
| `num_inference_steps` | 10-20 | 50-100 | Speed vs quality |
| `negative_prompt` | None | Detailed | What to avoid |
| `seed` | Random | Fixed | Reproducibility |

---

## Before You Start

- See [QUICKSTART.md](./QUICKSTART.md) for 5-minute first generation
- See [LAB_PREP.md](./LAB_PREP.md) for downloads and setup
- See [ELI5.md](./ELI5.md) for intuitive explanations

---

## Common Challenges

| Challenge | Solution |
|-----------|----------|
| "SDXL OOM" | Use `enable_vae_tiling()` or reduce resolution |
| "Blurry generations" | Increase steps (30+), add quality tokens to prompt |
| "ControlNet not following" | Increase `controlnet_conditioning_scale` |
| "LoRA not applying" | Check path, trigger word, and weight |
| "Flux slow" | Use Flux-schnell (4 steps) for testing |

---

## Success Metrics

You've mastered this module when you can:

- [ ] Explain the diffusion process (forward and reverse)
- [ ] Generate high-quality images with effective prompts
- [ ] Use ControlNet for structural guidance
- [ ] Train a LoRA for a custom style
- [ ] Build a reproducible generation pipeline
