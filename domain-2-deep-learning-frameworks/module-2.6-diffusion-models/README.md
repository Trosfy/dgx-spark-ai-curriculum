# Module 2.6: Diffusion Models

**Domain:** 2 - Deep Learning Frameworks  
**Duration:** Week 15 (10-12 hours)  
**Prerequisites:** Module 2.5 (Hugging Face Ecosystem)  
**Priority:** P1 High

---

## Overview

Diffusion models have revolutionized image generation—from Midjourney to DALL-E 3, they're behind the AI art explosion. This module covers the theory and practice of diffusion models: from understanding the math of adding and removing noise, to generating images with Stable Diffusion and Flux, to training your own custom styles with LoRA.

On DGX Spark, you can run the largest diffusion models (SDXL, Flux) at full precision, train LoRAs without memory constraints, and even experiment with video generation models.

---

## Directory Structure

```
module-2.6-diffusion-models/
├── README.md                    # This file
├── labs/
│   ├── lab-2.6.1-diffusion-theory.ipynb     # DDPM from scratch
│   ├── lab-2.6.2-stable-diffusion.ipynb     # SDXL generation
│   ├── lab-2.6.3-controlnet.ipynb           # ControlNet workshop
│   ├── lab-2.6.4-flux-exploration.ipynb     # Flux comparison
│   ├── lab-2.6.5-lora-training.ipynb        # Custom style LoRA
│   └── lab-2.6.6-generation-pipeline.ipynb  # End-to-end system
├── scripts/
│   ├── __init__.py
│   ├── diffusion_utils.py       # Core diffusion operations
│   ├── image_utils.py           # Image preprocessing
│   └── training_utils.py        # LoRA training helpers
├── solutions/
│   └── (solution notebooks)
└── data/
    ├── README.md                # Data documentation
    └── sample_prompts.txt       # Example prompts
```

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Explain the theory behind diffusion models (forward and reverse processes)
- ✅ Generate images using Stable Diffusion and Flux
- ✅ Apply ControlNet for guided/structured image generation
- ✅ Fine-tune diffusion models with LoRA for custom styles

---

## Labs

### Lab 2.6.1: Diffusion Theory (2 hours)
**Objective:** Understand diffusion from first principles

Build a simple DDPM from scratch:
- Implement forward diffusion (noise addition)
- Visualize noise schedules (linear vs cosine)
- Build a U-Net denoiser
- Train on MNIST
- Generate new digits from noise

### Lab 2.6.2: Stable Diffusion Generation (2 hours)
**Objective:** Master SDXL for text-to-image generation

Learn professional image generation:
- Load and run SDXL on DGX Spark
- Master prompt engineering techniques
- Understand guidance scale effects
- Use negative prompts effectively
- Generate various art styles

### Lab 2.6.3: ControlNet Workshop (2 hours)
**Objective:** Add structural control to generation

Control image composition:
- Canny edge detection for outlines
- Depth maps for spatial composition
- Create custom control images
- Maintain character consistency

### Lab 2.6.4: Flux Exploration (2 hours)
**Objective:** Explore next-gen diffusion models

Compare architectures:
- Load Flux-schnell (fast) and Flux-dev (quality)
- Side-by-side comparison with SDXL
- Benchmark performance on DGX Spark
- Understand when to use each model

### Lab 2.6.5: LoRA Style Training (2 hours)
**Objective:** Train custom art styles

Create your own style:
- Prepare training datasets
- Configure and train SDXL LoRA
- Adjust LoRA strength
- Combine multiple LoRAs

### Lab 2.6.6: Image Generation Pipeline (2 hours)
**Objective:** Build a production-ready system

Complete pipeline development:
- Seed management for reproducibility
- Batch generation with variations
- Prompt templating system
- Metadata saving and loading
- Optional Gradio interface

---

## DGX Spark Performance

| Model | Resolution | Steps | Time | Memory |
|-------|------------|-------|------|--------|
| SDXL Base | 1024×1024 | 30 | ~5-8s | ~7GB |
| SDXL + Refiner | 1024×1024 | 50 | ~12-15s | ~14GB |
| Flux-schnell | 1024×1024 | 4 | ~3-4s | ~12GB |
| Flux-dev | 1024×1024 | 50 | ~15-20s | ~12GB |

---

## Quick Start

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load SDXL on DGX Spark
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,  # Native Blackwell support
    variant="fp16",
)
pipe = pipe.to("cuda")

# Generate
image = pipe(
    prompt="A majestic lion in the African savanna, golden hour, wildlife photography",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("lion.png")
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| SDXL OOM | Reduce resolution or use `enable_vae_tiling()` |
| ControlNet not following edges | Increase `controlnet_conditioning_scale` |
| Blurry generations | Increase steps (30+) or add quality tokens |
| LoRA not applying | Check path and trigger word |
| Flux slow inference | Use Flux-schnell (4 steps) for testing |

---

## Resources

- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [Stable Diffusion XL Paper](https://arxiv.org/abs/2307.01952)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [CivitAI](https://civitai.com/) - LoRA repository

---

## Next Steps

After completing this module, proceed to:
**Domain 3: LLM Systems** → Module 3.1: LLM Fine-Tuning
