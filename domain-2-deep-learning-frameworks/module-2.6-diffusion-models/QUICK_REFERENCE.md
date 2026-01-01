# Module 2.6: Diffusion Models - Quick Reference

## Essential Commands

### Load SDXL

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    variant="fp16"
)
pipe = pipe.to("cuda")
```

### Load Flux

```python
from diffusers import FluxPipeline

# Flux-schnell (fast, 4 steps)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

# Flux-dev (quality, 50 steps)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
```

### Load ControlNet

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.bfloat16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
```

---

## Key Patterns

### Pattern: Basic Generation

```python
image = pipe(
    prompt="A majestic lion in the savanna, golden hour, wildlife photography",
    negative_prompt="blurry, low quality, distorted, deformed",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024,
).images[0]

image.save("output.png")
```

### Pattern: Reproducible Generation (Seeds)

```python
import torch

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt="Your prompt here",
    generator=generator,
    num_inference_steps=30,
).images[0]

# Same seed = same image (with same settings)
```

### Pattern: Batch Generation

```python
prompts = [
    "A sunset over mountains",
    "A cat playing piano",
    "A futuristic city"
]

images = pipe(
    prompt=prompts,
    num_inference_steps=30,
    guidance_scale=7.5,
).images

for i, img in enumerate(images):
    img.save(f"batch_{i}.png")
```

### Pattern: ControlNet with Canny Edges

```python
import cv2
import numpy as np
from PIL import Image

# Create canny edge image
input_image = Image.open("input.png")
image_array = np.array(input_image)
edges = cv2.Canny(image_array, 100, 200)
edges = np.stack([edges] * 3, axis=-1)  # Convert to RGB
control_image = Image.fromarray(edges)

# Generate with control
image = pipe(
    prompt="A beautiful landscape, detailed, 4k",
    image=control_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=30,
).images[0]
```

### Pattern: Load and Apply LoRA

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load LoRA
pipe.load_lora_weights("path/to/lora", weight_name="lora.safetensors")

# Generate with trigger word
image = pipe(
    "A photo in my_style style",  # Use LoRA's trigger word
    num_inference_steps=30,
).images[0]

# Unload LoRA
pipe.unload_lora_weights()
```

### Pattern: Memory Optimization

```python
# Enable memory-efficient attention
pipe.enable_xformers_memory_efficient_attention()

# Enable VAE tiling for large images
pipe.enable_vae_tiling()

# Enable CPU offload (saves GPU memory)
pipe.enable_model_cpu_offload()

# Disable for maximum speed
pipe.disable_xformers_memory_efficient_attention()
```

### Pattern: DDPM Scheduler Comparison

```python
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler
)

# Faster schedulers (fewer steps needed)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=25).images[0]

# Or Euler (good quality/speed balance)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
image = pipe(prompt, num_inference_steps=30).images[0]
```

---

## Key Values

| Parameter | Recommended Range | Notes |
|-----------|-------------------|-------|
| `guidance_scale` | 7.0 - 8.5 | Higher = more prompt adherent |
| `num_inference_steps` | 25-50 | More = higher quality, slower |
| `controlnet_conditioning_scale` | 0.3 - 0.8 | Higher = stricter control |
| `lora_scale` | 0.5 - 1.0 | LoRA strength |

---

## Prompt Engineering Tips

### Quality Boosters
```
"8k uhd, high quality, detailed, professional"
"masterpiece, best quality, highly detailed"
"sharp focus, intricate details"
```

### Photography Style
```
"DSLR photo, 85mm lens, shallow depth of field"
"cinematic lighting, golden hour, professional photography"
```

### Art Style
```
"oil painting, impressionist style, textured brushstrokes"
"digital art, concept art, trending on artstation"
"watercolor painting, soft edges, flowing colors"
```

### Common Negative Prompts
```
"blurry, low quality, pixelated, jpeg artifacts"
"deformed, bad anatomy, extra limbs, mutated"
"oversaturated, overexposed, ugly"
"text, watermark, signature, logo"
```

---

## DGX Spark Performance

| Model | Resolution | Steps | Time | Memory |
|-------|------------|-------|------|--------|
| SDXL Base | 1024×1024 | 30 | ~5-8s | ~7 GB |
| SDXL + Refiner | 1024×1024 | 50 | ~12-15s | ~14 GB |
| Flux-schnell | 1024×1024 | 4 | ~3-4s | ~12 GB |
| Flux-dev | 1024×1024 | 50 | ~15-20s | ~12 GB |
| SDXL + ControlNet | 1024×1024 | 30 | ~8-10s | ~10 GB |

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Blurry output | Increase steps, add quality tokens |
| ControlNet ignored | Increase `controlnet_conditioning_scale` |
| OOM errors | Use `enable_vae_tiling()`, reduce resolution |
| LoRA not working | Check trigger word, verify path |
| Inconsistent results | Set seed with `Generator` |
| Oversaturated | Lower `guidance_scale` to 6-7 |

---

## Quick Links

- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [CivitAI](https://civitai.com/) - LoRA models
