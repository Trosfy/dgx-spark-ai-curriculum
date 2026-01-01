# Module 2.6: Diffusion Models - Troubleshooting Guide

## Quick Diagnostic

1. Check GPU memory: `nvidia-smi`
2. Verify diffusers version: `pip show diffusers`
3. Clear GPU memory: `torch.cuda.empty_cache()`
4. Check model downloads: `ls ~/.cache/huggingface/hub/`

---

## Loading Errors

### Error: SDXL taking too long to load

**Cause**: Large model downloading first time (~5 GB).

**Solution**:
```python
# Check download progress in another terminal
ls -la ~/.cache/huggingface/hub/

# Use streaming to see progress
from diffusers import StableDiffusionXLPipeline
import logging
logging.basicConfig(level=logging.INFO)

pipe = StableDiffusionXLPipeline.from_pretrained(...)
```

---

### Error: Flux model requires authentication

**Cause**: Flux models need license acceptance.

**Solution**:
```python
# 1. Accept license at huggingface.co/black-forest-labs/FLUX.1-schnell
# 2. Login
from huggingface_hub import login
login()

# 3. Load model
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
```

---

## Generation Errors

### Error: `CUDA out of memory`

**Solutions**:
```python
# Solution 1: Enable VAE tiling
pipe.enable_vae_tiling()

# Solution 2: Reduce resolution
image = pipe(prompt, height=768, width=768).images[0]

# Solution 3: Enable CPU offload
pipe.enable_model_cpu_offload()

# Solution 4: Clear memory before loading
import gc
import torch
del old_pipe  # Delete old pipeline
gc.collect()
torch.cuda.empty_cache()
```

---

### Error: Blurry or low-quality output

**Solutions**:
```python
# Solution 1: Increase inference steps
image = pipe(prompt, num_inference_steps=50).images[0]  # Instead of 20

# Solution 2: Add quality tokens to prompt
prompt = "A detailed photograph of ..., 8k, high quality, sharp focus"

# Solution 3: Use negative prompts
negative = "blurry, low quality, pixelated, compressed, artifacts"
image = pipe(prompt, negative_prompt=negative).images[0]

# Solution 4: Adjust guidance scale
image = pipe(prompt, guidance_scale=8.0).images[0]  # Try 7-9 range
```

---

### Error: Generated images don't match prompt

**Solutions**:
```python
# Solution 1: Increase guidance scale
image = pipe(prompt, guidance_scale=12.0).images[0]  # Stronger adherence

# Solution 2: Simplify prompt
# Instead of: "A beautiful majestic stunning gorgeous lion"
# Use: "A lion, wildlife photography, detailed"

# Solution 3: Check for conflicting terms
# Bad: "A bright, dark image of..."
# Good: "A brightly lit image of..."

# Solution 4: Use prompt weighting (if supported)
prompt = "(main subject:1.3), secondary details"
```

---

### Error: Oversaturated or unnatural colors

**Solution**:
```python
# Lower guidance scale
image = pipe(
    prompt,
    guidance_scale=6.0,  # Lower from 7.5
    negative_prompt="oversaturated, unnatural colors"
).images[0]
```

---

## ControlNet Errors

### Error: ControlNet not following control image

**Solutions**:
```python
# Solution 1: Increase conditioning scale
image = pipe(
    prompt,
    image=control_image,
    controlnet_conditioning_scale=0.8,  # Increase from 0.5
).images[0]

# Solution 2: Verify control image preprocessing
import cv2
import numpy as np
from PIL import Image

# Ensure proper edge detection
image_array = np.array(control_image)
edges = cv2.Canny(image_array, 100, 200)
control_image = Image.fromarray(edges)

# Solution 3: Match resolution
control_image = control_image.resize((1024, 1024))  # Match generation size
```

---

### Error: ControlNet edge detection not working

**Cause**: Wrong image format or grayscale issues.

**Solution**:
```python
import cv2
import numpy as np
from PIL import Image

# Ensure RGB input
input_image = Image.open("image.png").convert("RGB")
image_array = np.array(input_image)

# Apply Canny edge detection
edges = cv2.Canny(image_array, 100, 200)

# Convert to 3-channel for ControlNet
edges_rgb = np.stack([edges] * 3, axis=-1)
control_image = Image.fromarray(edges_rgb)
```

---

## LoRA Errors

### Error: LoRA not affecting generation

**Causes**: Wrong path, missing trigger word, or weight too low.

**Solutions**:
```python
# Solution 1: Verify LoRA loaded
pipe.load_lora_weights("./lora_model")
print("LoRA loaded successfully")

# Solution 2: Use trigger word
# If trained with trigger "my_style":
image = pipe("A photo in my_style style").images[0]

# Solution 3: Adjust LoRA scale
pipe.load_lora_weights("./lora_model", scale=1.0)  # Full strength
# Or
pipe.fuse_lora(lora_scale=0.8)
```

---

### Error: LoRA training loss not decreasing

**Solutions**:
```python
# Solution 1: Adjust learning rate
# For SDXL LoRA: 1e-4 to 1e-5 is typical
optimizer = torch.optim.AdamW(params, lr=1e-4)

# Solution 2: Check dataset
# Ensure images are properly preprocessed
# Resolution should match training resolution (1024×1024 for SDXL)

# Solution 3: Increase training steps
# 1000-5000 steps typical for style LoRAs

# Solution 4: Reduce rank if overfitting
config = LoraConfig(r=8)  # Instead of r=16
```

---

### Error: `ValueError: LoRA already loaded`

**Solution**:
```python
# Unload existing LoRA first
pipe.unload_lora_weights()

# Then load new one
pipe.load_lora_weights("./new_lora")
```

---

## Flux-Specific Errors

### Error: Flux generation very slow

**Cause**: Using Flux-dev instead of Flux-schnell.

**Solution**:
```python
# Use Flux-schnell for speed (4 steps)
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda")

# Only needs 4 steps!
image = pipe(prompt, num_inference_steps=4).images[0]
```

---

### Error: Flux output different from SDXL

**Explanation**: Flux is a different architecture with different characteristics.

**Tips**:
```python
# Flux often needs different prompting
# More direct, less elaborate prompts work better

# SDXL style:
"A majestic lion in the African savanna, golden hour light, wildlife photography, 8k uhd"

# Flux style:
"A lion in the African savanna at sunset"
```

---

## Reset Procedures

### Memory Reset

```python
import gc
import torch

# Delete pipeline
del pipe

# Clear cache
gc.collect()
torch.cuda.empty_cache()

# Verify
print(f"Memory freed: {torch.cuda.memory_allocated()/1e9:.2f} GB used")
```

### Clear Model Cache

```bash
# Remove specific model cache
rm -rf ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0

# Model will re-download on next load
```

---

## Still Stuck?

1. **Check Diffusers documentation** - Most issues documented there
2. **Print pipeline info** - `print(pipe)` shows structure
3. **Try minimal example** - Simplify prompt, use defaults
4. **Check solution notebooks** - in `solutions/` folder

**Debug template:**
```python
def debug_generation(pipe, prompt):
    print(f"Pipeline: {type(pipe)}")
    print(f"Device: {pipe.device}")
    print(f"Prompt: {prompt[:50]}...")

    # Try minimal generation
    image = pipe(
        prompt,
        num_inference_steps=10,  # Fast test
        height=512,  # Small size
        width=512
    ).images[0]

    print(f"Generated: {image.size}")
    return image
```
