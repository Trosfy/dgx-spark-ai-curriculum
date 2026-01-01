# Module 2.6: Diffusion Models - Quickstart

## Time: ~5 minutes

## What You'll Do
Generate an image with Stable Diffusion XL using 4 lines of code.

## Before You Start
- [ ] DGX Spark container running
- [ ] diffusers library installed

## Let's Go!

### Step 1: Load SDXL
```python
from diffusers import StableDiffusionXLPipeline
import torch

print("Loading Stable Diffusion XL (this may take a minute)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,  # Native Blackwell support
    variant="fp16"
)
pipe = pipe.to("cuda")
print("SDXL loaded!")
```

### Step 2: Generate an Image
```python
prompt = "A majestic lion in the African savanna at golden hour, wildlife photography, 8k"
negative_prompt = "blurry, low quality, distorted, deformed"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

print(f"Generated image size: {image.size}")
```

### Step 3: Save and Display
```python
# Save
image.save("lion.png")
print("Saved to lion.png")

# Display in Jupyter
from IPython.display import display
display(image)
```

### Step 4: Try Different Prompts
```python
prompts = [
    "A cozy cabin in snowy mountains, warm light from windows, winter scene",
    "Cyberpunk city street at night, neon lights, rain reflections",
    "Underwater coral reef with tropical fish, sunlight rays, crystal clear water"
]

for i, p in enumerate(prompts):
    img = pipe(p, negative_prompt=negative_prompt, num_inference_steps=25).images[0]
    img.save(f"generated_{i+1}.png")
    display(img)
```

## You Did It!

You just:
- Loaded SDXL on DGX Spark's 128GB unified memory
- Generated images from text prompts
- Used negative prompts to improve quality
- Created multiple images with different styles

In the full module, you'll learn:
- Diffusion theory (DDPM from scratch)
- ControlNet for guided generation
- Flux models (next-gen diffusion)
- LoRA training for custom styles
- Building generation pipelines

## Next Steps
1. **Understand the theory**: Start with [Lab 2.6.1](./labs/lab-2.6.1-diffusion-theory.ipynb)
2. **Add control**: Try ControlNet in [Lab 2.6.3](./labs/lab-2.6.3-controlnet.ipynb)
3. **Train custom style**: See [Lab 2.6.5](./labs/lab-2.6.5-lora-training.ipynb)
