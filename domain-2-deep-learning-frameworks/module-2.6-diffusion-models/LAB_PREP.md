# Module 2.6: Diffusion Models - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 2.6.1 Diffusion Theory | 10 min | 2 hr | ~2 hr |
| 2.6.2 Stable Diffusion | 15 min | 2 hr | ~2.5 hr |
| 2.6.3 ControlNet | 15 min | 2 hr | ~2.5 hr |
| 2.6.4 Flux Exploration | 15 min | 2 hr | ~2.5 hr |
| 2.6.5 LoRA Training | 20 min | 2 hr | ~2.5 hr |
| 2.6.6 Generation Pipeline | 10 min | 2 hr | ~2 hr |

**Total**: ~10-12 hours

---

## Required Downloads

### Models (Large Downloads)

```python
# SDXL Base (~5 GB) - Labs 2.6.2, 2.6.3, 2.6.5, 2.6.6
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)

# ControlNet for SDXL (~2.5 GB) - Lab 2.6.3
from diffusers import ControlNetModel
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0"
)

# Flux-schnell (~12 GB) - Lab 2.6.4
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
```

### Packages

```bash
pip install diffusers accelerate safetensors opencv-python
```

---

## Environment Setup

### 1. Start NGC Container

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### 2. Install Packages

```bash
pip install diffusers accelerate safetensors opencv-python peft --quiet
```

### 3. Verify Setup

```python
import torch
from diffusers import StableDiffusionXLPipeline

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Quick generation test
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16,
    variant="fp16"
).to("cuda")

image = pipe("A test image", num_inference_steps=10).images[0]
print(f"Generated: {image.size}")
```

---

## Pre-Lab Checklist

### Lab 2.6.1: Diffusion Theory
- [ ] Container running with GPU
- [ ] matplotlib for visualization
- [ ] MNIST dataset (auto-downloads)

### Lab 2.6.2: Stable Diffusion
- [ ] SDXL downloaded (~5 GB)
- [ ] Understand prompting basics

### Lab 2.6.3: ControlNet
- [ ] SDXL ControlNet downloaded
- [ ] opencv-python installed
- [ ] Sample images for edge detection

### Lab 2.6.4: Flux Exploration
- [ ] Flux-schnell downloaded (~12 GB)
- [ ] HuggingFace login (Flux requires acceptance)

### Lab 2.6.5: LoRA Training
- [ ] peft library installed
- [ ] Training images prepared (10-20 images)
- [ ] Completed Lab 2.6.2

### Lab 2.6.6: Generation Pipeline
- [ ] Completed Labs 2.6.2-2.6.3
- [ ] Understand seeds and reproducibility

---

## Model Memory Requirements

| Model | BF16 Memory | Notes |
|-------|------------|-------|
| SDXL Base | ~7 GB | Main generation model |
| SDXL + Refiner | ~14 GB | Two-stage generation |
| SDXL + ControlNet | ~10 GB | With one ControlNet |
| Flux-schnell | ~12 GB | Fast 4-step generation |
| Flux-dev | ~12 GB | High-quality 50-step |
| LoRA Training | ~16 GB | With gradient checkpointing |

All models fit comfortably in DGX Spark's 128GB.

---

## Quick Start Commands

```bash
# Inside NGC container
cd /workspace/domain-2-deep-learning-frameworks/module-2.6-diffusion-models

# Install dependencies
pip install diffusers accelerate safetensors opencv-python peft --quiet

# Pre-download SDXL (optional but recommended)
python -c "
from diffusers import StableDiffusionXLPipeline

print('Downloading SDXL (this may take a while)...')
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    variant='fp16'
)
print('SDXL downloaded!')
"
```

---

## Expected File Structure

```
/workspace/
├── domain-2-deep-learning-frameworks/
│   └── module-2.6-diffusion-models/
│       ├── README.md
│       ├── QUICKSTART.md
│       ├── STUDY_GUIDE.md
│       ├── QUICK_REFERENCE.md
│       ├── ELI5.md
│       ├── LAB_PREP.md
│       ├── TROUBLESHOOTING.md
│       ├── labs/
│       │   ├── lab-2.6.1-diffusion-theory.ipynb
│       │   ├── lab-2.6.2-stable-diffusion.ipynb
│       │   ├── lab-2.6.3-controlnet.ipynb
│       │   ├── lab-2.6.4-flux-exploration.ipynb
│       │   ├── lab-2.6.5-lora-training.ipynb
│       │   └── lab-2.6.6-generation-pipeline.ipynb
│       ├── scripts/
│       ├── data/
│       │   └── sample_prompts.txt
│       └── solutions/
```

---

## Accessing Flux Models

Flux models require accepting a license on HuggingFace:

1. Go to [huggingface.co/black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
2. Accept the license agreement
3. Login to HuggingFace in your notebook:

```python
from huggingface_hub import login
login()  # Enter your token
```

---

## DGX Spark Advantages

| Feature | Benefit |
|---------|---------|
| 128GB Memory | SDXL + Refiner simultaneously |
| 128GB Memory | Multiple ControlNets loaded |
| BF16 Native | Faster generation, native precision |
| Fast Memory | Train LoRAs at full precision |
