# Module 4.1: Multimodal AI - Lab Preparation Guide

## Time Estimates

| Lab | Preparation | Execution | Total |
|-----|-------------|-----------|-------|
| 4.1.1 Vision-Language | 20 min | 2h | 2.5h |
| 4.1.2 Image Generation | 15 min | 2h | 2.25h |
| 4.1.3 Multimodal RAG | 15 min | 2h | 2.25h |
| 4.1.4 Document AI | 20 min | 2h | 2.5h |
| 4.1.5 Audio Transcription | 10 min | 2h | 2.2h |

---

## Required Downloads

### Models (Download Before Labs)

```bash
# Lab 4.1.1: Vision-Language Models
# LLaVA-7B (~14GB) - Required
huggingface-cli download llava-hf/llava-1.5-7b-hf

# LLaVA-13B (~26GB) - Optional, better quality
huggingface-cli download llava-hf/llava-1.5-13b-hf

# Qwen2-VL-7B (~18GB) - Alternative VLM
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct
```

```bash
# Lab 4.1.2: Image Generation
# SDXL (~6.5GB) - Required
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0

# ControlNet (~2.5GB) - Optional
huggingface-cli download diffusers/controlnet-canny-sdxl-1.0
```

```bash
# Lab 4.1.3: Multimodal RAG
# CLIP model (~1.5GB)
huggingface-cli download openai/clip-vit-large-patch14

# Embedding model for text (~500MB)
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
```

```bash
# Lab 4.1.5: Audio Transcription
# Whisper large-v3 (~3GB)
# Downloads automatically on first use
```

**Total download size**: ~50GB (all optional models) or ~25GB (required only)
**Estimated download time**: 30-60 minutes on fast connection

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

### 2. Install All Dependencies

```bash
# Inside the container - run this once
pip install transformers>=4.46.0 accelerate>=0.28.0 bitsandbytes>=0.43.0
pip install diffusers>=0.27.0 controlnet_aux>=0.0.7
pip install sentence-transformers>=2.3.0 chromadb>=0.4.22
pip install pdf2image pytesseract pymupdf>=1.23.0
pip install openai-whisper soundfile librosa scipy
pip install pillow>=10.0.0 opencv-python qwen-vl-utils matplotlib
```

### 3. Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

**Expected output**:
```
CUDA available: True
Device: NVIDIA GH200 480GB  # or similar Blackwell
Memory: 128.0 GB
```

### 4. Clear Memory (Fresh Start)

```python
import torch, gc
torch.cuda.empty_cache()
gc.collect()
print(f"Free memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")
```

---

## Pre-Lab Checklists

### Lab 4.1.1: Vision-Language Demo

- [ ] LLaVA-7B model downloaded
- [ ] At least 20GB GPU memory free
- [ ] Sample images prepared (or will download)
- [ ] Reviewed concepts: Transformers, attention mechanism

**Quick Test**:
```python
from transformers import LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("LLaVA loaded successfully!")
del model; torch.cuda.empty_cache()
```

---

### Lab 4.1.2: Image Generation

- [ ] SDXL base model downloaded
- [ ] At least 10GB GPU memory free
- [ ] ControlNet downloaded (optional)
- [ ] Reviewed concepts: Diffusion models, denoising

**Quick Test**:
```python
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
)
print("SDXL loaded successfully!")
del pipe; torch.cuda.empty_cache()
```

---

### Lab 4.1.3: Multimodal RAG

- [ ] CLIP model downloaded
- [ ] ChromaDB installed
- [ ] Sample image dataset prepared (or will download)
- [ ] Reviewed concepts: Embeddings, vector search, RAG

**Quick Test**:
```python
import chromadb
from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
client = chromadb.Client()
print("CLIP and ChromaDB ready!")
```

---

### Lab 4.1.4: Document AI Pipeline

- [ ] PyMuPDF installed
- [ ] Tesseract OCR installed (for advanced OCR)
- [ ] Sample PDFs prepared (or will download)
- [ ] VLM from Lab 4.1.1 available

**Quick Test**:
```python
import fitz  # pymupdf
import pytesseract
print(f"PyMuPDF version: {fitz.version}")
print(f"Tesseract available: {pytesseract.get_tesseract_version()}")
```

**Note**: If Tesseract is not installed:
```bash
apt-get update && apt-get install -y tesseract-ocr
```

---

### Lab 4.1.5: Audio Transcription

- [ ] FFmpeg installed (for audio processing)
- [ ] Whisper will download on first use (~3GB)
- [ ] Sample audio files prepared (or will download)
- [ ] At least 6GB GPU memory free

**Quick Test**:
```python
import whisper
model = whisper.load_model("base")  # Small test model
print("Whisper ready!")
del model; torch.cuda.empty_cache()
```

**Note**: If FFmpeg is not installed:
```bash
apt-get update && apt-get install -y ffmpeg
```

---

## Sample Data

### Images for VLM Testing

```python
import requests
from PIL import Image
import os

os.makedirs("/workspace/sample_images", exist_ok=True)

# Sample images
urls = {
    "cat": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "landscape": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Altja_j%C3%B5gi_Lahemaal.jpg/1200px-Altja_j%C3%B5gi_Lahemaal.jpg",
    "document": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Obama_signs_the_FDA_Food_Safety_Modernization_Act_%28cropped%29.jpg/800px-President_Obama_signs_the_FDA_Food_Safety_Modernization_Act_%28cropped%29.jpg"
}

for name, url in urls.items():
    img = Image.open(requests.get(url, stream=True).raw)
    img.save(f"/workspace/sample_images/{name}.jpg")
    print(f"Saved {name}.jpg")
```

### Audio for Whisper Testing

```python
# Download sample audio (requires internet)
import urllib.request
os.makedirs("/workspace/sample_audio", exist_ok=True)

# LibriSpeech sample
url = "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"
urllib.request.urlretrieve(url, "/workspace/sample_audio/sample.wav")
print("Sample audio downloaded!")
```

---

## Common Setup Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Not mounting HuggingFace cache | Downloads models every time | Use `-v $HOME/.cache/huggingface:/root/.cache/huggingface` |
| Forgetting `--ipc=host` | DataLoader crashes | Always include in docker run |
| Not clearing GPU memory | OOM on second model | Call `torch.cuda.empty_cache()` between models |
| Using wrong dtype | OOM or slow | Use `torch.bfloat16` on DGX Spark |
| Missing system packages | Tesseract/FFmpeg errors | Install with apt-get first |

---

## Expected Directory Structure

After preparation, your workspace should look like:

```
/workspace/
├── module-4.1/
│   ├── sample_images/
│   │   ├── cat.jpg
│   │   ├── landscape.jpg
│   │   └── document.jpg
│   ├── sample_audio/
│   │   └── sample.wav
│   ├── sample_pdfs/
│   │   └── sample.pdf
│   └── outputs/
│       └── (generated during labs)
```

---

## Quick Start Commands

```bash
# Copy-paste this block to set up everything:
cd /workspace
mkdir -p module-4.1/{sample_images,sample_audio,sample_pdfs,outputs}

# Install all dependencies
pip install transformers>=4.46.0 accelerate>=0.28.0 bitsandbytes>=0.43.0 \
    diffusers>=0.27.0 controlnet_aux>=0.0.7 \
    sentence-transformers>=2.3.0 chromadb>=0.4.22 \
    pdf2image pytesseract pymupdf>=1.23.0 \
    openai-whisper soundfile librosa scipy \
    pillow>=10.0.0 opencv-python qwen-vl-utils matplotlib

# Install system dependencies
apt-get update && apt-get install -y tesseract-ocr ffmpeg

# Verify setup
python -c "import torch; print('GPU Ready!' if torch.cuda.is_available() else 'No GPU')"
```
