# Module 4.1: Multimodal AI

**Domain:** 3 - LLM Systems
**Duration:** Week 24 (8-10 hours)
**Prerequisites:** Module 3.4 (AI Agents)

---

## Overview

Modern AI is multimodal - it can see, read, hear, and create across different types of data. This module covers vision-language models, image generation, audio processing, and building pipelines that combine multiple modalities. DGX Spark's 128GB unified memory makes running large multimodal models locally not just possible, but *practical*!

### Why Multimodal AI Matters

Think about how you experience the world - you don't just read text OR look at images OR listen to sounds. You do all of these simultaneously, and your brain seamlessly combines this information. Multimodal AI aims to give machines this same capability.

**Real-world applications:**
- **Healthcare**: Analyzing X-rays while reading patient notes
- **E-commerce**: Understanding product images and descriptions together
- **Accessibility**: Describing images for visually impaired users
- **Creative**: Generating images from text descriptions
- **Documentation**: Extracting information from complex PDFs with tables and figures

---

## Learning Outcomes

By the end of this module, you will be able to:

- Use vision-language models to understand and describe images
- Generate images using state-of-the-art diffusion models
- Build multimodal RAG systems that can search across images and text
- Create document AI pipelines for processing complex PDFs
- Transcribe and analyze audio using Whisper

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 4.1.1 | Use vision-language models for image analysis | Apply |
| 4.1.2 | Generate images using Stable Diffusion | Apply |
| 4.1.3 | Build multimodal RAG systems | Apply |
| 4.1.4 | Process documents with OCR and layout analysis | Apply |
| 4.1.5 | Transcribe and analyze audio content | Apply |

---

## DGX Spark Advantage

| Model | VRAM Required | Fits on DGX Spark? |
|-------|---------------|-------------------|
| LLaVA-1.5-7B | ~16GB | Easily |
| LLaVA-1.5-13B | ~28GB | Yes |
| Qwen2-VL-7B | ~18GB | Easily |
| Qwen2-VL-72B (4-bit) | ~45GB | Yes |
| SDXL | ~8GB | Easily |
| Flux.1-dev | ~24GB | Yes |
| Whisper-large-v3 | ~4GB | Easily |
| CLIP-ViT-L/14 | ~2GB | Easily |

The 128GB unified memory means you can run multiple models simultaneously - for example, running CLIP for image search, an LLM for answering questions, and keeping embeddings in memory.

---

## Topics

### 4.1.1 Vision-Language Models
- LLaVA architecture and usage
- CLIP embeddings for image-text alignment
- Qwen-VL for advanced visual understanding
- Document understanding (OCR, layout)

### 4.1.2 Image Generation
- Stable Diffusion fundamentals
- SDXL for high-quality generation
- ControlNet for guided generation
- Flux for state-of-the-art results

### 4.1.3 Audio Models
- Whisper transcription
- Audio preprocessing
- Speech-to-text pipelines

### 4.1.4 Multimodal Pipelines
- Document AI (PDF processing)
- Multimodal RAG
- Combining vision, language, and audio

---

## Labs

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 4.1.1 | Vision-Language Demo | 2h | LLaVA/Qwen-VL for image understanding |
| 4.1.2 | Image Generation | 2h | SDXL/Flux with ControlNet |
| 4.1.3 | Multimodal RAG | 2h | Index images + text, query with NL |
| 4.1.4 | Document AI Pipeline | 2h | PDF to OCR to QA |
| 4.1.5 | Audio Transcription | 2h | Whisper + LLM for audio Q&A |

---

## Directory Structure

```
module-4.1-multimodal/
├── README.md                           # This file
├── labs/
│   ├── 01-vision-language-demo.ipynb   # VLM introduction
│   ├── 02-image-generation.ipynb       # Diffusion models
│   ├── 03-multimodal-rag.ipynb         # CLIP + vector search
│   ├── 04-document-ai-pipeline.ipynb   # PDF processing
│   └── 05-audio-transcription.ipynb    # Whisper integration
├── scripts/
│   ├── __init__.py
│   ├── vlm_utils.py                    # Vision-language utilities
│   ├── image_generation.py             # Image gen helpers
│   ├── multimodal_rag.py               # Multimodal RAG system
│   ├── document_ai.py                  # Document processing
│   └── audio_utils.py                  # Audio transcription
├── solutions/
│   ├── 01-vision-language-demo-solution.ipynb
│   ├── 02-image-generation-solution.ipynb
│   ├── 03-multimodal-rag-solution.ipynb
│   ├── 04-document-ai-pipeline-solution.ipynb
│   └── 05-audio-transcription-solution.ipynb
├── pipelines/                          # End-to-end pipelines
└── data/
    └── README.md                       # Data documentation
```

---

## Setup

### NGC Container (Required)

```bash
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

### Additional Dependencies

```bash
# Inside the container (versions pinned for compatibility)
pip install transformers>=4.45.0 accelerate>=0.27.0 bitsandbytes>=0.42.0
pip install diffusers>=0.27.0 controlnet_aux>=0.0.7
pip install sentence-transformers>=2.3.0 chromadb>=0.4.22
pip install pdf2image pytesseract pymupdf>=1.23.0
pip install openai-whisper soundfile librosa scipy
pip install pillow>=10.0.0 opencv-python qwen-vl-utils matplotlib
```

---

## Guidance

### VLM on DGX Spark

```python
# LLaVA-13B fits easily in 128GB
# Qwen2-VL-72B works with quantization

from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Image Generation Performance

- SDXL: ~5-8 seconds for 1024x1024
- Flux: ~15-20 seconds (higher quality)

### Multimodal RAG with CLIP

```python
from transformers import CLIPModel, CLIPProcessor

# Embed images with CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Store in vector DB alongside text embeddings
# Query with natural language
```

---

## Milestone Checklist

- [ ] Vision-language model demo complete
- [ ] Image generation with various techniques
- [ ] Multimodal RAG indexing images + text
- [ ] Document AI pipeline processing PDFs
- [ ] Audio transcription and Q&A pipeline

---

## Resources

- [LLaVA Project](https://llava-vl.github.io/)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [Stable Diffusion](https://stability.ai/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Whisper](https://github.com/openai/whisper)
- [Diffusers Library](https://huggingface.co/docs/diffusers)
