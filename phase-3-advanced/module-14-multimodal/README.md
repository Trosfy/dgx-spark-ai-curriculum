# Module 14: Multimodal AI

**Phase:** 3 - Advanced  
**Duration:** Week 24 (8-10 hours)  
**Prerequisites:** Module 13 (AI Agents)

---

## Overview

Modern AI is multimodal. This module covers vision-language models, image generation, audio processing, and building pipelines that combine multiple modalities. DGX Spark's 128GB memory makes running large multimodal models locally practical.

---

## Learning Outcomes

By the end of this module, you will be able to:

- ✅ Work with vision-language models for image understanding
- ✅ Implement image generation with diffusion models
- ✅ Build multimodal pipelines combining vision and language
- ✅ Fine-tune multimodal models

---

## Learning Objectives

| ID | Objective | Bloom's Level |
|----|-----------|---------------|
| 14.1 | Use vision-language models for image analysis | Apply |
| 14.2 | Generate images using Stable Diffusion | Apply |
| 14.3 | Build multimodal RAG systems | Apply |
| 14.4 | Fine-tune vision-language models | Apply |

---

## Topics

### 14.1 Vision-Language Models
- LLaVA architecture
- CLIP embeddings
- Qwen-VL, InternVL
- Document understanding (OCR, layout)

### 14.2 Image Generation
- Stable Diffusion fundamentals
- ControlNet
- SDXL, Flux
- LoRA for style

### 14.3 Audio Models
- Whisper transcription
- Text-to-speech
- Audio understanding

### 14.4 Multimodal Pipelines
- Document AI
- Video understanding
- Multimodal RAG

---

## Tasks

| # | Task | Time | Deliverable |
|---|------|------|-------------|
| 14.1 | Vision-Language Demo | 2h | LLaVA/Qwen-VL for image understanding |
| 14.2 | Image Generation | 2h | SDXL/Flux with ControlNet |
| 14.3 | Multimodal RAG | 2h | Index images + text, query with NL |
| 14.4 | Document AI Pipeline | 2h | PDF → OCR → QA |
| 14.5 | Audio Transcription | 2h | Whisper + LLM for audio Q&A |

---

## Guidance

### VLM on DGX Spark

```python
# LLaVA-34B fits easily in 128GB
# Qwen2-VL-72B works with quantization

from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-34b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Image Generation Performance

- SDXL: ~5-8 seconds for 1024×1024
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

- [LLaVA](https://llava-vl.github.io/)
- [Stable Diffusion](https://stability.ai/)
- [Whisper](https://github.com/openai/whisper)
- [CLIP](https://openai.com/research/clip)
