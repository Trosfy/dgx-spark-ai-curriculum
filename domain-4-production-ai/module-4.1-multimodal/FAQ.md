# Module 4.1: Multimodal AI - Frequently Asked Questions

## Setup & Environment

### Q: Which VLM should I use - LLaVA or Qwen-VL?

**A**: For most cases, start with **LLaVA-1.5-7B** (good quality, easy to use). Use **Qwen2-VL** if you need better instruction following or multilingual support. LLaVA-13B offers better quality if you have memory to spare.

| Model | Best For |
|-------|----------|
| LLaVA-7B | Learning, quick experiments |
| LLaVA-13B | Better quality, general use |
| Qwen2-VL-7B | Instruction following, multilingual |

---

### Q: How much GPU memory do I need for each lab?

**A**: With DGX Spark's 128GB unified memory, you have plenty for any single model. Here's what each requires:

| Lab | Minimum Memory | Comfortable |
|-----|----------------|-------------|
| 4.1.1 VLM | 16GB | 30GB |
| 4.1.2 Image Gen | 8GB | 15GB |
| 4.1.3 Multimodal RAG | 4GB | 10GB |
| 4.1.4 Document AI | 16GB | 30GB |
| 4.1.5 Audio | 4GB | 8GB |

**Important**: Clear memory between labs with `torch.cuda.empty_cache()`.

---

### Q: Can I run multiple models simultaneously?

**A**: Yes! This is a DGX Spark advantage. With 128GB, you can run CLIP for search + an LLM for response + keep embeddings in memory. Just be mindful of total usage.

```python
# Check available memory
free = (torch.cuda.get_device_properties(0).total_memory -
        torch.cuda.memory_allocated()) / 1e9
print(f"Available: {free:.1f} GB")
```

---

## Concepts

### Q: What's the difference between a VLM and just using CLIP + LLM separately?

**A**:

- **CLIP + LLM**: Two separate models. CLIP describes the image in embedding space, LLM generates text. They don't truly "understand" each other.

- **VLM (LLaVA, Qwen-VL)**: Single unified model trained end-to-end. Image and text are processed together, so the model can reason about visual details while generating text.

**Result**: VLMs give more coherent, detailed responses about images.

---

### Q: Why use CLIP if VLMs are better?

**A**: Different use cases:

| CLIP | VLMs |
|------|------|
| Fast embedding (~50ms) | Slow generation (~3s) |
| Good for search/retrieval | Good for understanding/description |
| Low memory (~2GB) | Higher memory (16GB+) |
| No generation | Full text generation |

**Best practice**: Use CLIP to find relevant images, then VLM to analyze the top results.

---

### Q: What's the difference between SDXL and Flux?

**A**:

- **SDXL**: Faster (~5-8s), good quality, well-supported ecosystem, more ControlNet options
- **Flux**: Slower (~15-20s), higher quality, better prompt following, newer architecture

**Start with SDXL** for learning. Use Flux when quality matters more than speed.

---

### Q: How does multimodal RAG work differently from text RAG?

**A**:

**Text RAG**:
1. Split documents into chunks
2. Embed chunks with text embedder
3. Query with text, retrieve text chunks

**Multimodal RAG**:
1. Process images and text
2. Embed images with CLIP, text with text embedder
3. **Cross-modal query**: Text query can retrieve images, image query can retrieve text
4. VLM can describe retrieved images before passing to LLM

---

### Q: What does "guidance scale" mean in image generation?

**A**: Guidance scale controls how closely the model follows your prompt:

- **Low (1-5)**: More creative/random, less prompt adherence
- **Medium (7-9)**: Good balance (recommended)
- **High (10+)**: Very strict prompt following, can look artificial

**Default**: Use 7.5 for SDXL, 3.5 for Flux.

---

## Troubleshooting

### Q: My VLM gives generic responses like "I see an image"

**A**: This usually means the image isn't being processed correctly. Check:

1. Image is in correct format (PIL Image or tensor)
2. Prompt includes the `<image>` token in the right place
3. Image is passed to processor correctly

```python
# Correct LLaVA format
prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")
```

---

### Q: SDXL generates weird artifacts or distortions

**A**: Common causes:

1. **Too few steps**: Increase to 30-50
2. **Wrong dtype**: Use `bfloat16`, not `float32`
3. **Bad prompt**: Avoid conflicting descriptions
4. **Seed issue**: Try different seeds

```python
# Good settings
image = pipe(
    prompt="A serene landscape, photorealistic, 8k",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=40,
    guidance_scale=7.5
).images[0]
```

---

### Q: Whisper transcription is slow

**A**: Whisper speed depends on model size:

| Model | Speed | Quality |
|-------|-------|---------|
| tiny | 32x real-time | Low |
| base | 16x real-time | OK |
| small | 6x real-time | Good |
| medium | 2x real-time | Better |
| large-v3 | 0.5x real-time | Best |

**Tip**: Use `base` or `small` for development, `large-v3` for final output.

---

### Q: My multimodal RAG returns wrong images

**A**: Check:

1. **Embeddings normalized**: CLIP embeddings must be L2-normalized
2. **Same embedding model for index and query**
3. **Sufficient images indexed**: Need variety for good retrieval

```python
# Always normalize
embedding = F.normalize(embedding, dim=-1)
```

---

## Beyond the Basics

### Q: Can I use these models for production applications?

**A**: Yes, but consider:

- **LLaVA/Qwen-VL**: Open weights, check licenses for commercial use
- **SDXL**: Stability AI license (mostly permissive)
- **Flux**: Black Forest Labs license (check restrictions)
- **Whisper**: MIT license (very permissive)

**Important**: Add safety filters for user-facing applications (Module 4.2).

---

### Q: How do I make VLMs faster for real-time applications?

**A**: Options:

1. **Quantization**: 4-bit reduces latency ~2x
2. **Smaller models**: 7B vs 13B
3. **Batching**: Process multiple images at once
4. **vLLM**: Optimized serving (covered in Module 3.3)

---

### Q: Can I fine-tune these multimodal models?

**A**: Yes, but it's more complex:

- **VLMs**: Can fine-tune with LoRA on instruction data
- **SDXL**: Fine-tune with DreamBooth or LoRA for specific styles
- **Whisper**: Can fine-tune for domain-specific audio

This curriculum focuses on using pre-trained models. Fine-tuning multimodal models is an advanced topic.

---

### Q: How does document AI compare to just using OCR?

**A**:

| OCR Only | Document AI |
|----------|-------------|
| Extracts raw text | Understands structure |
| Loses layout | Preserves tables, headers |
| Can't read charts | VLM describes charts |
| Fast | Slower |

**Document AI = OCR + Layout Analysis + VLM + LLM**

---

## Still Have Questions?

- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for error-specific help
- Review [ELI5.md](./ELI5.md) for concept clarification
- Review [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) for code patterns
- See module [Resources](./README.md#resources) for official documentation
