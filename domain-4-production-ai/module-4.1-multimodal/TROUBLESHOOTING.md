# Module 4.1: Multimodal AI - Troubleshooting Guide

## Quick Diagnostic

**Before diving into specific errors, try these:**

1. Check GPU memory: `nvidia-smi` or `torch.cuda.memory_summary()`
2. Clear cache: `torch.cuda.empty_cache(); gc.collect()`
3. Restart kernel/container
4. Verify you're in the correct directory

---

## Memory Errors

### Error: `CUDA out of memory` with VLM

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

**Causes**:
1. Model too large for available memory
2. Previous model still loaded
3. Large images not resized

**Solutions**:

```python
# Solution 1: Clear all previous models
import torch, gc
torch.cuda.empty_cache()
gc.collect()

# Solution 2: Use 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-13b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Solution 3: Resize large images
from PIL import Image

def resize_for_vlm(image_path, max_size=512):
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    return img
```

**Prevention**: Always clear memory before loading new models.

---

### Error: `CUDA out of memory` with SDXL

**Symptoms**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:

```python
# Solution 1: Enable CPU offload
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

# Solution 2: Enable attention slicing
pipe.enable_attention_slicing()

# Solution 3: Reduce image size
image = pipe(
    prompt="...",
    height=768,  # Instead of 1024
    width=768
).images[0]

# Solution 4: Use sequential CPU offload (slower but lower memory)
pipe.enable_sequential_cpu_offload()
```

---

## Import/Dependency Errors

### Error: `ModuleNotFoundError: No module named 'transformers'`

**Solutions**:
```bash
pip install transformers>=4.46.0 accelerate>=0.28.0
```

---

### Error: `No module named 'qwen_vl_utils'`

**Cause**: Qwen-VL requires special utilities package.

**Solution**:
```bash
pip install qwen-vl-utils
```

---

### Error: `ImportError: libGL.so.1: cannot open shared object file`

**Cause**: OpenCV missing system dependency.

**Solution**:
```bash
apt-get update && apt-get install -y libgl1-mesa-glx
# Or use headless OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

---

### Error: `tesseract is not installed or it's not in your PATH`

**Solution**:
```bash
apt-get update && apt-get install -y tesseract-ocr
# Verify
tesseract --version
```

---

### Error: `ffmpeg not found` (Whisper)

**Solution**:
```bash
apt-get update && apt-get install -y ffmpeg
# Verify
ffmpeg -version
```

---

## Model Loading Errors

### Error: `ValueError: Tokenizer class LlamaTokenizer does not exist`

**Cause**: Using wrong tokenizer or outdated transformers.

**Solution**:
```python
# Use Auto classes
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Update if needed
pip install transformers>=4.46.0 --upgrade
```

---

### Error: `KeyError: 'qwen2_vl'` or Qwen-VL not recognized

**Cause**: Transformers version too old for Qwen2-VL.

**Solution**:
```bash
pip install transformers>=4.46.0 --upgrade
```

---

### Error: `safetensors_rust.SafetensorError` during model load

**Cause**: Corrupted download or disk space issue.

**Solution**:
```bash
# Clear the corrupted model cache
rm -rf ~/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf

# Re-download
huggingface-cli download llava-hf/llava-1.5-7b-hf
```

---

## Generation Errors

### Error: SDXL generates black/noisy images

**Causes**:
1. Wrong dtype
2. Too few inference steps
3. Guidance scale too low

**Solutions**:
```python
# Use correct dtype
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.bfloat16  # Not float32
).to("cuda")

# Use sufficient steps and guidance
image = pipe(
    prompt="A cat sitting on a beach",
    num_inference_steps=30,  # Minimum 20-30
    guidance_scale=7.5       # 7-9 is typical
).images[0]
```

---

### Error: ControlNet produces distorted results

**Causes**:
1. Input image not preprocessed correctly
2. Conditioning scale too high

**Solutions**:
```python
# Properly preprocess control image
from controlnet_aux import CannyDetector
canny = CannyDetector()
control_image = canny(input_image, low_threshold=100, high_threshold=200)

# Adjust conditioning scale
image = pipe(
    prompt="...",
    image=control_image,
    controlnet_conditioning_scale=0.5  # Try 0.3-0.8
).images[0]
```

---

### Error: Whisper transcription is wrong language

**Cause**: Auto-detect guessed wrong language.

**Solution**:
```python
result = model.transcribe(
    "audio.mp3",
    language="en",  # Explicitly set language
    task="transcribe"
)
```

---

### Error: Whisper `RuntimeError: expected scalar type Half`

**Cause**: Model precision mismatch.

**Solution**:
```python
model = whisper.load_model("large-v3")
# Force to correct device and dtype
model = model.to("cuda").float()
```

---

## ChromaDB/RAG Errors

### Error: `chromadb.errors.InvalidCollectionException`

**Cause**: Collection already exists with different settings.

**Solution**:
```python
# Delete and recreate
client.delete_collection("multimodal")
collection = client.create_collection("multimodal", embedding_function=ef)

# Or get existing
collection = client.get_or_create_collection("multimodal", embedding_function=ef)
```

---

### Error: CLIP embeddings not matching in similarity search

**Cause**: Embeddings not normalized.

**Solution**:
```python
import torch.nn.functional as F

# Always normalize CLIP embeddings
image_embedding = model.get_image_features(**inputs)
image_embedding = F.normalize(image_embedding, dim=-1)

text_embedding = model.get_text_features(**inputs)
text_embedding = F.normalize(text_embedding, dim=-1)
```

---

## PDF/Document Errors

### Error: `fitz.FileDataError: cannot open document`

**Cause**: PDF is corrupted or password-protected.

**Solution**:
```python
import fitz

try:
    doc = fitz.open("document.pdf")
except fitz.FileDataError:
    # Try with password
    doc = fitz.open("document.pdf", password="")
    # Or skip the file
    print("Cannot open PDF, may be corrupted")
```

---

### Error: `pdf2image.exceptions.PDFPageCountError`

**Cause**: poppler not installed.

**Solution**:
```bash
apt-get update && apt-get install -y poppler-utils
```

---

## Reset Procedures

### Full Environment Reset

```bash
# 1. Exit container
exit

# 2. Clear HuggingFace cache (if needed)
rm -rf ~/.cache/huggingface/hub/models--*

# 3. Restart container
docker run --gpus all -it --rm \
    -v $HOME/workspace:/workspace \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    nvcr.io/nvidia/pytorch:25.11-py3
```

### Memory-Only Reset

```python
import torch
import gc

# Delete all model variables
del model, processor, pipe  # Add any other model vars

# Clear all CUDA memory
torch.cuda.empty_cache()
gc.collect()

# Verify
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

---

## ❓ Frequently Asked Questions

### Setup & Environment

**Q: Which VLM should I use - LLaVA or Qwen-VL?**

A: For most cases, start with **LLaVA-1.5-7B** (good quality, easy to use). Use **Qwen2-VL** if you need better instruction following or multilingual support. LLaVA-13B offers better quality if you have memory to spare.

| Model | Best For |
|-------|----------|
| LLaVA-7B | Learning, quick experiments |
| LLaVA-13B | Better quality, general use |
| Qwen2-VL-7B | Instruction following, multilingual |

---

**Q: How much GPU memory do I need for each lab?**

A: With DGX Spark's 128GB unified memory, you have plenty for any single model. Here's what each requires:

| Lab | Minimum Memory | Comfortable |
|-----|----------------|-------------|
| 4.1.1 VLM | 16GB | 30GB |
| 4.1.2 Image Gen | 8GB | 15GB |
| 4.1.3 Multimodal RAG | 4GB | 10GB |
| 4.1.4 Document AI | 16GB | 30GB |
| 4.1.5 Audio | 4GB | 8GB |

**Important**: Clear memory between labs with `torch.cuda.empty_cache()`.

---

**Q: Can I run multiple models simultaneously?**

A: Yes! This is a DGX Spark advantage. With 128GB, you can run CLIP for search + an LLM for response + keep embeddings in memory. Just be mindful of total usage.

```python
# Check available memory
free = (torch.cuda.get_device_properties(0).total_memory -
        torch.cuda.memory_allocated()) / 1e9
print(f"Available: {free:.1f} GB")
```

---

### Concepts

**Q: What's the difference between a VLM and just using CLIP + LLM separately?**

A:

- **CLIP + LLM**: Two separate models. CLIP describes the image in embedding space, LLM generates text. They don't truly "understand" each other.

- **VLM (LLaVA, Qwen-VL)**: Single unified model trained end-to-end. Image and text are processed together, so the model can reason about visual details while generating text.

**Result**: VLMs give more coherent, detailed responses about images.

---

**Q: Why use CLIP if VLMs are better?**

A: Different use cases:

| CLIP | VLMs |
|------|------|
| Fast embedding (~50ms) | Slow generation (~3s) |
| Good for search/retrieval | Good for understanding/description |
| Low memory (~2GB) | Higher memory (16GB+) |
| No generation | Full text generation |

**Best practice**: Use CLIP to find relevant images, then VLM to analyze the top results.

---

**Q: What's the difference between SDXL and Flux?**

A:

- **SDXL**: Faster (~5-8s), good quality, well-supported ecosystem, more ControlNet options
- **Flux**: Slower (~15-20s), higher quality, better prompt following, newer architecture

**Start with SDXL** for learning. Use Flux when quality matters more than speed.

---

**Q: How does multimodal RAG work differently from text RAG?**

A:

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

**Q: What does "guidance scale" mean in image generation?**

A: Guidance scale controls how closely the model follows your prompt:

- **Low (1-5)**: More creative/random, less prompt adherence
- **Medium (7-9)**: Good balance (recommended)
- **High (10+)**: Very strict prompt following, can look artificial

**Default**: Use 7.5 for SDXL, 3.5 for Flux.

---

### Additional Troubleshooting FAQs

**Q: My VLM gives generic responses like "I see an image"**

A: This usually means the image isn't being processed correctly. Check:

1. Image is in correct format (PIL Image or tensor)
2. Prompt includes the `<image>` token in the right place
3. Image is passed to processor correctly

```python
# Correct LLaVA format
prompt = "USER: <image>\nDescribe this image.\nASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt")
```

---

**Q: SDXL generates weird artifacts or distortions**

A: Common causes:

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

**Q: Whisper transcription is slow**

A: Whisper speed depends on model size:

| Model | Speed | Quality |
|-------|-------|---------|
| tiny | 32x real-time | Low |
| base | 16x real-time | OK |
| small | 6x real-time | Good |
| medium | 2x real-time | Better |
| large-v3 | 0.5x real-time | Best |

**Tip**: Use `base` or `small` for development, `large-v3` for final output.

---

**Q: My multimodal RAG returns wrong images**

A: Check:

1. **Embeddings normalized**: CLIP embeddings must be L2-normalized
2. **Same embedding model for index and query**
3. **Sufficient images indexed**: Need variety for good retrieval

```python
# Always normalize
embedding = F.normalize(embedding, dim=-1)
```

---

### Beyond the Basics

**Q: Can I use these models for production applications?**

A: Yes, but consider:

- **LLaVA/Qwen-VL**: Open weights, check licenses for commercial use
- **SDXL**: Stability AI license (mostly permissive)
- **Flux**: Black Forest Labs license (check restrictions)
- **Whisper**: MIT license (very permissive)

**Important**: Add safety filters for user-facing applications (Module 4.2).

---

**Q: How do I make VLMs faster for real-time applications?**

A: Options:

1. **Quantization**: 4-bit reduces latency ~2x
2. **Smaller models**: 7B vs 13B
3. **Batching**: Process multiple images at once
4. **vLLM**: Optimized serving (covered in Module 3.3)

---

**Q: Can I fine-tune these multimodal models?**

A: Yes, but it's more complex:

- **VLMs**: Can fine-tune with LoRA on instruction data
- **SDXL**: Fine-tune with DreamBooth or LoRA for specific styles
- **Whisper**: Can fine-tune for domain-specific audio

This curriculum focuses on using pre-trained models. Fine-tuning multimodal models is an advanced topic.

---

**Q: How does document AI compare to just using OCR?**

A:

| OCR Only | Document AI |
|----------|-------------|
| Extracts raw text | Understands structure |
| Loses layout | Preserves tables, headers |
| Can't read charts | VLM describes charts |
| Fast | Slower |

**Document AI = OCR + Layout Analysis + VLM + LLM**

---

## Still Stuck?

1. **Check the notebook comments** - Often contain hints
2. **Review prerequisites** - Missing foundation knowledge?
3. **Search the error message** - Include "DGX Spark" or "Blackwell" in search
4. **Ask with context** - Include: full error traceback, code, what you tried
5. **Check model cards** - HuggingFace model pages often have usage examples
