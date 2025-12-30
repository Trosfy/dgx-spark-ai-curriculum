# Module 14: Multimodal AI - Data

This directory contains sample data and data generation utilities for the Multimodal AI module.

## Overview

Most data for this module is generated programmatically within the notebooks to avoid large file storage and ensure reproducibility. This README documents the data used and how to obtain or generate it.

## Data Sources

### Images

The notebooks use several types of images:

1. **Synthetic Images**: Generated using PIL/Pillow
   - Simple shapes (circles, squares, triangles)
   - Color variations
   - Document mockups (invoices, reports)

2. **Sample Images from URLs**: Loaded from Unsplash (royalty-free)
   - Landscape photos
   - Street scenes
   - Various subjects for VLM testing

3. **User-Provided Images**: You can use your own images for testing

### Audio

Audio data is primarily synthetic for demonstration:

1. **Sine Wave Tones**: Generated with NumPy
2. **DTMF Tones**: Phone dial sounds
3. **User Audio**: Load your own .wav, .mp3, or .flac files

### Documents

Document images are created programmatically:

1. **Sample Invoice**: Generated with PIL drawing
2. **Sample Report**: Business report mockup
3. **User Documents**: Upload your own PDFs or document images

## Generating Sample Data

### Create Sample Images

```python
from PIL import Image, ImageDraw

# Simple colored shape
def create_sample_image(shape='circle', color='red'):
    img = Image.new('RGB', (224, 224), 'white')
    draw = ImageDraw.Draw(img)

    if shape == 'circle':
        draw.ellipse([50, 50, 174, 174], fill=color)
    elif shape == 'square':
        draw.rectangle([50, 50, 174, 174], fill=color)

    return img

# Usage
img = create_sample_image('circle', 'blue')
img.save('sample_circle.png')
```

### Create Sample Audio

```python
import numpy as np
import soundfile as sf

def create_sample_audio(duration=5.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    return audio.astype(np.float32)

# Usage
audio = create_sample_audio()
sf.write('sample_audio.wav', audio, 16000)
```

### Create Sample Documents

```python
from scripts.document_ai import create_sample_invoice, create_sample_report

# Generate sample documents
invoice = create_sample_invoice()
invoice.save('sample_invoice.png')

report = create_sample_report()
report.save('sample_report.png')
```

## Loading External Data

### Images from URL

```python
import requests
from PIL import Image
from io import BytesIO

def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert('RGB')

# Example: Unsplash images (royalty-free)
image = load_image_from_url("https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=800")
```

### Audio Files

```python
import librosa

# Load any audio file (resampled to 16kHz for Whisper)
audio, sr = librosa.load("your_audio.mp3", sr=16000, mono=True)
```

### PDF Documents

```python
import fitz  # PyMuPDF

# Convert PDF pages to images
doc = fitz.open("document.pdf")
for page_num, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(f"page_{page_num}.png")
```

## Dataset Suggestions

For more comprehensive testing, consider these public datasets:

### Vision-Language

- **Flickr30k**: Image captioning dataset
  - https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

- **COCO Captions**: Common Objects in Context
  - https://cocodataset.org/

### Document AI

- **DocVQA**: Document Visual Question Answering
  - https://www.docvqa.org/

- **FUNSD**: Form Understanding in Noisy Scanned Documents
  - https://guillaumejaume.github.io/FUNSD/

### Audio/Speech

- **LibriSpeech**: English speech corpus
  - https://www.openslr.org/12/

- **Common Voice**: Mozilla's multilingual dataset
  - https://commonvoice.mozilla.org/

## File Structure

```
data/
├── README.md          # This file
├── images/            # Store sample images here (gitignored)
├── audio/             # Store audio files here (gitignored)
└── documents/         # Store document files here (gitignored)
```

## Notes

1. **Memory Considerations**: On DGX Spark with 128GB unified memory, you can process large batches of images or long audio files without concern.

2. **GPU Optimization**: All models use bfloat16 for optimal Blackwell performance.

3. **Caching**: HuggingFace models are cached in `~/.cache/huggingface`. Mount this directory when using Docker containers.

4. **Privacy**: For sensitive data, all processing happens locally on DGX Spark - no data leaves your machine.
