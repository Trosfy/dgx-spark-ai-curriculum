# Module 4.1: Multimodal AI - Data Directory

This directory contains sample data files and documentation for the multimodal AI labs.

## Directory Structure

```
data/
├── README.md          # This file
├── images/            # Sample images for VLM and RAG labs
├── documents/         # Sample PDFs for Document AI lab
├── audio/             # Sample audio files for transcription lab
└── embeddings/        # Cached embeddings (generated during labs)
```

## Sample Data Sources

### Images

For the Vision-Language and Multimodal RAG labs, you can use:

1. **Public Domain Images**:
   - [Wikimedia Commons](https://commons.wikimedia.org/)
   - [Unsplash](https://unsplash.com/) (free for commercial use)
   - [Pexels](https://www.pexels.com/)

2. **Sample URLs used in labs**:
   ```python
   # Cat image
   "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

   # Dog image
   "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
   ```

3. **Recommended test images**:
   - Animals (cats, dogs, birds)
   - Landscapes (mountains, beaches, cities)
   - Objects (cars, furniture, food)
   - Documents (invoices, receipts, forms)

### Documents

For the Document AI lab, you can use:

1. **Sample PDFs** (generated in lab):
   - The lab creates sample PDFs programmatically
   - These include multi-page documents with text, tables, and figures

2. **Real-world test documents**:
   - Invoices and receipts
   - Research papers (from arXiv)
   - Business reports
   - Government forms

3. **Creating test PDFs**:
   ```python
   import fitz  # PyMuPDF

   doc = fitz.open()
   page = doc.new_page()
   page.insert_text((72, 72), "Sample Document", fontsize=24)
   doc.save("sample.pdf")
   ```

### Audio

For the Audio Transcription lab, you can use:

1. **Test Audio Sources**:
   - [LibriSpeech](https://www.openslr.org/12/) - Public domain audiobooks
   - [Mozilla Common Voice](https://commonvoice.mozilla.org/) - Crowd-sourced speech
   - Personal recordings (meetings, podcasts)

2. **Supported formats**:
   - WAV (recommended)
   - MP3
   - M4A
   - FLAC
   - OGG

3. **Creating test audio**:
   ```python
   import soundfile as sf
   import numpy as np

   # Generate 5 seconds of audio
   sample_rate = 16000
   duration = 5.0
   t = np.linspace(0, duration, int(sample_rate * duration))
   audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

   sf.write("test_audio.wav", audio, sample_rate)
   ```

## Data Formats

### Image Formats
- **Input**: PNG, JPEG, WebP, BMP, GIF (first frame)
- **Output**: PNG (recommended for quality), JPEG (for size)
- **Resolution**: 1024x1024 recommended for SDXL

### Document Formats
- **Input**: PDF (native or scanned), PNG, JPEG, TIFF
- **Output**: JSON (structured data), Markdown (text), SRT/VTT (subtitles)

### Audio Formats
- **Input**: WAV, MP3, M4A, FLAC, OGG
- **Sample rate**: 16 kHz (Whisper native)
- **Channels**: Mono recommended

## Storage Considerations

| Data Type | Typical Size | DGX Spark Storage |
|-----------|--------------|-------------------|
| Images | 100KB - 5MB | Minimal impact |
| PDFs | 100KB - 50MB | Minimal impact |
| Audio (1 hour) | 50-200MB | Monitor if many |
| Embeddings | 3KB per item | ~1GB per 100K items |
| Generated images | 1-5MB each | Monitor if many |

## Generated Data

During the labs, the following data may be generated:

1. **ChromaDB Databases** (`~/.chroma/` or specified path)
   - Vector embeddings for RAG systems
   - Can be deleted to reset

2. **HuggingFace Cache** (`~/.cache/huggingface/`)
   - Model weights (several GB per model)
   - Shared across sessions

3. **Generated Images** (in working directory)
   - SDXL/Flux outputs
   - ControlNet results

4. **Transcripts** (in working directory)
   - TXT, SRT, VTT, JSON formats

## Cleanup Commands

```bash
# Clear HuggingFace cache (WARNING: will require re-download)
rm -rf ~/.cache/huggingface/

# Clear ChromaDB data
rm -rf ~/.chroma/

# Clear generated images in workspace
rm -f /workspace/*.png /workspace/*.jpg

# Clear transcripts
rm -f /workspace/*.srt /workspace/*.vtt /workspace/*.txt
```

## Privacy Considerations

When working with real data:

1. **Personal Images**: Be mindful of privacy when using photos with faces
2. **Confidential Documents**: Don't use sensitive business documents for testing
3. **Audio Recordings**: Ensure consent for any recordings with identifiable speakers
4. **Generated Content**: Review AI-generated content before sharing

## Data Generation Scripts

The labs include utility functions to generate sample data:

- `create_sample_pdf()` - Generate test PDF documents
- `create_sample_audio()` - Generate test audio files
- `load_image_from_url()` - Download and cache images

These allow you to test the labs without needing external data.

---

## Next Steps

1. Run the labs in order (01 → 05)
2. Use sample URLs provided in the notebooks
3. Gradually introduce your own data for testing
4. Experiment with different data types and edge cases
