# Example: Multimodal Document Intelligence

This is a minimal working example for Option B: Multimodal Document Intelligence.

## What This Example Includes

- `document_processor.py` - Document ingestion and extraction
- `vlm_extractor.py` - Vision-language model integration
- `demo.py` - Quick demonstration script

## Quick Start

```bash
# Navigate to this directory
cd examples/option-b-document-intelligence

# Run the demo
python demo.py
```

## Files Overview

### document_processor.py

A minimal document processor with:
- PDF text extraction
- Image extraction from documents
- Page-level processing

### vlm_extractor.py

Vision-language model wrapper with:
- Image analysis
- Structured data extraction
- OCR fallback

### demo.py

Interactive demo showing:
- Document loading
- Visual element extraction
- Structured output generation

## Extending This Example

To build your full capstone, you'll want to:

1. **Add real PDF processing** with PyMuPDF or pdf2image
2. **Integrate VLM** like Qwen2-VL or LLaVA
3. **Build multimodal RAG** with image embeddings
4. **Create structured extractors** for specific document types
5. **Add validation** for extracted data
6. **Build an API** for document upload and processing

## Memory Requirements

This example is designed to run on DGX Spark:
- Uses Qwen2-VL-7B (or similar)
- 4-bit quantization
- Estimated: ~8GB GPU memory

For your full capstone, you can scale up to larger VLMs.

## Next Steps

1. Review the code in each file
2. Run the demo to see it working
3. Add your specific document types
4. Follow the main Option B notebook for full implementation
