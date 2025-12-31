#!/usr/bin/env python3
"""
Document Intelligence Demo

Quick demonstration of the document processing pipeline.
"""

from document_processor import SimpleDocumentProcessor, StructuredExtractor
from vlm_extractor import SimpleVLMExtractor


def main():
    print("=" * 60)
    print("MULTIMODAL DOCUMENT INTELLIGENCE DEMO")
    print("=" * 60)

    # 1. Document Processing
    print("\n1. DOCUMENT PROCESSING")
    print("-" * 40)

    processor = SimpleDocumentProcessor()
    doc = processor.process_pdf("sample_report.pdf")

    print(f"Document: {doc.filename}")
    print(f"Pages: {doc.total_pages}")
    print(f"Full text preview: {doc.full_text[:100]}...")

    tables = processor.extract_tables(doc)
    images = processor.extract_images(doc)
    print(f"Found {len(tables)} tables and {len(images)} images")

    # 2. Structured Extraction
    print("\n2. STRUCTURED EXTRACTION")
    print("-" * 40)

    extractor = StructuredExtractor()
    extractor.register_schema("technical_doc", {
        "fields": {
            "topic": {"keywords": ["machine learning", "AI", "neural"]},
            "type": {"keywords": ["report", "paper", "guide"]},
        }
    })

    result = extractor.extract(doc, "technical_doc")
    print(f"Extracted type: {result['type']}")
    for field, value in result["extracted_fields"].items():
        print(f"  - {field}: {value}")

    # 3. VLM Analysis
    print("\n3. VLM ANALYSIS (Demo Mode)")
    print("-" * 40)

    vlm = SimpleVLMExtractor(load_model=False)

    analysis = vlm.analyze_image(
        "chart.png",
        "Describe the data shown in this chart."
    )

    print(f"Analysis confidence: {analysis.confidence}")
    print(f"Content preview: {analysis.content[:200]}...")

    # 4. Invoice Extraction Example
    print("\n4. INVOICE EXTRACTION EXAMPLE")
    print("-" * 40)

    invoice_schema = {
        "fields": {
            "invoice_id": {"type": "string"},
            "vendor": {"type": "string"},
            "amount": {"type": "number"},
            "date": {"type": "date"},
            "line_items": {"type": "array"}
        }
    }

    invoice_result = vlm.extract_structured("invoice.png", invoice_schema)
    print("Extracted invoice data:")
    if invoice_result.structured_data:
        for key, value in invoice_result.structured_data.items():
            print(f"  - {key}: {value}")

    # 5. Pipeline Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("""
This demo shows the core components:

1. Document Processing
   - Load and parse PDF/images
   - Extract text, tables, images
   - Page-level organization

2. Structured Extraction
   - Define extraction schemas
   - Keyword-based matching
   - Field extraction

3. VLM Analysis
   - Image understanding
   - Natural language descriptions
   - Visual element detection

4. Combined Pipeline
   - Document -> Pages -> Elements
   - VLM for complex content
   - Structured output generation

For your full capstone:
- Add real PDF processing (PyMuPDF)
- Load actual VLM model (Qwen2-VL)
- Build vector store for multimodal RAG
- Create FastAPI endpoint for uploads
- Add validation and error handling
""")


if __name__ == "__main__":
    main()
