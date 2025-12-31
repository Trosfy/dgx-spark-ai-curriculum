#!/usr/bin/env python3
"""
Simple Document Processor

A minimal implementation for document ingestion and processing.
This is a starting point - extend this for your capstone!
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DocumentPage:
    """Represents a single page from a document."""
    page_number: int
    text: str
    images: List[str]  # Paths to extracted images
    tables: List[Dict]  # Extracted tables
    metadata: Dict[str, Any]


@dataclass
class Document:
    """Represents a processed document."""
    filename: str
    pages: List[DocumentPage]
    metadata: Dict[str, Any]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class SimpleDocumentProcessor:
    """
    Simple document processor (mock for demo).

    In your capstone, replace this with real PDF/image processing.
    """

    def __init__(self, output_dir: str = "./extracted"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_pdf(self, pdf_path: str) -> Document:
        """
        Process a PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Processed Document object
        """
        # Mock implementation - replace with real PDF processing
        # In production, use: PyMuPDF, pdf2image, pdfplumber

        print(f"Processing: {pdf_path}")

        # Simulate extracted pages
        mock_pages = [
            DocumentPage(
                page_number=1,
                text="This is a sample document page 1. It contains text about AI and machine learning.",
                images=["page1_figure1.png"],
                tables=[{"header": ["Name", "Value"], "rows": [["Learning Rate", "0.001"]]}],
                metadata={"has_header": True}
            ),
            DocumentPage(
                page_number=2,
                text="Page 2 continues with more technical content about neural networks and deep learning.",
                images=[],
                tables=[],
                metadata={"has_header": False}
            ),
        ]

        return Document(
            filename=Path(pdf_path).name,
            pages=mock_pages,
            metadata={
                "processed": True,
                "format": "pdf",
                "source": pdf_path
            }
        )

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a standalone image.

        Args:
            image_path: Path to image file

        Returns:
            Extracted information
        """
        # Mock implementation
        return {
            "path": image_path,
            "type": "image",
            "detected_content": ["text", "diagram"],
            "ocr_text": "Sample OCR extracted text from image",
            "description": "A technical diagram showing system architecture"
        }

    def extract_tables(self, document: Document) -> List[Dict]:
        """Extract all tables from a document."""
        all_tables = []
        for page in document.pages:
            for table in page.tables:
                all_tables.append({
                    "page": page.page_number,
                    "table": table
                })
        return all_tables

    def extract_images(self, document: Document) -> List[Dict]:
        """Extract all image references from a document."""
        all_images = []
        for page in document.pages:
            for img in page.images:
                all_images.append({
                    "page": page.page_number,
                    "image": img
                })
        return all_images


class StructuredExtractor:
    """
    Extract structured data from documents.

    Define extraction schemas for specific document types.
    """

    def __init__(self):
        self.schemas = {}

    def register_schema(self, doc_type: str, schema: Dict):
        """Register an extraction schema."""
        self.schemas[doc_type] = schema
        print(f"Registered schema: {doc_type}")

    def extract(self, document: Document, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured data based on schema.

        Args:
            document: Processed document
            doc_type: Type of document (must have registered schema)

        Returns:
            Extracted structured data
        """
        if doc_type not in self.schemas:
            return {"error": f"No schema for {doc_type}"}

        schema = self.schemas[doc_type]

        # Mock extraction - in production, use LLM for extraction
        result = {
            "document": document.filename,
            "type": doc_type,
            "extracted_fields": {}
        }

        # Simple keyword matching for demo
        text = document.full_text.lower()

        for field, config in schema.get("fields", {}).items():
            keywords = config.get("keywords", [])
            for kw in keywords:
                if kw.lower() in text:
                    result["extracted_fields"][field] = f"Found: {kw}"
                    break
            else:
                result["extracted_fields"][field] = None

        return result


# Example usage
if __name__ == "__main__":
    print("Document Processor Demo")
    print("=" * 50)

    # Create processor
    processor = SimpleDocumentProcessor()

    # Process a mock document
    doc = processor.process_pdf("sample_report.pdf")

    print(f"\nProcessed: {doc.filename}")
    print(f"Pages: {doc.total_pages}")
    print(f"Tables: {len(processor.extract_tables(doc))}")
    print(f"Images: {len(processor.extract_images(doc))}")

    # Structured extraction
    print("\n" + "-" * 50)
    extractor = StructuredExtractor()

    # Register a schema
    extractor.register_schema("research_paper", {
        "fields": {
            "title": {"keywords": ["abstract", "introduction"]},
            "methodology": {"keywords": ["method", "approach"]},
            "results": {"keywords": ["results", "findings"]},
        }
    })

    # Extract
    result = extractor.extract(doc, "research_paper")
    print(f"\nExtracted fields: {json.dumps(result, indent=2)}")
