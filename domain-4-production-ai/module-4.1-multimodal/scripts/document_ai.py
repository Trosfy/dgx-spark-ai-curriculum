"""
Document AI Utilities

This module provides utilities for document processing, OCR, and
VLM-based document understanding on DGX Spark.

Example usage:
    from document_ai import DocumentProcessor

    # Initialize
    processor = DocumentProcessor()
    processor.load()

    # Analyze a document
    doc_image = Image.open("invoice.png")
    result = processor.analyze(doc_image)
    print(result['summary'])

    # Extract fields
    fields = processor.extract_fields(doc_image, ["Total", "Date", "Invoice #"])

    processor.cleanup()
"""

import torch
import gc
import time
from typing import Optional, List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> str:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        return f"Allocated: {allocated:.2f}GB"
    return "No GPU"


def create_sample_invoice() -> Image.Image:
    """
    Create a sample invoice image for testing.

    Returns:
        PIL Image of a fake invoice

    Example:
        >>> invoice = create_sample_invoice()
        >>> invoice.save("sample_invoice.png")
    """
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)

    # Try to load DejaVu fonts (common on Linux/NGC containers)
    # Falls back to default bitmap font if unavailable (e.g., on minimal systems)
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header
    draw.text((50, 30), "INVOICE", fill='navy', font=font_large)
    draw.text((600, 30), "#INV-2024-0042", fill='black', font=font_medium)

    # Company info
    draw.text((50, 80), "TechCorp Solutions Inc.", fill='black', font=font_medium)
    draw.text((50, 105), "123 Innovation Street", fill='gray', font=font_small)
    draw.text((50, 125), "San Francisco, CA 94105", fill='gray', font=font_small)

    # Bill To
    draw.text((50, 180), "Bill To:", fill='black', font=font_medium)
    draw.text((50, 205), "Acme Corporation", fill='black', font=font_small)
    draw.text((50, 225), "456 Business Ave", fill='gray', font=font_small)

    # Date info
    draw.text((500, 180), "Date: December 15, 2024", fill='black', font=font_small)
    draw.text((500, 200), "Due: January 15, 2025", fill='black', font=font_small)

    # Table header
    y = 320
    draw.rectangle([50, y, 750, y+30], fill='lightgray')
    draw.text((60, y+5), "Description", fill='black', font=font_small)
    draw.text((400, y+5), "Qty", fill='black', font=font_small)
    draw.text((500, y+5), "Unit Price", fill='black', font=font_small)
    draw.text((650, y+5), "Amount", fill='black', font=font_small)

    # Table rows
    items = [
        ("AI Development Services", "40", "$150.00", "$6,000.00"),
        ("Model Training (GPU hours)", "100", "$25.00", "$2,500.00"),
        ("Data Preprocessing", "20", "$75.00", "$1,500.00"),
    ]

    y = 360
    for desc, qty, unit, amount in items:
        draw.text((60, y), desc, fill='black', font=font_small)
        draw.text((400, y), qty, fill='black', font=font_small)
        draw.text((500, y), unit, fill='black', font=font_small)
        draw.text((650, y), amount, fill='black', font=font_small)
        draw.line([(50, y+25), (750, y+25)], fill='lightgray', width=1)
        y += 35

    # Totals
    y = 480
    draw.text((500, y), "Subtotal:", fill='black', font=font_small)
    draw.text((650, y), "$10,000.00", fill='black', font=font_small)

    draw.text((500, y+25), "Tax (8.5%):", fill='black', font=font_small)
    draw.text((650, y+25), "$850.00", fill='black', font=font_small)

    draw.line([(500, y+50), (750, y+50)], fill='black', width=2)

    draw.text((500, y+60), "Total Due:", fill='black', font=font_medium)
    draw.text((650, y+60), "$10,850.00", fill='navy', font=font_medium)

    return img


def create_sample_report() -> Image.Image:
    """
    Create a sample business report image for testing.

    Returns:
        PIL Image of a fake report
    """
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)

    # Try to load DejaVu fonts (common on Linux/NGC containers)
    # Falls back to default bitmap font if unavailable
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    draw.text((50, 30), "Q4 2024 Performance Report", fill='darkblue', font=font_large)
    draw.line([(50, 65), (750, 65)], fill='darkblue', width=2)

    # Executive Summary
    draw.text((50, 85), "Executive Summary", fill='black', font=font_medium)
    summary = "Revenue increased 23% to $3.4M. Profit margin improved to 47%."
    draw.text((50, 115), summary, fill='gray', font=font_small)

    # Key Metrics Table
    draw.text((50, 180), "Quarterly Breakdown", fill='black', font=font_medium)

    y = 210
    draw.rectangle([50, y, 700, y+25], fill='lightgray')
    headers = ["Quarter", "Revenue", "Profit", "Growth"]
    x_positions = [60, 180, 340, 500]

    for x, header in zip(x_positions, headers):
        draw.text((x, y+5), header, fill='black', font=font_small)

    rows = [
        ("Q1 2024", "$2.1M", "$700K", "+12%"),
        ("Q2 2024", "$2.4M", "$900K", "+15%"),
        ("Q3 2024", "$2.8M", "$1.2M", "+18%"),
        ("Q4 2024", "$3.4M", "$1.6M", "+23%"),
    ]

    y = 240
    for row in rows:
        for x, cell in zip(x_positions, row):
            draw.text((x, y), cell, fill='black', font=font_small)
        y += 25

    return img


class DocumentProcessor:
    """
    Document Processing Pipeline.

    Uses VLMs for document understanding, information extraction,
    and question answering about documents.

    Attributes:
        model: Loaded VLM model
        processor: VLM processor
        documents: Dictionary of stored documents

    Example:
        >>> proc = DocumentProcessor()
        >>> proc.load()
        >>> image = Image.open("invoice.png")
        >>> result = proc.analyze(image)
        >>> print(result)
    """

    def __init__(self, model_name: str = "qwen"):
        """
        Initialize Document Processor.

        Args:
            model_name: VLM to use ('qwen' or 'llava')
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.documents: Dict[str, Dict] = {}
        self._loaded = False

    def load(self) -> None:
        """Load the VLM for document processing."""
        if self._loaded:
            return

        clear_gpu_memory()
        print(f"Loading {self.model_name} for document processing...")
        start_time = time.time()

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_id = "Qwen/Qwen2-VL-7B-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        load_time = time.time() - start_time
        self._loaded = True
        print(f"Loaded in {load_time:.1f}s")

    def _ask(self, image: Image.Image, question: str, max_tokens: int = 300) -> str:
        """Internal method to ask a question about an image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def add_document(
        self,
        doc_id: str,
        image: Image.Image,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a document to the collection.

        Args:
            doc_id: Unique identifier
            image: Document image
            metadata: Optional metadata
        """
        self.documents[doc_id] = {
            'image': image,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        print(f"Added document: {doc_id}")

    def analyze(self, image: Image.Image) -> Dict[str, str]:
        """
        Perform comprehensive analysis of a document.

        Args:
            image: Document image

        Returns:
            Dictionary with analysis results
        """
        if not self._loaded:
            self.load()

        analysis = {}

        analysis['type'] = self._ask(
            image,
            "What type of document is this? (e.g., invoice, report, form, letter)"
        )

        analysis['summary'] = self._ask(
            image,
            "Provide a brief summary of this document's key information."
        )

        analysis['has_tables'] = self._ask(
            image,
            "Does this document contain any tables? If yes, briefly describe them."
        )

        return analysis

    def extract_fields(
        self,
        image: Image.Image,
        fields: List[str]
    ) -> Dict[str, str]:
        """
        Extract specific fields from a document.

        Args:
            image: Document image
            fields: List of field names to extract

        Returns:
            Dictionary of field values
        """
        if not self._loaded:
            self.load()

        fields_str = "\n".join([f"- {field}" for field in fields])
        prompt = f"""Extract the following fields from this document:
{fields_str}

Format your response as:
Field Name: Value"""

        response = self._ask(image, prompt)

        # Parse response
        extracted = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                extracted[key.strip()] = value.strip()

        return extracted

    def ask(self, question: str, doc_id: Optional[str] = None) -> str:
        """
        Ask a question about document(s).

        Args:
            question: Question to ask
            doc_id: Specific document ID (None = use most recent)

        Returns:
            Answer string
        """
        if not self._loaded:
            self.load()

        if doc_id:
            if doc_id not in self.documents:
                return f"Document '{doc_id}' not found"
            image = self.documents[doc_id]['image']
        elif self.documents:
            doc_id = list(self.documents.keys())[-1]
            image = self.documents[doc_id]['image']
        else:
            return "No documents available"

        return self._ask(image, question)

    def compare(self, doc_id1: str, doc_id2: str) -> str:
        """
        Compare two documents.

        Args:
            doc_id1: First document ID
            doc_id2: Second document ID

        Returns:
            Comparison analysis
        """
        if doc_id1 not in self.documents or doc_id2 not in self.documents:
            return "One or both documents not found"

        # Get summaries of both
        summary1 = self._ask(
            self.documents[doc_id1]['image'],
            "Summarize the key information in this document in 2-3 sentences."
        )

        summary2 = self._ask(
            self.documents[doc_id2]['image'],
            "Summarize the key information in this document in 2-3 sentences."
        )

        return f"Document 1: {summary1}\n\nDocument 2: {summary2}"

    def list_documents(self) -> List[str]:
        """Get list of document IDs."""
        return list(self.documents.keys())

    def cleanup(self) -> None:
        """Release resources."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            clear_gpu_memory()


if __name__ == "__main__":
    print("Document AI Demo")
    print("=" * 50)

    # Create sample invoice
    invoice = create_sample_invoice()
    invoice.save("sample_invoice.png")
    print("Created sample_invoice.png")

    # Initialize processor
    processor = DocumentProcessor()
    processor.load()

    # Analyze
    result = processor.analyze(invoice)
    print(f"\\nDocument type: {result['type']}")
    print(f"Summary: {result['summary']}")

    # Extract fields
    fields = processor.extract_fields(invoice, ["Invoice Number", "Total", "Due Date"])
    print(f"\\nExtracted fields: {fields}")

    processor.cleanup()
