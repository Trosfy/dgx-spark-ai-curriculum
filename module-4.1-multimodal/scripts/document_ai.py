"""
Document AI Utilities for DGX Spark

This module provides utilities for processing documents including PDFs,
performing OCR, layout analysis, and extracting structured information.

Example:
    >>> from scripts.document_ai import DocumentProcessor
    >>> processor = DocumentProcessor()
    >>> doc = processor.process_pdf("document.pdf")
    >>> print(doc.text)
"""

import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Literal
from enum import Enum

import torch
from PIL import Image


class BlockType(Enum):
    """Types of content blocks in a document."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"
    LIST = "list"
    FOOTER = "footer"
    HEADER = "header"
    PAGE_NUMBER = "page_number"


@dataclass
class TextBlock:
    """A block of text extracted from a document."""
    text: str
    block_type: BlockType
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableData:
    """Extracted table data."""
    rows: List[List[str]]
    headers: Optional[List[str]] = None
    page_number: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.rows:
            return ""

        lines = []

        # Headers
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        # Rows
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def to_dict(self) -> List[Dict[str, str]]:
        """Convert table to list of dictionaries."""
        if not self.headers or not self.rows:
            return []

        return [
            dict(zip(self.headers, row))
            for row in self.rows
        ]


@dataclass
class ProcessedDocument:
    """A fully processed document."""
    source_path: str
    num_pages: int
    text_blocks: List[TextBlock]
    tables: List[TableData]
    figures: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    @property
    def text(self) -> str:
        """Get all text content concatenated."""
        return "\n\n".join(
            block.text for block in self.text_blocks
            if block.block_type in [BlockType.TEXT, BlockType.HEADING]
        )

    @property
    def pages(self) -> Dict[int, List[TextBlock]]:
        """Get text blocks organized by page."""
        result = {}
        for block in self.text_blocks:
            if block.page_number not in result:
                result[block.page_number] = []
            result[block.page_number].append(block)
        return result


class DocumentProcessor:
    """
    Process documents with OCR and layout analysis.

    This processor supports PDFs, images, and common document formats.
    It uses a combination of PyMuPDF for text extraction and OCR
    for scanned documents.

    Attributes:
        use_gpu: Whether to use GPU acceleration.
        ocr_engine: OCR engine to use.

    Example:
        >>> processor = DocumentProcessor()
        >>> doc = processor.process_pdf("report.pdf")
        >>> print(f"Extracted {len(doc.text_blocks)} text blocks")
    """

    def __init__(
        self,
        use_gpu: bool = True,
        ocr_engine: Literal["tesseract", "easyocr"] = "tesseract",
        language: str = "eng",
    ):
        """
        Initialize the document processor.

        Args:
            use_gpu: Whether to use GPU acceleration for OCR.
            ocr_engine: OCR engine ("tesseract" or "easyocr").
            language: Language code for OCR.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.ocr_engine = ocr_engine
        self.language = language

        self._ocr_reader = None
        self._vlm_model = None
        self._vlm_processor = None

    def _ensure_ocr_ready(self) -> None:
        """Lazy initialize OCR engine."""
        if self._ocr_reader is not None:
            return

        if self.ocr_engine == "easyocr":
            import easyocr
            self._ocr_reader = easyocr.Reader(
                [self.language],
                gpu=self.use_gpu,
            )
        # tesseract doesn't need initialization

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        extract_tables: bool = True,
        extract_figures: bool = True,
        ocr_if_needed: bool = True,
        dpi: int = 150,
    ) -> ProcessedDocument:
        """
        Process a PDF document.

        Args:
            pdf_path: Path to PDF file.
            extract_tables: Whether to extract tables.
            extract_figures: Whether to extract figures.
            ocr_if_needed: Apply OCR to scanned pages.
            dpi: DPI for rendering pages (for OCR).

        Returns:
            ProcessedDocument with all extracted content.

        Example:
            >>> doc = processor.process_pdf("research_paper.pdf")
            >>> print(f"Pages: {doc.num_pages}")
            >>> print(f"Tables found: {len(doc.tables)}")
        """
        import fitz  # PyMuPDF

        pdf_path = Path(pdf_path)
        print(f"Processing PDF: {pdf_path}")
        start_time = time.time()

        pdf = fitz.open(pdf_path)

        text_blocks = []
        tables = []
        figures = []
        metadata = {
            "title": pdf.metadata.get("title", ""),
            "author": pdf.metadata.get("author", ""),
            "created": pdf.metadata.get("creationDate", ""),
            "pages": pdf.page_count,
        }

        for page_num, page in enumerate(pdf, 1):
            print(f"  Processing page {page_num}/{pdf.page_count}...")

            # Extract text directly
            text = page.get_text()

            if text.strip():
                # Has selectable text - extract structured blocks
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" in block:
                        # Text block
                        block_text = " ".join(
                            " ".join(span["text"] for span in line["spans"])
                            for line in block["lines"]
                        ).strip()

                        if block_text:
                            bbox = tuple(block["bbox"])

                            # Classify block type
                            block_type = self._classify_block(block_text, bbox, page.rect)

                            text_blocks.append(TextBlock(
                                text=block_text,
                                block_type=block_type,
                                page_number=page_num,
                                bbox=bbox,
                            ))

                    elif "image" in block and extract_figures:
                        # Image block
                        figures.append({
                            "page": page_num,
                            "bbox": tuple(block["bbox"]),
                            "width": block.get("width"),
                            "height": block.get("height"),
                        })

            elif ocr_if_needed:
                # No selectable text - likely scanned, use OCR
                print(f"    Page {page_num} appears scanned, applying OCR...")
                ocr_blocks = self._ocr_page(page, page_num, dpi)
                text_blocks.extend(ocr_blocks)

            # Extract tables
            if extract_tables:
                page_tables = self._extract_tables(page, page_num)
                tables.extend(page_tables)

        pdf.close()

        elapsed = time.time() - start_time
        print(f"  Processed in {elapsed:.1f}s")
        print(f"  Found: {len(text_blocks)} text blocks, {len(tables)} tables, {len(figures)} figures")

        return ProcessedDocument(
            source_path=str(pdf_path),
            num_pages=pdf.page_count,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            metadata=metadata,
        )

    def _classify_block(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page_rect: Any,
    ) -> BlockType:
        """Classify a text block by its content and position."""
        # Check for heading patterns
        if len(text) < 100 and text.strip():
            # Short text that could be a heading
            if re.match(r"^[A-Z][A-Z0-9\s]+$", text.strip()):
                return BlockType.HEADING
            if re.match(r"^\d+\.?\s+[A-Z]", text.strip()):
                return BlockType.HEADING
            if re.match(r"^(Chapter|Section|Part)\s+\d", text.strip(), re.I):
                return BlockType.HEADING

        # Check position for headers/footers
        x0, y0, x1, y1 = bbox
        page_height = page_rect.height

        if y0 < page_height * 0.1:
            return BlockType.HEADER
        if y1 > page_height * 0.9:
            if re.match(r"^\d+$", text.strip()):
                return BlockType.PAGE_NUMBER
            return BlockType.FOOTER

        # Check for list patterns
        if re.match(r"^[\u2022\-\*]\s", text.strip()):
            return BlockType.LIST
        if re.match(r"^\d+[\.\)]\s", text.strip()):
            return BlockType.LIST

        return BlockType.TEXT

    def _ocr_page(
        self,
        page: Any,
        page_num: int,
        dpi: int,
    ) -> List[TextBlock]:
        """Apply OCR to a page."""
        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Apply OCR
        if self.ocr_engine == "tesseract":
            return self._ocr_tesseract(img, page_num)
        else:
            return self._ocr_easyocr(img, page_num)

    def _ocr_tesseract(
        self,
        image: Image.Image,
        page_num: int,
    ) -> List[TextBlock]:
        """OCR using Tesseract."""
        import pytesseract

        # Get OCR data with bounding boxes
        data = pytesseract.image_to_data(
            image,
            lang=self.language,
            output_type=pytesseract.Output.DICT,
        )

        blocks = []
        current_block = []
        current_bbox = None

        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            if conf > 30 and text:  # Filter low-confidence OCR
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]

                current_block.append(text)

                if current_bbox is None:
                    current_bbox = [x, y, x + w, y + h]
                else:
                    current_bbox[0] = min(current_bbox[0], x)
                    current_bbox[1] = min(current_bbox[1], y)
                    current_bbox[2] = max(current_bbox[2], x + w)
                    current_bbox[3] = max(current_bbox[3], y + h)

            elif current_block:
                # End of block
                blocks.append(TextBlock(
                    text=" ".join(current_block),
                    block_type=BlockType.TEXT,
                    page_number=page_num,
                    bbox=tuple(current_bbox),
                ))
                current_block = []
                current_bbox = None

        # Don't forget last block
        if current_block:
            blocks.append(TextBlock(
                text=" ".join(current_block),
                block_type=BlockType.TEXT,
                page_number=page_num,
                bbox=tuple(current_bbox) if current_bbox else None,
            ))

        return blocks

    def _ocr_easyocr(
        self,
        image: Image.Image,
        page_num: int,
    ) -> List[TextBlock]:
        """OCR using EasyOCR."""
        self._ensure_ocr_ready()

        import numpy as np
        img_array = np.array(image)

        results = self._ocr_reader.readtext(img_array)

        blocks = []
        for bbox, text, conf in results:
            if conf > 0.3 and text.strip():
                # Convert bbox format
                x0 = min(p[0] for p in bbox)
                y0 = min(p[1] for p in bbox)
                x1 = max(p[0] for p in bbox)
                y1 = max(p[1] for p in bbox)

                blocks.append(TextBlock(
                    text=text.strip(),
                    block_type=BlockType.TEXT,
                    page_number=page_num,
                    bbox=(x0, y0, x1, y1),
                    confidence=conf,
                ))

        return blocks

    def _extract_tables(
        self,
        page: Any,
        page_num: int,
    ) -> List[TableData]:
        """Extract tables from a page."""
        tables = []

        try:
            # Use PyMuPDF's table extraction
            page_tables = page.find_tables()

            for table in page_tables:
                if table.row_count > 1:  # At least header + 1 row
                    rows = []
                    for row in table.extract():
                        # Clean row data
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(cleaned)

                    if rows:
                        tables.append(TableData(
                            rows=rows[1:] if len(rows) > 1 else [],
                            headers=rows[0] if rows else None,
                            page_number=page_num,
                            bbox=table.bbox if hasattr(table, 'bbox') else None,
                        ))

        except Exception as e:
            # Table extraction failed, continue without tables
            pass

        return tables

    def process_image(
        self,
        image_path: Union[str, Path, Image.Image],
        extract_text: bool = True,
    ) -> ProcessedDocument:
        """
        Process an image document (scanned page, screenshot, etc.).

        Args:
            image_path: Path to image or PIL Image.
            extract_text: Whether to apply OCR.

        Returns:
            ProcessedDocument with extracted content.

        Example:
            >>> doc = processor.process_image("scanned_page.jpg")
            >>> print(doc.text)
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
            source = str(image_path)
        else:
            image = image_path.convert("RGB")
            source = "image"

        text_blocks = []

        if extract_text:
            if self.ocr_engine == "tesseract":
                text_blocks = self._ocr_tesseract(image, 1)
            else:
                text_blocks = self._ocr_easyocr(image, 1)

        return ProcessedDocument(
            source_path=source,
            num_pages=1,
            text_blocks=text_blocks,
            tables=[],
            figures=[],
            metadata={"width": image.width, "height": image.height},
        )

    def extract_with_vlm(
        self,
        image: Union[str, Path, Image.Image],
        query: str,
        vlm_model: Optional[Any] = None,
        vlm_processor: Optional[Any] = None,
    ) -> str:
        """
        Extract specific information from a document using a VLM.

        Args:
            image: Document image or path.
            query: What information to extract.
            vlm_model: Vision-language model (loaded if not provided).
            vlm_processor: VLM processor.

        Returns:
            Extracted information as text.

        Example:
            >>> info = processor.extract_with_vlm(
            ...     "invoice.jpg",
            ...     "Extract the invoice number, date, and total amount"
            ... )
        """
        if vlm_model is None:
            from .vlm_utils import load_llava, analyze_image_llava
            vlm_model, vlm_processor = load_llava("llava-hf/llava-1.5-7b-hf")

        from .vlm_utils import analyze_image_llava

        # Load image if path
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        response = analyze_image_llava(
            vlm_model,
            vlm_processor,
            img,
            query,
            max_new_tokens=512,
        )

        return response


def pdf_to_images(
    pdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    fmt: str = "png",
) -> List[Path]:
    """
    Convert PDF pages to images.

    Args:
        pdf_path: Path to PDF file.
        output_dir: Directory for output images (temp dir if None).
        dpi: Resolution for rendering.
        fmt: Output format (png, jpg, etc.).

    Returns:
        List of paths to generated images.

    Example:
        >>> images = pdf_to_images("document.pdf", "output/")
        >>> print(f"Created {len(images)} images")
    """
    import fitz
    import tempfile

    pdf_path = Path(pdf_path)

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(pdf_path)
    image_paths = []

    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for page_num, page in enumerate(pdf, 1):
        pix = page.get_pixmap(matrix=mat)
        output_path = output_dir / f"page_{page_num:03d}.{fmt}"
        pix.save(str(output_path))
        image_paths.append(output_path)

    pdf.close()

    print(f"Converted {len(image_paths)} pages to images")
    return image_paths


def merge_text_blocks(
    blocks: List[TextBlock],
    min_gap: int = 20,
) -> List[TextBlock]:
    """
    Merge adjacent text blocks that belong together.

    Args:
        blocks: List of text blocks to merge.
        min_gap: Maximum vertical gap to merge (pixels).

    Returns:
        Merged text blocks.
    """
    if not blocks:
        return []

    # Sort by page and position
    sorted_blocks = sorted(
        blocks,
        key=lambda b: (b.page_number, b.bbox[1] if b.bbox else 0, b.bbox[0] if b.bbox else 0),
    )

    merged = []
    current = None

    for block in sorted_blocks:
        if current is None:
            current = block
            continue

        # Check if should merge
        should_merge = (
            block.page_number == current.page_number
            and block.block_type == current.block_type
            and current.bbox is not None
            and block.bbox is not None
        )

        if should_merge:
            # Check vertical gap
            gap = block.bbox[1] - current.bbox[3]
            if gap < min_gap:
                # Merge blocks
                current = TextBlock(
                    text=current.text + " " + block.text,
                    block_type=current.block_type,
                    page_number=current.page_number,
                    bbox=(
                        min(current.bbox[0], block.bbox[0]),
                        current.bbox[1],
                        max(current.bbox[2], block.bbox[2]),
                        block.bbox[3],
                    ),
                )
                continue

        merged.append(current)
        current = block

    if current is not None:
        merged.append(current)

    return merged


def create_document_qa_prompt(
    document: ProcessedDocument,
    question: str,
    max_context_length: int = 4000,
) -> str:
    """
    Create a prompt for document Q&A.

    Args:
        document: Processed document.
        question: User's question.
        max_context_length: Maximum context length in characters.

    Returns:
        Formatted prompt for an LLM.

    Example:
        >>> prompt = create_document_qa_prompt(doc, "What is the main topic?")
    """
    # Get document text, truncated if needed
    text = document.text
    if len(text) > max_context_length:
        text = text[:max_context_length] + "...[truncated]"

    # Include tables if any
    table_text = ""
    if document.tables:
        table_text = "\n\nTables in document:\n"
        for i, table in enumerate(document.tables[:3]):  # Limit tables
            table_text += f"\nTable {i+1}:\n{table.to_markdown()}\n"

    prompt = f"""Document content:
{text}
{table_text}

Based on the document above, please answer the following question:
{question}

Answer:"""

    return prompt


if __name__ == "__main__":
    print("Document AI Utils - DGX Spark Optimized")
    print("=" * 50)

    # Check for OCR engines
    try:
        import pytesseract
        print("Tesseract OCR: Available")
    except ImportError:
        print("Tesseract OCR: Not installed")

    try:
        import easyocr
        print("EasyOCR: Available")
    except ImportError:
        print("EasyOCR: Not installed")

    try:
        import fitz
        print("PyMuPDF: Available")
    except ImportError:
        print("PyMuPDF: Not installed")
