"""
Module 4.1: Multimodal AI - Utility Scripts

This package provides reusable utilities for multimodal AI tasks:
- Vision-Language Models (VLMs)
- Image Generation
- Multimodal RAG
- Document AI
- Audio Transcription

Example usage:
    from scripts.vlm_utils import VLMPipeline
    from scripts.image_generation import ImageGenerator
    from scripts.multimodal_rag import MultimodalRAG
    from scripts.document_ai import DocumentProcessor
    from scripts.audio_utils import AudioTranscriber
"""

from .vlm_utils import VLMPipeline, load_image_from_url, clear_gpu_memory
from .image_generation import ImageGenerator, create_image_grid
from .multimodal_rag import MultimodalRAG, CLIPEmbedder
from .document_ai import DocumentProcessor, create_sample_invoice
from .audio_utils import AudioTranscriber, load_audio, save_audio

__all__ = [
    # VLM
    "VLMPipeline",
    "load_image_from_url",
    "clear_gpu_memory",
    # Image Generation
    "ImageGenerator",
    "create_image_grid",
    # Multimodal RAG
    "MultimodalRAG",
    "CLIPEmbedder",
    # Document AI
    "DocumentProcessor",
    "create_sample_invoice",
    # Audio
    "AudioTranscriber",
    "load_audio",
    "save_audio",
]

__version__ = "1.0.0"
