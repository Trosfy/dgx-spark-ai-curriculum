"""
Module 4.1: Multimodal AI Utilities

This package provides utility functions for working with multimodal AI models
on DGX Spark's 128GB unified memory architecture.

Submodules:
    - vlm_utils: Vision-language model utilities
    - image_generation: Image generation helpers
    - multimodal_rag: Multimodal RAG system
    - document_ai: Document processing utilities
    - audio_utils: Audio transcription utilities
"""

from . import vlm_utils
from . import image_generation
from . import multimodal_rag
from . import document_ai
from . import audio_utils

__version__ = "1.0.0"
__author__ = "Professor SPARK"
