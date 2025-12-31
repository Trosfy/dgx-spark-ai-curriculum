"""
Embedding utilities for RAG systems.

This module provides utilities for loading and using embedding models.

Example Usage:
    from scripts.embedding_utils import load_embedding_model, embed_documents

    model = load_embedding_model()
    embeddings = embed_documents(model, texts)
"""

from typing import List, Optional, Union
import numpy as np
import torch

from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32
) -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace embedding model optimized for DGX Spark.

    Args:
        model_name: HuggingFace model name
        device: Device to use ("cuda" or "cpu"), auto-detected if None
        normalize: Whether to L2-normalize embeddings
        batch_size: Batch size for encoding

    Returns:
        Configured HuggingFaceEmbeddings model

    Example:
        >>> model = load_embedding_model()
        >>> print(f"Using device: {model.model_kwargs['device']}")
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": normalize,
            "batch_size": batch_size
        }
    )


def embed_documents(
    model: HuggingFaceEmbeddings,
    texts: List[str],
    show_progress: bool = False
) -> np.ndarray:
    """
    Embed a list of documents.

    Args:
        model: Embedding model
        texts: List of text strings
        show_progress: Whether to show progress bar

    Returns:
        numpy array of embeddings (n_docs, embedding_dim)

    Example:
        >>> embeddings = embed_documents(model, ["Hello world", "Goodbye"])
        >>> print(f"Shape: {embeddings.shape}")
    """
    embeddings = model.embed_documents(texts)
    return np.array(embeddings)


def embed_query(
    model: HuggingFaceEmbeddings,
    query: str
) -> np.ndarray:
    """
    Embed a single query.

    Args:
        model: Embedding model
        query: Query string

    Returns:
        numpy array of query embedding (embedding_dim,)

    Example:
        >>> query_emb = embed_query(model, "What is RAG?")
        >>> print(f"Dimension: {len(query_emb)}")
    """
    embedding = model.embed_query(query)
    return np.array(embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (-1 to 1, higher is more similar)

    Example:
        >>> sim = cosine_similarity(emb1, emb2)
        >>> print(f"Similarity: {sim:.4f}")
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(
    query_emb: np.ndarray,
    doc_embs: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and all documents.

    Args:
        query_emb: Query embedding (embedding_dim,)
        doc_embs: Document embeddings (n_docs, embedding_dim)

    Returns:
        Similarity scores for each document (n_docs,)

    Example:
        >>> similarities = batch_cosine_similarity(query_emb, doc_embs)
        >>> top_idx = np.argmax(similarities)
    """
    # Normalize if not already
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)

    return np.dot(doc_norms, query_norm)


def get_embedding_dimension(model: HuggingFaceEmbeddings) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model: Embedding model

    Returns:
        Embedding dimension

    Example:
        >>> dim = get_embedding_dimension(model)
        >>> print(f"Embedding dimension: {dim}")
    """
    test_emb = model.embed_query("test")
    return len(test_emb)
