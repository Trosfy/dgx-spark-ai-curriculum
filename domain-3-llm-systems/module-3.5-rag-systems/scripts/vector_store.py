"""
Vector store utilities for RAG systems.

This module provides utilities for creating and managing vector stores.

Example Usage:
    from scripts.vector_store import create_chroma_store, create_faiss_store

    vectorstore = create_chroma_store(chunks, embedding_model)
"""

from typing import List, Optional
from pathlib import Path
import shutil

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_chroma_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "documents",
    overwrite: bool = True
) -> Chroma:
    """
    Create a ChromaDB vector store from documents.

    Args:
        documents: List of LangChain Documents
        embedding_model: Embedding model to use
        persist_directory: Directory to persist the database
        collection_name: Name for the collection
        overwrite: Whether to overwrite existing database

    Returns:
        Chroma vector store

    Example:
        >>> store = create_chroma_store(chunks, model)
        >>> results = store.similarity_search("query", k=5)
    """
    if overwrite and Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )


def load_vector_store(
    persist_directory: str,
    embedding_model: HuggingFaceEmbeddings,
    collection_name: str = "documents"
) -> Chroma:
    """
    Load an existing ChromaDB vector store.

    Args:
        persist_directory: Directory where database is stored
        embedding_model: Embedding model (must match original)
        collection_name: Name of the collection

    Returns:
        Chroma vector store

    Example:
        >>> store = load_vector_store("./chroma_db", model)
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )


def create_faiss_store(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    use_gpu: bool = True
) -> "FAISS":
    """
    Create a FAISS vector store with optional GPU acceleration.

    Args:
        documents: List of LangChain Documents
        embedding_model: Embedding model to use
        use_gpu: Whether to use GPU (requires faiss-gpu)

    Returns:
        FAISS vector store

    Example:
        >>> store = create_faiss_store(chunks, model, use_gpu=True)
        >>> results = store.similarity_search("query", k=5)
    """
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        raise ImportError("Please install faiss: pip install faiss-gpu")

    return FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )


def search_with_score(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    filter_dict: Optional[dict] = None
) -> List[tuple]:
    """
    Search vector store and return results with scores.

    Args:
        vectorstore: Vector store to search
        query: Search query
        k: Number of results
        filter_dict: Optional metadata filter

    Returns:
        List of (Document, score) tuples

    Example:
        >>> results = search_with_score(store, "query", k=5)
        >>> for doc, score in results:
        ...     print(f"{score:.4f}: {doc.page_content[:50]}")
    """
    if filter_dict:
        return vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
    return vectorstore.similarity_search_with_score(query, k=k)
