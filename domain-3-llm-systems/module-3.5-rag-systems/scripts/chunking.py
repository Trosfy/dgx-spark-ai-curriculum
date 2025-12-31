"""
Chunking utilities for RAG systems.

This module provides various chunking strategies for document processing.

Example Usage:
    from scripts.chunking import create_chunks, ChunkingStrategy

    # Fixed-size chunking
    chunks = create_chunks(documents, ChunkingStrategy.FIXED_512)

    # Semantic chunking
    chunks = create_chunks(documents, ChunkingStrategy.SEMANTIC)
"""

from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter
)
from langchain.schema import Document


class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_256 = "fixed_256"
    FIXED_512 = "fixed_512"
    FIXED_1024 = "fixed_1024"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = None

    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


def create_fixed_size_chunks(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Create fixed-size chunks from documents.

    Args:
        documents: List of LangChain Documents
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Documents

    Example:
        >>> docs = [Document(page_content="Long text here...")]
        >>> chunks = create_fixed_size_chunks(docs, chunk_size=256)
        >>> print(f"Created {len(chunks)} chunks")
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def create_semantic_chunks(
    documents: List[Document],
    max_chunk_size: int = 1500
) -> List[Document]:
    """
    Create semantic chunks based on markdown headers.

    Args:
        documents: List of LangChain Documents
        max_chunk_size: Maximum size before secondary splitting

    Returns:
        List of semantically chunked Documents

    Example:
        >>> docs = [Document(page_content="# Header\\n\\nContent...")]
        >>> chunks = create_semantic_chunks(docs)
    """
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100
    )

    all_chunks = []

    for doc in documents:
        header_chunks = markdown_splitter.split_text(doc.page_content)

        for chunk in header_chunks:
            header_context = ""
            for key in ['header_1', 'header_2', 'header_3']:
                if key in chunk.metadata:
                    level = int(key[-1])
                    header_context += "#" * level + " " + chunk.metadata[key] + "\n"

            content = header_context + chunk.page_content

            if len(content) > max_chunk_size:
                sub_chunks = size_splitter.split_text(content)
                for i, sub in enumerate(sub_chunks):
                    all_chunks.append(Document(
                        page_content=sub,
                        metadata={
                            "source": doc.metadata.get("source", "unknown"),
                            **chunk.metadata,
                            "sub_chunk": i
                        }
                    ))
            else:
                all_chunks.append(Document(
                    page_content=content,
                    metadata={
                        "source": doc.metadata.get("source", "unknown"),
                        **chunk.metadata
                    }
                ))

    return all_chunks


def create_sentence_chunks(
    documents: List[Document],
    sentences_per_chunk: int = 5,
    sentence_overlap: int = 1
) -> List[Document]:
    """
    Create chunks based on sentence boundaries.

    Args:
        documents: List of LangChain Documents
        sentences_per_chunk: Number of sentences per chunk
        sentence_overlap: Overlap in sentences

    Returns:
        List of sentence-based chunks

    Example:
        >>> docs = [Document(page_content="Sentence one. Sentence two.")]
        >>> chunks = create_sentence_chunks(docs, sentences_per_chunk=3)
    """
    try:
        from nltk.tokenize import sent_tokenize
        # Ensure punkt data is available
        try:
            sent_tokenize("test")
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
    except ImportError:
        raise ImportError("Please install nltk: pip install nltk")

    all_chunks = []

    for doc in documents:
        sentences = sent_tokenize(doc.page_content)

        i = 0
        chunk_idx = 0
        while i < len(sentences):
            end = min(i + sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end]
            chunk_text = " ".join(chunk_sentences)

            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_idx": chunk_idx,
                    "sentence_count": len(chunk_sentences)
                }
            ))

            i += sentences_per_chunk - sentence_overlap
            chunk_idx += 1

    return all_chunks


def create_chunks(
    documents: List[Document],
    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_512,
    **kwargs
) -> List[Document]:
    """
    Create chunks using the specified strategy.

    Args:
        documents: List of LangChain Documents
        strategy: ChunkingStrategy enum value
        **kwargs: Additional arguments for the chunking function

    Returns:
        List of chunked Documents

    Example:
        >>> from scripts.chunking import create_chunks, ChunkingStrategy
        >>> chunks = create_chunks(docs, ChunkingStrategy.FIXED_512)
    """
    strategy_map = {
        ChunkingStrategy.FIXED_256: lambda d: create_fixed_size_chunks(d, 256, 25),
        ChunkingStrategy.FIXED_512: lambda d: create_fixed_size_chunks(d, 512, 50),
        ChunkingStrategy.FIXED_1024: lambda d: create_fixed_size_chunks(d, 1024, 100),
        ChunkingStrategy.SEMANTIC: create_semantic_chunks,
        ChunkingStrategy.SENTENCE: create_sentence_chunks,
    }

    if strategy not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy}")

    return strategy_map[strategy](documents, **kwargs)
