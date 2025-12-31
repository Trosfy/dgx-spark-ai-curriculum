"""
RAG Systems Utilities - Module 3.5

This package provides reusable components for building RAG systems:
- Chunking strategies
- Embedding utilities
- Vector store management
- Hybrid search
- Reranking pipelines
- Evaluation metrics

Example Usage:
    from scripts.chunking import ChunkingStrategy, create_chunks
    from scripts.hybrid_search import HybridRetriever
    from scripts.evaluation import RAGEvaluator
"""

from .chunking import (
    ChunkingStrategy,
    create_chunks,
    create_fixed_size_chunks,
    create_semantic_chunks,
    create_sentence_chunks
)

from .embedding_utils import (
    load_embedding_model,
    embed_documents,
    embed_query
)

from .vector_store import (
    create_chroma_store,
    create_faiss_store,
    load_vector_store,
    search_with_score
)

from .hybrid_search import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever
)

from .reranking import (
    CrossEncoderReranker,
    TwoStageRetriever
)

from .evaluation import (
    EvaluationSample,
    EvaluationResult,
    RAGEvaluator,
    calculate_retrieval_metrics
)

__version__ = "1.0.0"
__all__ = [
    # Chunking
    "ChunkingStrategy",
    "create_chunks",
    "create_fixed_size_chunks",
    "create_semantic_chunks",
    "create_sentence_chunks",
    # Embeddings
    "load_embedding_model",
    "embed_documents",
    "embed_query",
    # Vector stores
    "create_chroma_store",
    "create_faiss_store",
    "load_vector_store",
    "search_with_score",
    # Hybrid search
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    # Reranking
    "CrossEncoderReranker",
    "TwoStageRetriever",
    # Evaluation
    "EvaluationSample",
    "EvaluationResult",
    "RAGEvaluator",
    "calculate_retrieval_metrics",
]
