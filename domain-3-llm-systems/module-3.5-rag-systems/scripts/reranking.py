"""
Reranking utilities for RAG systems.

This module provides cross-encoder reranking for improved retrieval quality.

Example Usage:
    from scripts.reranking import TwoStageRetriever

    retriever = TwoStageRetriever(documents, embedding_model, reranker)
    results = retriever.search("query", k=5, first_stage_k=50)
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch

from langchain.schema import Document


class CrossEncoderReranker:
    """
    Cross-encoder based reranker.

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> reranked = reranker.rerank("query", candidates, k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("cuda" or "cpu")
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(model_name, device=device)
        self.device = device

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Search query
            candidates: List of (Document, score) tuples
            k: Number of results to return

        Returns:
            Reranked list of (Document, score) tuples
        """
        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc, _ in candidates]
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        reranked = sorted(
            zip([doc for doc, _ in candidates], scores),
            key=lambda x: -x[1]
        )

        return [(doc, float(score)) for doc, score in reranked[:k]]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        return float(self.model.predict([[query, document]])[0])


class TwoStageRetriever:
    """
    Two-stage retrieval with bi-encoder + cross-encoder reranking.

    Example:
        >>> retriever = TwoStageRetriever(docs, embedding_model)
        >>> results = retriever.search("query", k=5, first_stage_k=50)
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model,
        reranker_model: str = "BAAI/bge-reranker-large"
    ):
        """
        Initialize two-stage retriever.

        Args:
            documents: List of LangChain Documents
            embedding_model: HuggingFace embedding model
            reranker_model: Cross-encoder model name
        """
        self.documents = documents
        self.embedding_model = embedding_model

        # Pre-compute embeddings
        texts = [doc.page_content for doc in documents]
        self.embeddings = np.array(embedding_model.embed_documents(texts))

        # Initialize reranker
        self.reranker = CrossEncoderReranker(reranker_model)

    def search(
        self,
        query: str,
        k: int = 5,
        first_stage_k: int = 50
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Two-stage retrieval.

        Args:
            query: Search query
            k: Final number of results
            first_stage_k: Candidates from first stage

        Returns:
            List of (Document, score, metadata) tuples
        """
        import time

        # Stage 1: Bi-encoder retrieval
        stage1_start = time.time()
        query_emb = np.array(self.embedding_model.embed_query(query))
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[-first_stage_k:][::-1]

        candidates = [(self.documents[i], similarities[i]) for i in top_indices]
        stage1_time = time.time() - stage1_start

        # Stage 2: Cross-encoder reranking
        stage2_start = time.time()
        reranked = self.reranker.rerank(query, candidates, k=k)
        stage2_time = time.time() - stage2_start

        # Build results with metadata
        results = []
        for i, (doc, rerank_score) in enumerate(reranked):
            doc_idx = self.documents.index(doc)
            bi_score = similarities[doc_idx]
            bi_rank = list(top_indices).index(doc_idx) + 1

            results.append((
                doc,
                rerank_score,
                {
                    "bi_encoder_score": float(bi_score),
                    "bi_encoder_rank": bi_rank,
                    "final_rank": i + 1,
                    "stage1_time_ms": stage1_time * 1000,
                    "stage2_time_ms": stage2_time * 1000
                }
            ))

        return results

    def search_without_reranking(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Single-stage retrieval (bi-encoder only)."""
        query_emb = np.array(self.embedding_model.embed_query(query))
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [(self.documents[i], similarities[i]) for i in top_indices]
