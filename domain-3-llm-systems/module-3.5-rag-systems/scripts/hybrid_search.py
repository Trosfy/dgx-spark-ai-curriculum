"""
Hybrid search utilities for RAG systems.

This module provides hybrid search combining dense and sparse retrieval.

Example Usage:
    from scripts.hybrid_search import HybridRetriever

    retriever = HybridRetriever(dense_retriever, sparse_retriever, alpha=0.5)
    results = retriever.search("query", k=5)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from langchain.schema import Document


class BM25Retriever:
    """
    BM25 sparse retrieval implementation.

    Example:
        >>> retriever = BM25Retriever(documents)
        >>> results = retriever.search("query", k=5)
    """

    def __init__(
        self,
        documents: List[Document],
        remove_stopwords: bool = True
    ):
        """
        Initialize BM25 retriever.

        Args:
            documents: List of LangChain Documents
            remove_stopwords: Whether to remove common words
        """
        try:
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
        except ImportError:
            raise ImportError("Install rank_bm25 and nltk: pip install rank_bm25 nltk")

        self.documents = documents
        self.remove_stopwords = remove_stopwords

        if remove_stopwords:
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                import nltk
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()

        self.tokenized_docs = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and optionally remove stopwords."""
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalnum() and t not in self.stop_words]

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for documents matching the query."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], scores[idx]))
        return results

    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for all documents."""
        tokenized_query = self._tokenize(query)
        return self.bm25.get_scores(tokenized_query)


class DenseRetriever:
    """
    Dense retrieval using embeddings.

    Example:
        >>> retriever = DenseRetriever(documents, embedding_model)
        >>> results = retriever.search("query", k=5)
    """

    def __init__(self, documents: List[Document], embedding_model):
        """
        Initialize dense retriever.

        Args:
            documents: List of LangChain Documents
            embedding_model: HuggingFace embedding model
        """
        self.documents = documents
        self.embedding_model = embedding_model

        texts = [doc.page_content for doc in documents]
        self.embeddings = np.array(embedding_model.embed_documents(texts))

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for semantically similar documents."""
        query_emb = np.array(self.embedding_model.embed_query(query))
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[-k:][::-1]

        return [(self.documents[idx], scores[idx]) for idx in top_indices]

    def get_scores(self, query: str) -> np.ndarray:
        """Get similarity scores for all documents."""
        query_emb = np.array(self.embedding_model.embed_query(query))
        return np.dot(self.embeddings, query_emb)


class HybridRetriever:
    """
    Hybrid retrieval combining dense and sparse methods.

    Example:
        >>> hybrid = HybridRetriever(dense, sparse, alpha=0.5)
        >>> results = hybrid.search("query", k=5)
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        alpha: float = 0.5,
        fusion_method: str = "rrf"
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Dense retriever
            sparse_retriever: Sparse (BM25) retriever
            alpha: Weight for dense (1-alpha for sparse)
            fusion_method: "rrf" or "linear"
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.documents = dense_retriever.documents

    def search(
        self,
        query: str,
        k: int = 5,
        first_stage_k: int = 50
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid search combining dense and sparse retrieval.

        Args:
            query: Search query
            k: Final number of results
            first_stage_k: Candidates from each retriever

        Returns:
            List of (Document, score) tuples
        """
        if self.fusion_method == "rrf":
            return self._rrf_search(query, k, first_stage_k)
        return self._linear_search(query, k)

    def _rrf_search(
        self,
        query: str,
        k: int,
        first_stage_k: int
    ) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion."""
        rrf_k = 60

        dense_results = self.dense.search(query, k=first_stage_k)
        sparse_results = self.sparse.search(query, k=first_stage_k)

        doc_to_id = {id(doc): i for i, doc in enumerate(self.documents)}
        rrf_scores = {}

        for rank, (doc, _) in enumerate(dense_results):
            doc_id = doc_to_id.get(id(doc), id(doc))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, {"doc": doc, "score": 0})
            rrf_scores[doc_id]["score"] += self.alpha / (rrf_k + rank + 1)

        for rank, (doc, _) in enumerate(sparse_results):
            doc_id = doc_to_id.get(id(doc), id(doc))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, {"doc": doc, "score": 0})
            rrf_scores[doc_id]["score"] += (1 - self.alpha) / (rrf_k + rank + 1)

        sorted_results = sorted(rrf_scores.values(), key=lambda x: -x["score"])
        return [(r["doc"], r["score"]) for r in sorted_results[:k]]

    def _linear_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Linear combination of normalized scores."""
        dense_scores = self.dense.get_scores(query)
        sparse_scores = self.sparse.get_scores(query)

        def normalize(scores):
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-6:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)

        dense_norm = normalize(dense_scores)
        sparse_norm = normalize(sparse_scores)
        hybrid_scores = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm

        top_indices = np.argsort(hybrid_scores)[-k:][::-1]
        return [(self.documents[idx], hybrid_scores[idx]) for idx in top_indices]
