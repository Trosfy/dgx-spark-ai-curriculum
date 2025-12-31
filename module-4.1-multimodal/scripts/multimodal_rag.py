"""
Multimodal RAG System for DGX Spark

This module provides a complete multimodal Retrieval-Augmented Generation system
that can index and search across both images and text using CLIP embeddings
and ChromaDB vector storage.

Example:
    >>> from scripts.multimodal_rag import MultimodalRAG
    >>> rag = MultimodalRAG()
    >>> rag.add_images(["photo1.jpg", "photo2.jpg"])
    >>> rag.add_documents(["doc1.txt", "doc2.txt"])
    >>> results = rag.search("a sunset over mountains", top_k=5)
"""

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

import torch
import numpy as np
from PIL import Image


class ContentType(Enum):
    """Types of content in the multimodal index."""
    IMAGE = "image"
    TEXT = "text"
    DOCUMENT = "document"


@dataclass
class SearchResult:
    """A single search result from the multimodal RAG system."""
    content_type: ContentType
    content_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[str] = None  # Text content or image path

    def __repr__(self) -> str:
        return f"SearchResult({self.content_type.value}, score={self.score:.3f}, id={self.content_id[:8]}...)"


class MultimodalRAG:
    """
    A multimodal RAG system for searching across images and text.

    This system uses CLIP for generating embeddings and ChromaDB for
    vector storage and retrieval.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        embedding_dim: Dimension of CLIP embeddings (768 for ViT-L/14).
        device: Device for model inference.

    Example:
        >>> rag = MultimodalRAG("my_collection")
        >>> rag.add_images(["vacation/photo1.jpg", "vacation/photo2.jpg"])
        >>> rag.add_text_chunks(["Summer vacation in Italy", "Beach sunset"])
        >>> results = rag.search("italian beach", top_k=3)
        >>> for r in results:
        ...     print(f"{r.content_type.value}: {r.score:.3f}")
    """

    def __init__(
        self,
        collection_name: str = "multimodal_index",
        clip_model: str = "openai/clip-vit-large-patch14",
        persist_directory: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the multimodal RAG system.

        Args:
            collection_name: Name for the ChromaDB collection.
            clip_model: CLIP model to use for embeddings.
            persist_directory: Directory to persist the vector database.
            device: Device for model inference (auto-detected if None).
        """
        self.collection_name = collection_name
        self.clip_model_name = clip_model
        self.persist_directory = persist_directory
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._clip_model = None
        self._clip_processor = None
        self._chroma_client = None
        self._collection = None

        self.embedding_dim = 768  # CLIP ViT-L/14

    def _ensure_clip_loaded(self) -> None:
        """Lazy load CLIP model."""
        if self._clip_model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        print(f"Loading CLIP model: {self.clip_model_name}")
        self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self._clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
        self._clip_model.eval()
        print(f"  Loaded on {self.device}")

    def _ensure_collection_ready(self) -> None:
        """Lazy initialize ChromaDB collection."""
        if self._collection is not None:
            return

        import chromadb
        from chromadb.config import Settings

        print("Initializing ChromaDB...")

        if self.persist_directory:
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._chroma_client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        print(f"  Collection '{self.collection_name}' ready with {self._collection.count()} items")

    def _get_content_id(self, content: str, content_type: ContentType) -> str:
        """Generate a unique ID for content."""
        hash_input = f"{content_type.value}:{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Generate CLIP embeddings for images."""
        self._ensure_clip_loaded()

        inputs = self._clip_processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate CLIP embeddings for texts."""
        self._ensure_clip_loaded()

        inputs = self._clip_processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP max length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    def add_images(
        self,
        image_paths: List[Union[str, Path]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> int:
        """
        Add images to the index.

        Args:
            image_paths: List of image file paths.
            metadata: Optional metadata for each image.
            batch_size: Number of images to process at once.
            show_progress: Whether to show progress bar.

        Returns:
            Number of images added.

        Example:
            >>> rag.add_images(
            ...     ["photo1.jpg", "photo2.jpg"],
            ...     metadata=[{"album": "vacation"}, {"album": "vacation"}]
            ... )
        """
        self._ensure_collection_ready()

        from tqdm import tqdm

        image_paths = [Path(p) for p in image_paths]

        # Filter out non-existent files
        valid_paths = [p for p in image_paths if p.exists()]
        if len(valid_paths) < len(image_paths):
            print(f"Warning: {len(image_paths) - len(valid_paths)} images not found")

        added = 0
        iterator = range(0, len(valid_paths), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding images")

        for i in iterator:
            batch_paths = valid_paths[i:i + batch_size]

            # Load images
            images = []
            ids = []
            metas = []

            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                    ids.append(self._get_content_id(str(path), ContentType.IMAGE))
                    meta = {
                        "content_type": ContentType.IMAGE.value,
                        "path": str(path),
                        "filename": path.name,
                    }
                    if metadata and i + j < len(metadata):
                        meta.update(metadata[i + j])
                    metas.append(meta)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not images:
                continue

            # Generate embeddings
            embeddings = self._embed_images(images)

            # Add to collection
            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metas,
            )

            added += len(images)

        print(f"Added {added} images to index")
        return added

    def add_text_chunks(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> int:
        """
        Add text chunks to the index.

        Args:
            texts: List of text strings.
            metadata: Optional metadata for each text.
            batch_size: Number of texts to process at once.
            show_progress: Whether to show progress bar.

        Returns:
            Number of texts added.

        Example:
            >>> rag.add_text_chunks([
            ...     "A beautiful sunset over the ocean",
            ...     "Mountain hiking in the Alps",
            ... ])
        """
        self._ensure_collection_ready()

        from tqdm import tqdm

        added = 0
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding texts")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            ids = []
            metas = []

            for j, text in enumerate(batch_texts):
                ids.append(self._get_content_id(text, ContentType.TEXT))
                meta = {
                    "content_type": ContentType.TEXT.value,
                    "text": text[:1000],  # Store first 1000 chars
                }
                if metadata and i + j < len(metadata):
                    meta.update(metadata[i + j])
                metas.append(meta)

            # Generate embeddings
            embeddings = self._embed_texts(batch_texts)

            # Add to collection
            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metas,
            )

            added += len(batch_texts)

        print(f"Added {added} text chunks to index")
        return added

    def add_documents(
        self,
        document_paths: List[Union[str, Path]],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add documents to the index (splits into chunks).

        Args:
            document_paths: List of document file paths.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            metadata: Optional metadata for each document.

        Returns:
            Number of chunks added.

        Example:
            >>> rag.add_documents(["article.txt", "paper.md"])
        """
        all_chunks = []
        all_metadata = []

        for i, path in enumerate(document_paths):
            path = Path(path)
            if not path.exists():
                print(f"Warning: {path} not found")
                continue

            # Read document
            text = path.read_text(encoding="utf-8", errors="ignore")

            # Split into chunks
            chunks = self._split_text(text, chunk_size, chunk_overlap)

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = {
                    "content_type": ContentType.DOCUMENT.value,
                    "source": str(path),
                    "filename": path.name,
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                }
                if metadata and i < len(metadata):
                    meta.update(metadata[i])
                all_metadata.append(meta)

        if all_chunks:
            return self.add_text_chunks(all_chunks, all_metadata)
        return 0

    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks

    def search(
        self,
        query: str,
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search the index with a text query.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return.
            content_types: Filter by content type(s).
            min_score: Minimum similarity score (0-1).

        Returns:
            List of SearchResult objects sorted by relevance.

        Example:
            >>> results = rag.search("sunset beach vacation", top_k=5)
            >>> for r in results:
            ...     print(f"{r.content_type.value}: {r.score:.3f}")
        """
        self._ensure_collection_ready()

        # Generate query embedding
        query_embedding = self._embed_texts([query])[0]

        # Build where clause for filtering
        where_clause = None
        if content_types:
            type_values = [ct.value for ct in content_types]
            if len(type_values) == 1:
                where_clause = {"content_type": type_values[0]}
            else:
                where_clause = {"content_type": {"$in": type_values}}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []

        if results["ids"] and results["ids"][0]:
            for id_, meta, distance in zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Convert distance to similarity score
                # ChromaDB returns cosine distance (1 - similarity)
                score = 1 - distance

                if score < min_score:
                    continue

                content_type = ContentType(meta.get("content_type", "text"))

                # Get content based on type
                if content_type == ContentType.IMAGE:
                    content = meta.get("path")
                else:
                    content = meta.get("text")

                search_results.append(SearchResult(
                    content_type=content_type,
                    content_id=id_,
                    score=score,
                    metadata=meta,
                    content=content,
                ))

        return search_results

    def search_by_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
    ) -> List[SearchResult]:
        """
        Search the index using an image query.

        Args:
            image: Query image (path or PIL Image).
            top_k: Maximum results to return.
            content_types: Filter by content type(s).

        Returns:
            List of SearchResult objects sorted by relevance.

        Example:
            >>> # Find similar images
            >>> results = rag.search_by_image("query.jpg", top_k=5)
        """
        self._ensure_collection_ready()

        # Load and embed image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        query_embedding = self._embed_images([img])[0]

        # Build where clause
        where_clause = None
        if content_types:
            type_values = [ct.value for ct in content_types]
            if len(type_values) == 1:
                where_clause = {"content_type": type_values[0]}
            else:
                where_clause = {"content_type": {"$in": type_values}}

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "distances"],
        )

        # Convert to SearchResult objects
        search_results = []

        if results["ids"] and results["ids"][0]:
            for id_, meta, distance in zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1 - distance
                content_type = ContentType(meta.get("content_type", "text"))

                if content_type == ContentType.IMAGE:
                    content = meta.get("path")
                else:
                    content = meta.get("text")

                search_results.append(SearchResult(
                    content_type=content_type,
                    content_id=id_,
                    score=score,
                    metadata=meta,
                    content=content,
                ))

        return search_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dict with index statistics.

        Example:
            >>> stats = rag.get_stats()
            >>> print(f"Total items: {stats['total_items']}")
        """
        self._ensure_collection_ready()

        total = self._collection.count()

        # Count by type
        type_counts = {}
        for ct in ContentType:
            try:
                results = self._collection.get(
                    where={"content_type": ct.value},
                    include=[],
                )
                type_counts[ct.value] = len(results["ids"])
            except Exception:
                type_counts[ct.value] = 0

        return {
            "collection_name": self.collection_name,
            "total_items": total,
            "by_type": type_counts,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
        }

    def delete(self, ids: List[str]) -> int:
        """
        Delete items from the index by ID.

        Args:
            ids: List of content IDs to delete.

        Returns:
            Number of items deleted.
        """
        self._ensure_collection_ready()
        self._collection.delete(ids=ids)
        return len(ids)

    def clear(self) -> None:
        """Clear all items from the index."""
        self._ensure_collection_ready()
        self._chroma_client.delete_collection(self.collection_name)
        self._collection = self._chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Cleared collection '{self.collection_name}'")

    def close(self) -> None:
        """Clean up resources."""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_processor
            self._clip_model = None
            self._clip_processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


def build_rag_pipeline(
    rag: MultimodalRAG,
    vlm_model: Any,
    vlm_processor: Any,
    model_type: str = "llava",
) -> callable:
    """
    Build a RAG pipeline that retrieves relevant content and generates answers.

    Args:
        rag: MultimodalRAG instance.
        vlm_model: Vision-language model.
        vlm_processor: VLM processor.
        model_type: Type of VLM ("llava" or "qwen").

    Returns:
        Callable pipeline function.

    Example:
        >>> pipeline = build_rag_pipeline(rag, model, processor)
        >>> answer = pipeline("What does the beach sunset look like?")
    """
    from .vlm_utils import analyze_image_llava, analyze_image_qwen

    analyze_fn = analyze_image_llava if model_type == "llava" else analyze_image_qwen

    def pipeline(
        query: str,
        top_k: int = 3,
        include_images: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the RAG pipeline.

        Args:
            query: User query.
            top_k: Number of results to retrieve.
            include_images: Whether to include image analysis.

        Returns:
            Dict with answer and sources.
        """
        # Retrieve relevant content
        results = rag.search(query, top_k=top_k)

        if not results:
            return {
                "answer": "No relevant content found.",
                "sources": [],
            }

        # Build context
        context_parts = []
        sources = []
        image_analyses = []

        for r in results:
            if r.content_type == ContentType.IMAGE and include_images:
                # Analyze image
                try:
                    analysis = analyze_fn(
                        vlm_model, vlm_processor,
                        r.content,
                        f"Describe this image in detail, focusing on aspects relevant to: {query}"
                    )
                    image_analyses.append(analysis)
                    context_parts.append(f"[Image: {r.metadata.get('filename', 'unknown')}]\n{analysis}")
                except Exception as e:
                    context_parts.append(f"[Image: {r.metadata.get('filename', 'unknown')} - could not analyze]")
            else:
                context_parts.append(r.content or "")

            sources.append({
                "type": r.content_type.value,
                "score": r.score,
                "content": r.content,
                "metadata": r.metadata,
            })

        context = "\n\n".join(context_parts)

        # Generate answer (simplified - in production, use an LLM)
        answer = f"Based on {len(results)} retrieved items:\n\n{context}"

        return {
            "answer": answer,
            "sources": sources,
            "image_analyses": image_analyses,
        }

    return pipeline


if __name__ == "__main__":
    print("Multimodal RAG System - DGX Spark Optimized")
    print("=" * 50)

    # Quick test
    rag = MultimodalRAG("test_collection")
    print(f"Created RAG system")

    stats = rag.get_stats()
    print(f"Stats: {stats}")
