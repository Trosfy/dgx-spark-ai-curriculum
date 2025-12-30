"""
Multimodal RAG Utilities

This module provides utilities for building Multimodal Retrieval-Augmented
Generation systems using CLIP embeddings and vector databases.

Example usage:
    from multimodal_rag import MultimodalRAG

    # Initialize
    rag = MultimodalRAG()
    rag.load_clip()

    # Index images
    for image_path in image_paths:
        image = Image.open(image_path)
        rag.add_image(image, metadata={"path": image_path})

    # Search with text
    results = rag.search("a red sports car")

    # Clean up
    rag.cleanup()
"""

import torch
import gc
import time
import base64
import numpy as np
from io import BytesIO
from typing import Optional, List, Dict, Tuple, Union
from PIL import Image


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage() -> str:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        return f"Allocated: {allocated:.2f}GB"
    return "No GPU"


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image

    Returns:
        Base64 encoded string

    Example:
        >>> b64 = image_to_base64(image)
        >>> recovered = base64_to_image(b64)
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Convert base64 string back to PIL Image.

    Args:
        b64_string: Base64 encoded image string

    Returns:
        PIL Image
    """
    return Image.open(BytesIO(base64.b64decode(b64_string)))


class CLIPEmbedder:
    """
    CLIP Embedding Generator.

    Creates joint embeddings for images and text in a shared space.

    Attributes:
        model: Loaded CLIP model
        processor: CLIP processor

    Example:
        >>> embedder = CLIPEmbedder()
        >>> embedder.load()
        >>> img_emb = embedder.embed_image(image)
        >>> text_emb = embedder.embed_text("a cat")
        >>> similarity = np.dot(img_emb, text_emb)
    """

    MODELS = {
        "base": "openai/clip-vit-base-patch32",
        "large": "openai/clip-vit-large-patch14",
        "huge": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    }

    def __init__(self, model_size: str = "large"):
        """
        Initialize CLIP Embedder.

        Args:
            model_size: Model size ('base', 'large', 'huge')
        """
        if model_size not in self.MODELS:
            raise ValueError(f"Unsupported model: {model_size}")

        self.model_size = model_size
        self.model_id = self.MODELS[model_size]
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        """Load CLIP model."""
        if self._loaded:
            return

        from transformers import CLIPProcessor, CLIPModel

        print(f"Loading CLIP ({self.model_size})...")
        start_time = time.time()

        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16  # Use bfloat16 for Blackwell optimization
        ).to("cuda")

        self.model.eval()

        load_time = time.time() - start_time
        self._loaded = True
        print(f"Loaded in {load_time:.1f}s")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Get CLIP embedding for an image.

        Args:
            image: PIL Image

        Returns:
            Normalized embedding as numpy array (768,)
        """
        if not self._loaded:
            self.load()

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()[0]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Get CLIP embedding for text.

        Args:
            text: Text string

        Returns:
            Normalized embedding as numpy array (768,)
        """
        if not self._loaded:
            self.load()

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()[0]

    def embed_images_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Get CLIP embeddings for a batch of images.

        Args:
            images: List of PIL Images
            batch_size: Processing batch size

        Returns:
            Array of embeddings (N, 768)
        """
        if not self._loaded:
            self.load()

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())

        return np.vstack(all_embeddings)

    def compute_similarity(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between image and text embeddings.

        Args:
            image_embedding: Image embedding
            text_embedding: Text embedding

        Returns:
            Similarity score (-1 to 1)
        """
        return float(np.dot(image_embedding, text_embedding))

    def cleanup(self) -> None:
        """Release model from memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            clear_gpu_memory()


class MultimodalRAG:
    """
    Multimodal Retrieval-Augmented Generation System.

    Combines CLIP embeddings with a vector database for searching
    images using natural language queries.

    Attributes:
        embedder: CLIP embedder
        collection: Vector database collection

    Example:
        >>> rag = MultimodalRAG()
        >>> rag.load_clip()
        >>> rag.add_image(img1, {"label": "cat"})
        >>> rag.add_image(img2, {"label": "dog"})
        >>> results = rag.search("a fluffy pet")
    """

    def __init__(self, collection_name: str = "multimodal_rag"):
        """
        Initialize Multimodal RAG.

        Args:
            collection_name: Name for the vector database collection
        """
        self.collection_name = collection_name
        self.embedder = CLIPEmbedder()
        self.collection = None
        self.vlm = None
        self._id_counter = 0

    def load_clip(self, model_size: str = "large") -> None:
        """
        Load CLIP for embeddings.

        Args:
            model_size: CLIP model size
        """
        self.embedder = CLIPEmbedder(model_size)
        self.embedder.load()
        self._setup_collection()

    def _setup_collection(self) -> None:
        """Setup ChromaDB collection."""
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Delete if exists
        try:
            client.delete_collection(self.collection_name)
        except ValueError:
            pass  # Collection doesn't exist

        self.collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_image(
        self,
        image: Image.Image,
        metadata: Optional[Dict] = None,
        image_id: Optional[str] = None
    ) -> str:
        """
        Add an image to the index.

        Args:
            image: PIL Image to add
            metadata: Optional metadata dictionary
            image_id: Optional custom ID

        Returns:
            ID of the added image
        """
        if image_id is None:
            image_id = f"img_{self._id_counter}"
            self._id_counter += 1

        # Get embedding
        embedding = self.embedder.embed_image(image)

        # Prepare metadata
        meta = metadata.copy() if metadata else {}
        meta["image_b64"] = image_to_base64(image)

        # Add to collection
        self.collection.add(
            ids=[image_id],
            embeddings=[embedding.tolist()],
            metadatas=[meta]
        )

        return image_id

    def add_images(
        self,
        images: List[Image.Image],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        Add multiple images to the index.

        Args:
            images: List of PIL Images
            metadata_list: Optional list of metadata dicts

        Returns:
            List of image IDs
        """
        if metadata_list is None:
            metadata_list = [{}] * len(images)

        ids = []
        for image, meta in zip(images, metadata_list):
            img_id = self.add_image(image, meta)
            ids.append(img_id)

        return ids

    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search images using a text query.

        Args:
            query: Natural language search query
            n_results: Number of results to return

        Returns:
            List of results with image, metadata, and similarity
        """
        query_embedding = self.embedder.embed_text(query)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['metadatas', 'distances']
        )

        formatted = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i].copy()
            image_b64 = meta.pop('image_b64', None)

            result = {
                'id': results['ids'][0][i],
                'similarity': 1 - results['distances'][0][i],
                'metadata': meta,
            }

            if image_b64:
                result['image'] = base64_to_image(image_b64)

            formatted.append(result)

        return formatted

    def search_by_image(
        self,
        query_image: Image.Image,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for similar images.

        Args:
            query_image: Image to search with
            n_results: Number of results

        Returns:
            List of similar images with metadata
        """
        query_embedding = self.embedder.embed_image(query_image)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['metadatas', 'distances']
        )

        formatted = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i].copy()
            image_b64 = meta.pop('image_b64', None)

            result = {
                'id': results['ids'][0][i],
                'similarity': 1 - results['distances'][0][i],
                'metadata': meta,
            }

            if image_b64:
                result['image'] = base64_to_image(image_b64)

            formatted.append(result)

        return formatted

    def count(self) -> int:
        """Get number of images in the index."""
        return self.collection.count()

    def cleanup(self) -> None:
        """Release resources."""
        self.embedder.cleanup()
        if self.vlm is not None:
            self.vlm.cleanup()


if __name__ == "__main__":
    print("Multimodal RAG Demo")
    print("=" * 50)

    # Create simple test images
    colors = ['red', 'blue', 'green', 'yellow']
    images = [Image.new('RGB', (224, 224), color=c) for c in colors]

    # Initialize RAG
    rag = MultimodalRAG()
    rag.load_clip()

    # Add images
    for img, color in zip(images, colors):
        rag.add_image(img, {"color": color})

    print(f"Indexed {rag.count()} images")

    # Search
    results = rag.search("something warm colored")
    print("\nSearch results for 'something warm colored':")
    for r in results:
        print(f"  - {r['metadata'].get('color')}: {r['similarity']:.3f}")

    rag.cleanup()
