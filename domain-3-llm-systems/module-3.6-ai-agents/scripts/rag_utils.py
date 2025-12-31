"""
RAG (Retrieval-Augmented Generation) Utilities for DGX Spark

This module provides production-ready utilities for building RAG pipelines,
including document loading, chunking, embedding, and retrieval operations.

Author: Professor SPARK
Course: DGX Spark AI Curriculum - Module 3.6: AI Agents & Agentic Systems
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import json
import time

import numpy as np


@dataclass
class Document:
    """
    Represents a document with content and metadata.

    Attributes:
        content: The text content of the document
        metadata: Dictionary of metadata (source, title, etc.)
        doc_id: Unique identifier for the document

    Example:
        >>> doc = Document(
        ...     content="The DGX Spark has 128GB of unified memory.",
        ...     metadata={"source": "dgx_spark_overview.txt", "section": "Hardware"}
        ... )
        >>> print(doc.doc_id[:8])  # First 8 chars of hash
        'a3b4c5d6'
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default="")

    def __post_init__(self):
        if not self.doc_id:
            # Generate a unique ID based on content hash
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class Chunk:
    """
    Represents a chunk of a document after splitting.

    Attributes:
        content: The text content of the chunk
        metadata: Dictionary of metadata (includes parent doc info)
        chunk_id: Unique identifier for the chunk
        embedding: Optional vector embedding of the chunk
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default="")
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()


class DocumentLoader:
    """
    Load documents from various sources.

    Supports:
        - Text files (.txt)
        - Markdown files (.md)
        - JSON files (.json)
        - Directories (recursive)

    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_directory("./data/sample_documents")
        >>> print(f"Loaded {len(docs)} documents")
        Loaded 5 documents
    """

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json'}

    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the document loader.

        Args:
            encoding: Text encoding to use (default: utf-8)
        """
        self.encoding = encoding

    def load_file(self, file_path: Union[str, Path]) -> Document:
        """
        Load a single file as a Document.

        Args:
            file_path: Path to the file

        Returns:
            Document object with content and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file extension not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        with open(path, 'r', encoding=self.encoding) as f:
            content = f.read()

        metadata = {
            'source': str(path),
            'filename': path.name,
            'extension': path.suffix,
            'size_bytes': path.stat().st_size
        }

        return Document(content=content, metadata=metadata)

    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories

        Returns:
            List of Document objects

        Example:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_directory("./docs", recursive=True)
        """
        path = Path(directory)
        documents = []

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_file(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")

        return documents


class TextChunker:
    """
    Split documents into chunks for embedding and retrieval.

    Implements recursive character text splitting with overlap,
    similar to LangChain's RecursiveCharacterTextSplitter.

    Example:
        >>> chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        >>> doc = Document(content="Long text here...")
        >>> chunks = chunker.chunk_document(doc)
        >>> print(f"Created {len(chunks)} chunks")
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Optional[Callable[[str], int]] = None
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to use for splitting (in order of preference)
            length_function: Function to measure text length (default: len)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function or len

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        final_chunks = []

        # Get the appropriate separator
        separator = separators[-1]  # Default to last (empty string)
        for sep in separators:
            if sep in text:
                separator = sep
                break

        # Split using the separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits that are too small, split those too large
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = self.length_function(split)

            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if self.length_function(chunk_text) > self.chunk_size:
                        # Recursively split with next separator
                        remaining_seps = separators[separators.index(separator) + 1:]
                        if remaining_seps:
                            final_chunks.extend(self._split_text(chunk_text, remaining_seps))
                        else:
                            final_chunks.append(chunk_text)
                    else:
                        final_chunks.append(chunk_text)
                current_chunk = [split]
                current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length + self.length_function(separator)

        # Add the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            final_chunks.append(chunk_text)

        return final_chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap from previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:]

            # Prepend overlap to current chunk
            overlapped_chunk = overlap_text + current_chunk
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split the text
        chunks = self._split_text(text, self.separators)

        # Add overlap
        chunks = self._add_overlap(chunks)

        # Filter empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]

        return chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a document into Chunk objects.

        Args:
            document: Document to split

        Returns:
            List of Chunk objects with metadata
        """
        text_chunks = self.chunk_text(document.content)

        chunks = []
        for i, text in enumerate(text_chunks):
            metadata = {
                **document.metadata,
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'parent_doc_id': document.doc_id
            }
            chunks.append(Chunk(content=text, metadata=metadata))

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks


class SimpleVectorStore:
    """
    A simple in-memory vector store for prototyping.

    For production, use ChromaDB, FAISS, or Pinecone.
    This class is useful for understanding how vector stores work.

    Example:
        >>> store = SimpleVectorStore()
        >>> store.add_chunks(chunks, embeddings)
        >>> results = store.search(query_embedding, k=5)
    """

    def __init__(self):
        """Initialize an empty vector store."""
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """
        Add chunks with their embeddings to the store.

        Args:
            chunks: List of Chunk objects
            embeddings: 2D numpy array of embeddings (n_chunks x embedding_dim)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for the most similar chunks.

        Args:
            query_embedding: 1D numpy array of query embedding
            k: Number of results to return
            threshold: Optional minimum similarity threshold

        Returns:
            List of (Chunk, similarity_score) tuples, sorted by similarity
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embedding_norms = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        similarities = embedding_norms @ query_norm

        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_k_indices:
            score = float(similarities[idx])
            if threshold is None or score >= threshold:
                results.append((self.chunks[idx], score))

        return results

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(path / "embeddings.npy", self.embeddings)

        # Save chunks as JSON
        chunks_data = [
            {
                'content': c.content,
                'metadata': c.metadata,
                'chunk_id': c.chunk_id
            }
            for c in self.chunks
        ]
        with open(path / "chunks.json", 'w') as f:
            json.dump(chunks_data, f)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load a vector store from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        # Load embeddings
        self.embeddings = np.load(path / "embeddings.npy")

        # Load chunks
        with open(path / "chunks.json", 'r') as f:
            chunks_data = json.load(f)

        self.chunks = [
            Chunk(
                content=c['content'],
                metadata=c['metadata'],
                chunk_id=c['chunk_id']
            )
            for c in chunks_data
        ]


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.

    This class orchestrates the full RAG workflow:
    1. Load and chunk documents
    2. Embed chunks using an embedding model
    3. Store embeddings in a vector store
    4. Retrieve relevant chunks for a query
    5. Generate response using retrieved context

    Example:
        >>> from langchain_community.embeddings import OllamaEmbeddings
        >>> from langchain_community.llms import Ollama
        >>>
        >>> embeddings = OllamaEmbeddings(model="nomic-embed-text")
        >>> llm = Ollama(model="llama3.1:70b")
        >>>
        >>> pipeline = RAGPipeline(embeddings, llm)
        >>> pipeline.load_documents("./data/sample_documents")
        >>> response = pipeline.query("What is DGX Spark's memory capacity?")
    """

    def __init__(
        self,
        embedding_model: Any,
        llm: Any,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        retrieval_k: int = 5
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model: Model for creating embeddings (e.g., OllamaEmbeddings)
            llm: Language model for generation (e.g., Ollama)
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            retrieval_k: Number of chunks to retrieve per query
        """
        self.embedding_model = embedding_model
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrieval_k = retrieval_k

        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.vector_store = SimpleVectorStore()

        self.chunks: List[Chunk] = []

    def load_documents(
        self,
        source: Union[str, Path, List[str]],
        show_progress: bool = True
    ) -> int:
        """
        Load documents and create embeddings.

        Args:
            source: File path, directory path, or list of paths
            show_progress: Whether to show progress messages

        Returns:
            Number of chunks created
        """
        # Load documents
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file():
                documents = [self.loader.load_file(path)]
            else:
                documents = self.loader.load_directory(path)
        else:
            documents = []
            for s in source:
                documents.extend(self.loader.load_directory(s))

        if show_progress:
            print(f"Loaded {len(documents)} documents")

        # Chunk documents
        self.chunks = self.chunker.chunk_documents(documents)

        if show_progress:
            print(f"Created {len(self.chunks)} chunks")

        # Create embeddings
        if show_progress:
            print("Creating embeddings...")

        start_time = time.time()
        texts = [c.content for c in self.chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.array(embeddings)

        if show_progress:
            elapsed = time.time() - start_time
            print(f"Embeddings created in {elapsed:.2f}s")

        # Store in vector store
        self.vector_store.add_chunks(self.chunks, embeddings)

        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The query string
            k: Number of chunks to retrieve (default: self.retrieval_k)

        Returns:
            List of (Chunk, similarity_score) tuples
        """
        k = k or self.retrieval_k

        # Embed query
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding)

        # Search
        results = self.vector_store.search(query_embedding, k=k)

        return results

    def format_context(self, chunks: List[Tuple[Chunk, float]]) -> str:
        """
        Format retrieved chunks as context for the LLM.

        Args:
            chunks: List of (Chunk, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, (chunk, score) in enumerate(chunks, 1):
            source = chunk.metadata.get('filename', 'Unknown')
            context_parts.append(
                f"[Source {i}: {source} (relevance: {score:.2f})]\n{chunk.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        return_sources: bool = False
    ) -> Union[str, Tuple[str, List[Tuple[Chunk, float]]]]:
        """
        Query the RAG system.

        Args:
            question: The user's question
            k: Number of chunks to retrieve
            return_sources: Whether to return source chunks

        Returns:
            Generated response (and optionally source chunks)

        Example:
            >>> response = pipeline.query("What is DGX Spark?")
            >>> print(response)
            'DGX Spark is NVIDIA's desktop AI supercomputer with 128GB unified memory...'
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve(question, k=k)

        if not retrieved:
            response = "I couldn't find any relevant information to answer your question."
            if return_sources:
                return response, []
            return response

        # Format context
        context = self.format_context(retrieved)

        # Create prompt
        prompt = f"""Use the following context to answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

        # Generate response
        response = self.llm.invoke(prompt)

        if return_sources:
            return response, retrieved
        return response


def compute_retrieval_metrics(
    retrieved: List[Tuple[Chunk, float]],
    ground_truth_ids: List[str]
) -> Dict[str, float]:
    """
    Compute retrieval metrics (precision, recall, MRR).

    Args:
        retrieved: List of (Chunk, score) tuples from retrieval
        ground_truth_ids: List of correct chunk IDs

    Returns:
        Dictionary with precision, recall, mrr metrics

    Example:
        >>> metrics = compute_retrieval_metrics(results, ["chunk_1", "chunk_2"])
        >>> print(f"Precision: {metrics['precision']:.2f}")
    """
    retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]
    ground_truth_set = set(ground_truth_ids)

    # Precision: fraction of retrieved that are relevant
    relevant_retrieved = sum(1 for rid in retrieved_ids if rid in ground_truth_set)
    precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0

    # Recall: fraction of relevant that are retrieved
    recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0.0

    # MRR: Mean Reciprocal Rank
    mrr = 0.0
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in ground_truth_set:
            mrr = 1.0 / i
            break

    return {
        'precision': precision,
        'recall': recall,
        'mrr': mrr,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    }


if __name__ == "__main__":
    # Example usage
    print("RAG Utilities Demo")
    print("=" * 50)

    # Create sample document
    doc = Document(
        content="""
        The DGX Spark is NVIDIA's revolutionary desktop AI supercomputer.

        It features 128GB of unified LPDDR5X memory shared between
        the CPU and GPU. This eliminates the need for data transfers
        across the PCIe bus, significantly improving performance.

        The Blackwell GB10 Superchip includes 6,144 CUDA cores and
        192 5th-generation Tensor Cores. It can achieve up to 1 PFLOP
        of FP4 compute performance.
        """,
        metadata={"source": "dgx_spark_overview.txt"}
    )

    print(f"Document ID: {doc.doc_id[:16]}...")
    print(f"Document length: {len(doc.content)} characters")

    # Chunk the document
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk.content)} chars - '{chunk.content[:50]}...'")

    print("\n" + "=" * 50)
    print("RAG utilities loaded successfully!")
    print("For full functionality, use with LangChain embeddings and LLMs.")
