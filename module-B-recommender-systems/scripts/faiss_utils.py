"""
FAISS Utilities for Fast Approximate Nearest Neighbor Search.

This module provides functions for building and querying FAISS indices,
enabling sub-millisecond retrieval from millions of items.

Professor SPARK's Note:
    "FAISS turns O(n) search into O(log n). For a million items,
    that's the difference between 1 second and 1 millisecond.
    This is how recommendation systems scale!"

DGX Spark Advantage:
    With 128GB unified memory, you can index hundreds of millions of items
    entirely in GPU memory. The 6,144 CUDA cores make batch queries blazing fast.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import time

# Handle FAISS import (may not be installed)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-gpu")


# =============================================================================
# Index Building
# =============================================================================

def build_flat_index(
    embeddings: np.ndarray,
    use_gpu: bool = True,
    metric: str = 'cosine'
) -> 'faiss.Index':
    """
    Build exact (flat) FAISS index.

    Use for small datasets (<100K items) where exact search is fast enough.
    Exact search is always 100% accurate.

    Args:
        embeddings: Item embeddings, shape (num_items, dim)
        use_gpu: Whether to use GPU acceleration
        metric: 'cosine' or 'l2' (Euclidean)

    Returns:
        FAISS index ready for queries

    Example:
        >>> embeddings = np.random.randn(10000, 128).astype('float32')
        >>> index = build_flat_index(embeddings, use_gpu=True)
        >>> print(f"Index contains {index.ntotal} vectors")
        Index contains 10000 vectors
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")

    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    dim = embeddings.shape[1]

    # Choose metric
    if metric == 'cosine':
        # For cosine similarity, normalize vectors and use inner product
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    else:  # L2
        index = faiss.IndexFlatL2(dim)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"🚀 Using GPU-accelerated FAISS index")
        except Exception as e:
            print(f"⚠️ GPU not available, using CPU: {e}")

    index.add(embeddings)
    return index


def build_ivf_index(
    embeddings: np.ndarray,
    nlist: Optional[int] = None,
    nprobe: int = 10,
    use_gpu: bool = True,
    metric: str = 'cosine'
) -> 'faiss.Index':
    """
    Build IVF (Inverted File) index for approximate search.

    Use for medium datasets (100K - 10M items). Trades small accuracy
    loss for much faster search.

    Args:
        embeddings: Item embeddings, shape (num_items, dim)
        nlist: Number of clusters. Default: sqrt(n)
        nprobe: Number of clusters to search. Higher = more accurate but slower.
        use_gpu: Whether to use GPU acceleration
        metric: 'cosine' or 'l2'

    Returns:
        FAISS index ready for queries

    Example:
        >>> embeddings = np.random.randn(1000000, 128).astype('float32')
        >>> index = build_ivf_index(embeddings, nprobe=20)
        >>> # Search is now ~100x faster than flat index!
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")

    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    num_items, dim = embeddings.shape

    # Default nlist: sqrt of dataset size
    if nlist is None:
        nlist = int(np.sqrt(num_items))
        nlist = max(nlist, 100)  # At least 100 clusters

    # Normalize for cosine similarity
    if metric == 'cosine':
        faiss.normalize_L2(embeddings)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    # Train the index (required for IVF)
    print(f"Training IVF index with {nlist} clusters...")
    index.train(embeddings)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"🚀 Using GPU-accelerated IVF index")
        except Exception as e:
            print(f"⚠️ GPU not available, using CPU: {e}")

    index.add(embeddings)
    index.nprobe = nprobe

    return index


def build_hnsw_index(
    embeddings: np.ndarray,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int = 50,
    metric: str = 'cosine'
) -> 'faiss.Index':
    """
    Build HNSW (Hierarchical Navigable Small World) index.

    Best for large datasets (10M+ items). Extremely fast search
    with high accuracy. Uses more memory than IVF.

    Args:
        embeddings: Item embeddings, shape (num_items, dim)
        M: Number of neighbors per node. Higher = more accurate but more memory.
        ef_construction: Search depth during construction. Higher = better quality.
        ef_search: Search depth during queries. Higher = more accurate but slower.
        metric: 'cosine' or 'l2'

    Returns:
        FAISS HNSW index (CPU only, but very fast)

    Example:
        >>> embeddings = np.random.randn(10000000, 128).astype('float32')
        >>> index = build_hnsw_index(embeddings, M=32, ef_search=100)
        >>> # Sub-millisecond search on 10 million items!
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")

    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    dim = embeddings.shape[1]

    # Normalize for cosine similarity
    if metric == 'cosine':
        faiss.normalize_L2(embeddings)
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWFlat(dim, M)

    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    print(f"Building HNSW index (this may take a while for large datasets)...")
    index.add(embeddings)

    return index


def build_ivfpq_index(
    embeddings: np.ndarray,
    nlist: Optional[int] = None,
    m: int = 32,
    nbits: int = 8,
    nprobe: int = 10,
    use_gpu: bool = True,
    metric: str = 'cosine'
) -> 'faiss.Index':
    """
    Build IVF-PQ (Product Quantization) index.

    Maximum compression for very large datasets. Uses ~32x less memory
    than flat index with good accuracy.

    Args:
        embeddings: Item embeddings, shape (num_items, dim)
        nlist: Number of clusters
        m: Number of subvector groups (must divide dim evenly)
        nbits: Bits per subvector code (usually 8)
        nprobe: Clusters to search
        use_gpu: Whether to use GPU
        metric: 'cosine' or 'l2'

    Returns:
        Compressed FAISS index

    Example:
        >>> # 100 million items, 128 dimensions
        >>> # Full embeddings: 100M * 128 * 4 bytes = 51 GB
        >>> # With PQ (m=32, nbits=8): 100M * 32 bytes = 3.2 GB
        >>> embeddings = np.random.randn(100000000, 128).astype('float32')
        >>> index = build_ivfpq_index(embeddings, m=32)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not installed")

    embeddings = np.ascontiguousarray(embeddings.astype('float32'))
    num_items, dim = embeddings.shape

    if dim % m != 0:
        raise ValueError(f"dim ({dim}) must be divisible by m ({m})")

    if nlist is None:
        nlist = int(np.sqrt(num_items))
        nlist = max(nlist, 100)

    # Normalize for cosine similarity
    if metric == 'cosine':
        faiss.normalize_L2(embeddings)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits,
                                 faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    print(f"Training IVF-PQ index...")
    index.train(embeddings)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"🚀 Using GPU-accelerated IVF-PQ index")
        except Exception as e:
            print(f"⚠️ GPU not available, using CPU: {e}")

    index.add(embeddings)
    index.nprobe = nprobe

    return index


# =============================================================================
# Index Selection Helper
# =============================================================================

def build_index(
    embeddings: np.ndarray,
    use_gpu: bool = True,
    metric: str = 'cosine'
) -> 'faiss.Index':
    """
    Automatically choose and build the best index for your dataset size.

    Args:
        embeddings: Item embeddings
        use_gpu: Whether to use GPU
        metric: 'cosine' or 'l2'

    Returns:
        Optimized FAISS index

    Example:
        >>> embeddings = model.encode_items(all_items)
        >>> index = build_index(embeddings)  # Automatically picks best type
    """
    num_items = embeddings.shape[0]

    if num_items < 50_000:
        print(f"📊 Dataset size: {num_items:,} → Using Flat index (exact search)")
        return build_flat_index(embeddings, use_gpu, metric)

    elif num_items < 1_000_000:
        print(f"📊 Dataset size: {num_items:,} → Using IVF index (approximate)")
        return build_ivf_index(embeddings, use_gpu=use_gpu, metric=metric)

    elif num_items < 10_000_000:
        print(f"📊 Dataset size: {num_items:,} → Using HNSW index (very fast)")
        return build_hnsw_index(embeddings, metric=metric)

    else:
        print(f"📊 Dataset size: {num_items:,} → Using IVF-PQ index (compressed)")
        return build_ivfpq_index(embeddings, use_gpu=use_gpu, metric=metric)


# =============================================================================
# Querying
# =============================================================================

def search(
    index: 'faiss.Index',
    query_embeddings: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors.

    Args:
        index: FAISS index
        query_embeddings: Query vectors, shape (num_queries, dim)
        k: Number of neighbors to return

    Returns:
        Tuple of (indices, scores):
        - indices: shape (num_queries, k), item IDs of neighbors
        - scores: shape (num_queries, k), similarity/distance scores

    Example:
        >>> user_embedding = model.encode_query(user_features)
        >>> indices, scores = search(index, user_embedding, k=100)
        >>> print(f"Top recommendation: item {indices[0, 0]} with score {scores[0, 0]:.3f}")
    """
    query_embeddings = np.ascontiguousarray(query_embeddings.astype('float32'))

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embeddings)

    scores, indices = index.search(query_embeddings, k)

    return indices, scores


def search_with_filter(
    index: 'faiss.Index',
    query_embeddings: np.ndarray,
    k: int,
    valid_items: Optional[np.ndarray] = None,
    blocked_items: Optional[np.ndarray] = None,
    oversample_factor: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search with item filtering (e.g., exclude already-seen items).

    Args:
        index: FAISS index
        query_embeddings: Query vectors
        k: Number of results after filtering
        valid_items: Only return these items (if provided)
        blocked_items: Exclude these items
        oversample_factor: Fetch this many extra candidates for filtering

    Returns:
        Tuple of (indices, scores) after filtering
    """
    # Fetch more candidates than needed
    fetch_k = k * oversample_factor
    indices, scores = search(index, query_embeddings, fetch_k)

    # Filter results
    filtered_indices = []
    filtered_scores = []

    for query_idx in range(len(query_embeddings)):
        query_indices = indices[query_idx]
        query_scores = scores[query_idx]

        valid_mask = np.ones(len(query_indices), dtype=bool)

        if blocked_items is not None:
            blocked_set = set(blocked_items)
            valid_mask &= np.array([idx not in blocked_set for idx in query_indices])

        if valid_items is not None:
            valid_set = set(valid_items)
            valid_mask &= np.array([idx in valid_set for idx in query_indices])

        valid_indices = query_indices[valid_mask][:k]
        valid_scores = query_scores[valid_mask][:k]

        # Pad if not enough results
        if len(valid_indices) < k:
            pad_size = k - len(valid_indices)
            valid_indices = np.pad(valid_indices, (0, pad_size), constant_values=-1)
            valid_scores = np.pad(valid_scores, (0, pad_size), constant_values=-np.inf)

        filtered_indices.append(valid_indices)
        filtered_scores.append(valid_scores)

    return np.array(filtered_indices), np.array(filtered_scores)


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_index(
    index: 'faiss.Index',
    query_embeddings: np.ndarray,
    k: int = 10,
    num_trials: int = 100
) -> Dict[str, float]:
    """
    Benchmark index query performance.

    Args:
        index: FAISS index
        query_embeddings: Sample queries for benchmarking
        k: Number of neighbors to retrieve
        num_trials: Number of benchmark iterations

    Returns:
        Dictionary with timing statistics

    Example:
        >>> stats = benchmark_index(index, test_queries, k=100)
        >>> print(f"Average latency: {stats['avg_ms']:.2f} ms")
        >>> print(f"QPS: {stats['qps']:.0f}")
    """
    query_embeddings = np.ascontiguousarray(query_embeddings.astype('float32'))
    faiss.normalize_L2(query_embeddings)

    # Warmup
    for _ in range(5):
        index.search(query_embeddings[:1], k)

    # Benchmark single queries
    single_times = []
    for i in range(min(num_trials, len(query_embeddings))):
        start = time.perf_counter()
        index.search(query_embeddings[i:i+1], k)
        single_times.append((time.perf_counter() - start) * 1000)  # ms

    # Benchmark batch queries
    batch_size = min(32, len(query_embeddings))
    batch_times = []
    for _ in range(num_trials // batch_size):
        batch = query_embeddings[:batch_size]
        start = time.perf_counter()
        index.search(batch, k)
        batch_times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'avg_ms': np.mean(single_times),
        'p50_ms': np.percentile(single_times, 50),
        'p99_ms': np.percentile(single_times, 99),
        'qps': 1000 / np.mean(single_times),  # Queries per second
        'batch_avg_ms': np.mean(batch_times),
        'batch_qps': batch_size * 1000 / np.mean(batch_times),
        'num_items': index.ntotal,
    }


def print_benchmark(stats: Dict[str, float], index_type: str = "Index") -> None:
    """Pretty print benchmark results."""
    print(f"\n{'═' * 45}")
    print(f"📊 {index_type} Benchmark Results")
    print(f"{'═' * 45}")
    print(f"Index size:        {stats['num_items']:,} items")
    print(f"{'─' * 45}")
    print(f"Single query:")
    print(f"  Average:         {stats['avg_ms']:.3f} ms")
    print(f"  P50:             {stats['p50_ms']:.3f} ms")
    print(f"  P99:             {stats['p99_ms']:.3f} ms")
    print(f"  Throughput:      {stats['qps']:.0f} QPS")
    print(f"{'─' * 45}")
    print(f"Batch query (32):")
    print(f"  Average:         {stats['batch_avg_ms']:.3f} ms")
    print(f"  Throughput:      {stats['batch_qps']:.0f} QPS")
    print(f"{'═' * 45}\n")


# =============================================================================
# Index Persistence
# =============================================================================

def save_index(index: 'faiss.Index', path: str) -> None:
    """
    Save FAISS index to disk.

    Args:
        index: FAISS index
        path: File path to save

    Note:
        GPU indices are automatically converted to CPU for saving.
    """
    # Convert GPU index to CPU if needed
    if hasattr(faiss, 'index_gpu_to_cpu'):
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            cpu_index = index
    else:
        cpu_index = index

    faiss.write_index(cpu_index, path)
    print(f"💾 Saved index to {path}")


def load_index(path: str, use_gpu: bool = True) -> 'faiss.Index':
    """
    Load FAISS index from disk.

    Args:
        path: File path to load
        use_gpu: Whether to move to GPU

    Returns:
        Loaded FAISS index
    """
    index = faiss.read_index(path)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"🚀 Loaded GPU index from {path}")
        except Exception as e:
            print(f"⚠️ GPU not available, using CPU: {e}")
    else:
        print(f"📁 Loaded CPU index from {path}")

    return index


if __name__ == "__main__":
    if not FAISS_AVAILABLE:
        print("FAISS not available. Skipping tests.")
    else:
        print("Testing FAISS utilities...")

        # Create test embeddings
        np.random.seed(42)
        num_items = 10000
        dim = 128
        embeddings = np.random.randn(num_items, dim).astype('float32')

        # Build index
        index = build_flat_index(embeddings, use_gpu=False)
        print(f"✅ Built index with {index.ntotal} items")

        # Search
        query = np.random.randn(1, dim).astype('float32')
        indices, scores = search(index, query, k=5)
        print(f"✅ Search results: {indices[0]}")
        print(f"   Scores: {scores[0]}")

        # Benchmark
        test_queries = np.random.randn(100, dim).astype('float32')
        stats = benchmark_index(index, test_queries, k=10, num_trials=50)
        print_benchmark(stats, "Flat Index")

        print("\n✅ All FAISS tests passed!")
