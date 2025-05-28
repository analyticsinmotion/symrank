import numpy as np
from typing import List, Sequence, Union
from .symrank import cosine_similarity as _cosine_similarity

def cosine_similarity_batch(
    query_vector: Sequence[float] | np.ndarray,
    candidate_vectors: Sequence[tuple[str, Sequence[float] | np.ndarray]],
    k: int = 5,
    batch_size: int | None = None,
) -> List[dict]:
    """
    Compute cosine similarity between a query vector and a list of candidate vectors in batches. 
    Return the top-k most similar. This function uses a high-performance Rust implementation (SIMD-accelerated) 
    under the hood. This version is optimized for memory efficiency with large candidate sets by processing in batches.

    Parameters:
        query_vector (Sequence[float] or np.ndarray): The query vector.
        candidate_vectors (Sequence[Tuple[str, Sequence[float] or np.ndarray]]):
            A list of (doc_id, vector) pairs to compare against.
        k (int): Number of top results to return.
        batch_size (int or None): Optional batch size to process candidates in chunks.

    Returns:
        List[dict]: A list of top-k results with "id" and "score" keys, sorted by descending similarity.
    """
    if not candidate_vectors:
        raise ValueError("candidate_vectors must not be empty")
    
    query_vector = _prepare_vector(query_vector)
    vector_size = query_vector.shape[0]
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    ids, vectors = zip(*candidate_vectors)
    ids = list(ids)
    vectors = [_prepare_vector(vec) for vec in vectors]
    for i, vec in enumerate(vectors):
        if vec.shape[0] != vector_size:
            raise ValueError(f"Candidate vector at index {i} has dimension {vec.shape[0]}, expected {vector_size}")

    total = len(vectors)
    batch_size = batch_size or total
    all_results = []

    # Pre-allocate the batch buffer ONCE for efficiency
    batch_vectors_np = np.empty((batch_size, vector_size), dtype=np.float32)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_vectors = vectors[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        # Fill the pre-allocated buffer
        batch_vectors_np[:len(batch_vectors)] = batch_vectors

        # Call Rust extension (returns top-k for this batch, sorted)
        batch_topk = _cosine_similarity(
            query_vector,
            batch_vectors_np[:len(batch_vectors)],
            k,
        )

        # Map indices to IDs for this batch
        all_results.extend([(batch_ids[i], score) for i, score in batch_topk])

    # Final top-k selection across all batches
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:k]
    return [{"id": id_, "score": score} for (id_, score) in all_results]


def _prepare_vector(vec: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Ensure the input vector is a 1D numpy array of type float32."""
    if isinstance(vec, (list, tuple)):
        vec = np.array(vec, dtype=np.float32)
    elif isinstance(vec, np.ndarray):
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
    else:
        raise TypeError("Vector must be a list, tuple, or np.ndarray")
    if vec.ndim != 1:
        raise ValueError(f"Vector must be 1D. Got shape {vec.shape}")
    return vec
