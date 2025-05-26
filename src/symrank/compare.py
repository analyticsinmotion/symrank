import numpy as np
from typing import List, Sequence, Tuple, Union, Optional
from .symrank import cosine_similarity

def compare(
    query_vector: Union[Sequence[float], np.ndarray],
    candidate_vectors: Sequence[Tuple[str, Union[Sequence[float], np.ndarray]]],
    method: str = "cosine",
    k: int = 5,
    vector_size: int = 1536,
    batch_size: Optional[int] = None,
) -> List[dict]:
    """
    Compare a query vector to a list of candidate vectors and return the top-k most similar.

    Parameters:
        query_vector (Sequence[float] or np.ndarray): The query vector.
        candidate_vectors (Sequence[Tuple[str, Sequence[float] or np.ndarray]]): 
            A list of (doc_id, vector) pairs to compare against.
        method (str): Similarity method to use. Currently only "cosine" is supported.
        k (int): Number of top results to return.
        vector_size (int): Expected dimensionality of all vectors (default: 1536).
        batch_size (int or None): Optional batch size to process candidates in chunks.

    Returns:
        List[dict]: A list of top-k results with "id" and "score" keys, sorted by descending similarity.
    """

    if method != "cosine":
        raise ValueError(f"Only 'cosine' method is currently supported. Got: {method}")

    query_vector = _prepare_vector(query_vector, vector_size)
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)  # <-- Ensure contiguous ONCE

    ids, vectors = zip(*candidate_vectors)
    ids = list(ids)
    vectors = [_prepare_vector(vec, vector_size) for vec in vectors]

    total = len(vectors)
    batch_size = batch_size or total
    all_results = []

    # Pre-allocate the batch buffer ONCE
    batch_vectors_np = np.empty((batch_size, vector_size), dtype=np.float32)

    for start_idx in range(0, total, batch_size):
        batch_vectors = vectors[start_idx:start_idx + batch_size]
        batch_ids = ids[start_idx:start_idx + batch_size]

        # Fill the pre-allocated buffer
        batch_vectors_np[:len(batch_vectors)] = batch_vectors

        if batch_vectors_np.shape[1] != vector_size:
            raise ValueError(f"Candidate vectors must have size {vector_size}. Got {batch_vectors_np.shape[1]}")

        # Call Rust: only pass k (no use_heap)
        batch_topk = cosine_similarity(
            query_vector,
            batch_vectors_np[:len(batch_vectors)],  # Only pass the filled portion
            k,
        )

        # Map indices to IDs here
        all_results.extend([(batch_ids[i], score) for i, score in batch_topk])

    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:k]

    return [{"id": id_, "score": score} for (id_, score) in all_results]


def _prepare_vector(vec: Union[Sequence[float], np.ndarray], expected_size: int) -> np.ndarray:
    """Ensure the input vector is a 1D numpy array of expected size and type (float32)."""
    if isinstance(vec, (list, tuple)):
        vec = np.array(vec, dtype=np.float32)
    elif isinstance(vec, np.ndarray):
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32)
    else:
        raise TypeError("Vector must be a list, tuple, or np.ndarray")

    if vec.ndim != 1:
        raise ValueError(f"Vector must be 1D. Got shape {vec.shape}")

    if vec.shape[0] != expected_size:
        raise ValueError(f"Vector size mismatch: expected {expected_size}, got {vec.shape[0]}")

    return vec
