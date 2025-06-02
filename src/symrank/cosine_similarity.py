import numpy as np
from typing import Sequence
from .cosine_similarity_simd import cosine_similarity_simd
from .cosine_similarity_batch import cosine_similarity_batch

def cosine_similarity(
    query_vector: Sequence[float] | np.ndarray,
    candidate_vectors: Sequence[tuple[str, Sequence[float] | np.ndarray]],
    k: int = 5,
    batch_size: int | None = None,
) -> list[dict]:
    """
    Unified cosine similarity API.
    If batch_size is None or 0, uses SIMD implementation (process all at once).
    If batch_size is a positive integer (>= 1), uses batch implementation.
    """
    if batch_size in (None, 0):
        return cosine_similarity_simd(query_vector, candidate_vectors, k=k)
    else:
        return cosine_similarity_batch(query_vector, candidate_vectors, k=k, batch_size=batch_size)