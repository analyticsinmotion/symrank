# src/symrank/cosine_similarity_matrix.py

import numpy as np
from typing import Sequence
from .symrank import cosine_similarity as _cosine_similarity


def cosine_similarity_matrix(
    query: Sequence[float] | np.ndarray,
    candidate_matrix: np.ndarray,
    ids: Sequence[str],
    k: int = 5,
    batch_size: int | None = None,
) -> list[dict]:
    """
    Compute cosine similarity using a pre-built candidate matrix.

    Optimized for cases where candidates are already stored as a single
    NumPy array, avoiding Python object iteration overhead.

    Args:
        query: Query vector (D,) - will be converted to float32
        candidate_matrix: Candidate vectors (N, D) - will be converted to float32 C-contiguous
        ids: Document IDs, must match N
        k: Number of top results to return (returns empty list if k <= 0)
        batch_size: Optional batch size for memory control (must be positive or None)

    Returns:
        List of dicts with 'id' and 'score', sorted by descending similarity

    Raises:
        ValueError: If shapes don't match, matrix is empty, or batch_size is invalid
        TypeError: If candidate_matrix is not ndarray

    Examples:
        >>> query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        >>> ids = ["doc1", "doc2"]
        >>> cosine_similarity_matrix(query, matrix, ids, k=1)
        [{'id': 'doc1', 'score': 1.0}]
    """
    # Short-circuit for non-positive k
    if k <= 0:
        return []

    # Validate candidate_matrix type and shape
    if not isinstance(candidate_matrix, np.ndarray):
        raise TypeError("candidate_matrix must be a NumPy ndarray")

    if candidate_matrix.ndim != 2:
        raise ValueError(f"candidate_matrix must be 2D, got shape {candidate_matrix.shape}")

    n, d = candidate_matrix.shape

    if n == 0:
        raise ValueError("candidate_matrix must not be empty")

    if len(ids) != n:
        raise ValueError(f"len(ids)={len(ids)} must match candidate_matrix.shape[0]={n}")

    # Validate batch_size
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None")

    # Prepare query: single conversion pass (list/tuple → ndarray → float32 → 1D → C-contiguous)
    q = np.asarray(query, dtype=np.float32).ravel()

    if q.shape[0] != d:
        raise ValueError(f"query dim {q.shape[0]} != candidate_matrix dim {d}")

    q = np.ascontiguousarray(q)

    # Prepare candidate_matrix: single conversion pass (dtype + contiguity)
    mat = np.ascontiguousarray(candidate_matrix, dtype=np.float32)

    # Batch processing with zero-copy slices
    bs = batch_size or n
    ids_list = list(ids)
    all_results: list[tuple[str, float]] = []

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = mat[start:end]  # Zero-copy view

        # Cap k per batch to prevent memory bloat when k > n and batching
        batch_k = min(k, end - start)

        # Call Rust: returns [(batch_idx, score), ...]
        batch_topk = _cosine_similarity(q, batch, batch_k)

        # Map batch indices to global IDs
        for i, score in batch_topk:
            all_results.append((ids_list[start + i], float(score)))

    # Final top-k selection across all batches
    all_results.sort(key=lambda x: x[1], reverse=True)
    all_results = all_results[:min(k, len(all_results))]

    return [{"id": id_, "score": score} for id_, score in all_results]
