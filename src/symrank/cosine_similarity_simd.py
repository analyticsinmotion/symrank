import numpy as np
from typing import List, Sequence, Tuple
from .symrank import cosine_similarity as _cosine_similarity_simd

def cosine_similarity_simd(
    query_vector: Sequence[float] | np.ndarray,
    candidate_vectors: Sequence[tuple[str, Sequence[float] | np.ndarray]],
    k: int = 5,
    batch_size: int | None = None,
) -> list[dict]:
    """
    Compute cosine similarity between a query vector and a list of candidate vectors. Return the top-k most similar.

    This function uses a high-performance Rust implementation (SIMD-accelerated) under the hood.

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

    # 1) Split IDs and raw vectors
    ids, vecs = zip(*candidate_vectors)
    ids = list(ids)

    # 2) Stack into one (N×D) float32 C-contiguous array
    #    Let NumPy do all the type-casting and shape checks in C.
    vectors = np.array(vecs, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D candidate array, got shape {vectors.shape}")
    vectors = np.ascontiguousarray(vectors)

    # 3) Prepare the query as a flat float32 C-contiguous array
    q = np.asarray(query_vector, dtype=np.float32).ravel()
    q = np.ascontiguousarray(q)

    # 4) Batch through Rust, collecting (id, score)
    total = vectors.shape[0]
    batch_size = batch_size or total
    all_results: List[Tuple[str, float]] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = vectors[start:end]
        # slice of a C-contiguous array is also C-contiguous along first axis
        # but to be safe:
        if not batch.flags["C_CONTIGUOUS"]:
            batch = np.ascontiguousarray(batch)

        topk = _cosine_similarity_simd(q, batch, k)
        # topk: List[(index_in_batch, score)]
        batch_ids = ids[start:end]
        all_results.extend((batch_ids[i], score) for i, score in topk)

    # 5) Final top-k across all batches
    all_results.sort(key=lambda x: x[1], reverse=True)
    topk = all_results[:k]
    return [{"id": id_, "score": score} for id_, score in topk]
