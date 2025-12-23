#tests/test_cosine_similarity_matrix.py

import numpy as np
import pytest
from symrank import cosine_similarity, cosine_similarity_matrix
from typing import Any


def test_happy_path_small_matrix():
    """Basic functionality: matrix API matches expected results"""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [1.0, 0.0, 0.0],  # sim=1.0
        [0.0, 1.0, 0.0],  # sim=0.0
        [0.7, 0.7, 0.0],  # sim=0.7
    ], dtype=np.float32)
    ids = ["a", "b", "c"]

    result = cosine_similarity_matrix(query, matrix, ids, k=2)

    assert len(result) == 2
    assert result[0]["id"] == "a"
    assert pytest.approx(result[0]["score"]) == 1.0
    assert result[1]["id"] == "c"
    assert pytest.approx(result[1]["score"], abs=0.01) == 0.7


def test_equivalence_with_list_api():
    """Matrix API produces same results as list-of-tuples API"""
    query = np.random.rand(128).astype(np.float32)
    matrix = np.random.rand(100, 128).astype(np.float32)
    ids = [f"doc_{i}" for i in range(100)]

    # List-of-tuples API
    list_input = [(ids[i], matrix[i]) for i in range(100)]
    result_list = cosine_similarity(query, list_input, k=5)

    # Matrix API
    result_matrix = cosine_similarity_matrix(query, matrix, ids, k=5)

    # Should return same IDs in same order
    assert [r["id"] for r in result_list] == [r["id"] for r in result_matrix]

    # Scores should match (relaxed tolerance for cross-platform stability)
    for r1, r2 in zip(result_list, result_matrix):
        assert pytest.approx(r1["score"], rel=1e-5) == r2["score"]


def test_non_contiguous_input():
    """Non-contiguous input should work (after copy)"""
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Fortran-order (non C-contiguous)
    matrix = np.asfortranarray(np.random.rand(10, 3).astype(np.float32))
    assert not matrix.flags['C_CONTIGUOUS']

    ids = [f"doc_{i}" for i in range(10)]

    result = cosine_similarity_matrix(query, matrix, ids, k=3)
    assert len(result) == 3


def test_float64_input_converted():
    """Float64 candidate_matrix should be converted to float32"""
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float64)  # Intentionally float64
    ids = ["a", "b"]

    result = cosine_similarity_matrix(query, matrix, ids, k=1)
    assert result[0]["id"] == "a"
    assert pytest.approx(result[0]["score"]) == 1.0


def test_wrong_ids_length_raises():
    """IDs length must match matrix rows"""
    query = np.array([1.0, 2.0], dtype=np.float32)
    matrix = np.random.rand(10, 2).astype(np.float32)
    ids = ["a", "b", "c"]  # Wrong length

    with pytest.raises(ValueError, match="len\\(ids\\)=3 must match"):
        cosine_similarity_matrix(query, matrix, ids, k=5)


def test_dimension_mismatch_raises():
    """Query dimension must match matrix columns"""
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    matrix = np.random.rand(10, 5).astype(np.float32)  # Wrong D
    ids = [f"doc_{i}" for i in range(10)]

    with pytest.raises(ValueError, match="query dim 3 != candidate_matrix dim 5"):
        cosine_similarity_matrix(query, matrix, ids, k=5)


def test_empty_matrix_raises():
    """Empty candidate matrix should raise"""
    query = np.array([1.0, 2.0], dtype=np.float32)
    matrix = np.zeros((0, 2), dtype=np.float32)  # Empty
    ids = []

    with pytest.raises(ValueError, match="must not be empty"):
        cosine_similarity_matrix(query, matrix, ids, k=5)


def test_k_zero_returns_empty():
    """k=0 should return empty list"""
    query = np.array([1.0, 2.0], dtype=np.float32)
    matrix = np.random.rand(10, 2).astype(np.float32)
    ids = [f"doc_{i}" for i in range(10)]

    result = cosine_similarity_matrix(query, matrix, ids, k=0)
    assert result == []


def test_k_greater_than_n_returns_all():
    """k > N should return all candidates"""
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    ids = ["a", "b"]

    result = cosine_similarity_matrix(query, matrix, ids, k=10)
    assert len(result) == 2


def test_batch_size_equivalence():
    """Different batch sizes should produce identical results"""
    query = np.random.rand(64).astype(np.float32)
    matrix = np.random.rand(1000, 64).astype(np.float32)
    ids = [f"doc_{i}" for i in range(1000)]

    result_no_batch = cosine_similarity_matrix(query, matrix, ids, k=10, batch_size=None)
    result_batch_100 = cosine_similarity_matrix(query, matrix, ids, k=10, batch_size=100)
    result_batch_250 = cosine_similarity_matrix(query, matrix, ids, k=10, batch_size=250)

    # All should return same IDs
    ids_no_batch = [r["id"] for r in result_no_batch]
    ids_batch_100 = [r["id"] for r in result_batch_100]
    ids_batch_250 = [r["id"] for r in result_batch_250]

    assert ids_no_batch == ids_batch_100 == ids_batch_250


def test_not_ndarray_raises():
    """candidate_matrix must be ndarray"""
    query = np.array([1.0, 2.0], dtype=np.float32)
    # matrix = [[1.0, 2.0], [3.0, 4.0]]  # List, not ndarray
    matrix: Any = [[1.0, 2.0], [3.0, 4.0]]
    ids = ["a", "b"]

    with pytest.raises(TypeError, match="must be a NumPy ndarray"):
        cosine_similarity_matrix(query, matrix, ids, k=5)


def test_negative_batch_size_raises():
    """Negative batch_size should raise ValueError"""
    query = np.array([1.0, 2.0], dtype=np.float32)
    matrix = np.random.rand(10, 2).astype(np.float32)
    ids = [f"doc_{i}" for i in range(10)]

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        cosine_similarity_matrix(query, matrix, ids, k=5, batch_size=-5)


def test_large_k_with_small_batches():
    """Large k with small batch_size should not create memory bloat"""
    query = np.random.rand(64).astype(np.float32)
    matrix = np.random.rand(100, 64).astype(np.float32)
    ids = [f"doc_{i}" for i in range(100)]

    # k >> batch_size, but still works correctly
    result = cosine_similarity_matrix(query, matrix, ids, k=1000, batch_size=20)

    # Should return all 100 candidates (since k > n)
    assert len(result) == 100

    # Should be sorted by descending score
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)
