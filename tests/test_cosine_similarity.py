import numpy as np
import pytest
from symrank import cosine_similarity

# ------------------------------------------------------------
# Test 1
# Basic test with small candidate set (default sort path)
# ------------------------------------------------------------
def test_compare_default_threshold():
    query_vector = np.random.rand(1536).astype(np.float32)
    candidate_vectors = [
        (f"doc_{i}", np.random.rand(1536).astype(np.float32))
        for i in range(300)
    ]

    top_results = cosine_similarity(query_vector, candidate_vectors)
    assert len(top_results) == 5
    assert all("id" in result and "score" in result for result in top_results)

# ------------------------------------------------------------
# Test 2
# Invalid vector shape (e.g., wrong number of dimensions)
# ------------------------------------------------------------
def test_invalid_vector_shape():
    query_vector = np.random.rand(1536).astype(np.float32)
    candidate_vectors = [
        ("doc_1", np.random.rand(1536, 1).astype(np.float32))  # Invalid shape (2D)
    ]

    with pytest.raises(ValueError):
        cosine_similarity(query_vector, candidate_vectors)

# ------------------------------------------------------------
# Test 3
# Mismatched vector size (declared vector_size vs actual vector length)
# ------------------------------------------------------------
def test_mismatched_vector_size():
    query_vector = np.random.rand(1536).astype(np.float32)
    candidate_vectors = [
        ("doc_1", np.random.rand(512).astype(np.float32))  # Wrong size
    ]

    with pytest.raises(ValueError):
        cosine_similarity(query_vector, candidate_vectors)

# ------------------------------------------------------------
# Test 4
# Non-list, non-numpy query vector (invalid input type)
# ------------------------------------------------------------
def test_invalid_query_vector_type():
    query_vector = "this is not a vector"
    candidate_vectors = [
        ("doc_1", np.random.rand(1536).astype(np.float32))
    ]

    with pytest.raises(ValueError):
        cosine_similarity(query_vector, candidate_vectors)  # type: ignore

# ------------------------------------------------------------
# Test 5
# Empty candidate_vectors should raise ValueError
# ------------------------------------------------------------
def test_empty_candidate_vectors():
    query = np.random.rand(16).astype(np.float32)
    with pytest.raises(ValueError):
        cosine_similarity(query, [])

# ------------------------------------------------------------
# Test 6
# k = 0 should return an empty list
# ------------------------------------------------------------
def test_k_zero_returns_empty_list():
    query = np.random.rand(8).astype(np.float32)
    candidates = [(f"id_{i}", np.random.rand(8).astype(np.float32)) for i in range(5)]
    result = cosine_similarity(query, candidates, k=0)
    assert result == []

# ------------------------------------------------------------
# Test 7
# k greater than candidate count returns all candidates sorted by similarity
# ------------------------------------------------------------
def test_k_greater_than_candidate_count():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = [
        ("a", [1.0, 0.0]),  # Similarity 1.0
        ("b", [0.0, 1.0])   # Similarity 0.0
    ]
    result = cosine_similarity(query, candidates, k=5)
    assert len(result) == 2
    assert result[0]["id"] == "a" and pytest.approx(result[0]["score"]) == 1.0
    assert result[1]["id"] == "b" and pytest.approx(result[1]["score"]) == 0.0

# ------------------------------------------------------------
# Test 8
# Identical vectors should have a cosine similarity of 1.0
# ------------------------------------------------------------
def test_identical_vectors_score_one():
    query = [0.5, -0.5, 2.0]
    candidates = [
        ("same", [0.5, -0.5, 2.0]),  # identical
        ("other", [2.0, 2.0, 2.0])
    ]
    result = cosine_similarity(query, candidates, k=1)
    assert result[0]["id"] == "same"
    assert pytest.approx(result[0]["score"], rel=1e-6) == 1.0

# ------------------------------------------------------------
# Test 9
# Sequence inputs of ints should be accepted
# ------------------------------------------------------------
def test_integer_sequence_input():
    query = [1, 2, 3]
    candidates = [
        ("ints", [3, 2, 1]),
        ("floats", [1.0, 2.0, 3.0])
    ]
    result = cosine_similarity(query, candidates, k=2)
    assert {r["id"] for r in result} == {"ints", "floats"}

# ------------------------------------------------------------
# Test 10
# Negative and mixed values ordering should be correct
# ------------------------------------------------------------
def test_negative_and_mixed_values_ordering():
    query = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    candidates = [
        ("pos", [1.0, 1.0, 0.0]),
        ("neg", [-1.0, 1.0, 0.0]),
        ("zero", [0.0, 0.0, 0.0])
    ]
    result = cosine_similarity(query, candidates, k=3)
    scores = {item['id']: item['score'] for item in result}

    # positive similarity (0.0) should be greater than negative (-1.0)
    assert scores['pos'] >= scores['neg']
    assert scores['zero'] == pytest.approx(0.0)

  

# ------------------------------------------------------------
# Entry point for running this file standalone
# ------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-v"])
