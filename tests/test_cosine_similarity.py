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
# Batch processing with a large candidate set
# ------------------------------------------------------------
def test_compare_batching_large():
    query_vector = np.random.rand(1536).astype(np.float32)
    candidate_vectors = [
        (f"doc_{i}", np.random.rand(1536).astype(np.float32))
        for i in range(6000)
    ]

    top_results = cosine_similarity(query_vector, candidate_vectors, batch_size=3000)
    assert len(top_results) == 5
    assert all("id" in result and "score" in result for result in top_results)

# ------------------------------------------------------------
# Test 3
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
# Test 4
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
# Test 5
# Non-list, non-numpy query vector (invalid input type)
# ------------------------------------------------------------
def test_invalid_query_vector_type():
    query_vector = "this is not a vector"
    candidate_vectors = [
        ("doc_1", np.random.rand(1536).astype(np.float32))
    ]

    with pytest.raises(TypeError):
        cosine_similarity(query_vector, candidate_vectors) # type: ignore

# ------------------------------------------------------------
# Entry point for running this file standalone
# ------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-v"])
