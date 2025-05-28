import numpy as np
import pytest
from symrank.cosine_similarity_batch import cosine_similarity_batch

# ------------------------------------------------------------
# Test 1
# Basic functionality small candidate set
# ------------------------------------------------------------
def test_basic_small_set():
    query = np.random.rand(5).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(5).astype(np.float32)) for i in range(10)]
    topk = cosine_similarity_batch(query, candidates, k=3)
    assert isinstance(topk, list)
    assert len(topk) == 3
    for item in topk:
        assert "id" in item and "score" in item

# ------------------------------------------------------------
# Test 2
# Batching splits large candidate set correctly
# ------------------------------------------------------------
def test_batching_multiple_batches():
    query = np.random.rand(5).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(5).astype(np.float32)) for i in range(10)]
    result = cosine_similarity_batch(query, candidates, k=5, batch_size=4)
    assert isinstance(result, list)
    assert len(result) == 5

# ------------------------------------------------------------
# Test 3
# Empty candidate list should raise ValueError
# ------------------------------------------------------------
def test_empty_candidates():
    query = np.random.rand(5).astype(np.float32)
    with pytest.raises(ValueError):
        cosine_similarity_batch(query, [], k=3)

# ------------------------------------------------------------
# Test 4
# Invalid query vector type should raise TypeError
# ------------------------------------------------------------
def test_invalid_query_vector_type():
    candidates = [("doc1", np.random.rand(5).astype(np.float32))]
    with pytest.raises(TypeError):
        # passing a string (invalid) should trigger our TypeError
        cosine_similarity_batch("not a vector", candidates, k=1)  # type: ignore[arg-type]

# ------------------------------------------------------------
# Test 5
# Mismatched candidate dimension should raise ValueError
# ------------------------------------------------------------
def test_mismatched_candidate_dimension():
    query = np.random.rand(5).astype(np.float32)
    candidates = [
        ("doc1", np.random.rand(4).astype(np.float32))
    ]
    with pytest.raises(ValueError):
        cosine_similarity_batch(query, candidates, k=1)

# ------------------------------------------------------------
# Test 6
# K larger than number of candidates returns all candidates
# ------------------------------------------------------------
def test_k_larger_than_candidates():
    query = np.random.rand(5).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(5).astype(np.float32)) for i in range(3)]
    result = cosine_similarity_batch(query, candidates, k=5)
    assert len(result) == 3

# ------------------------------------------------------------
# Test 7
# Zero k returns empty list
# ------------------------------------------------------------
def test_zero_k_returns_empty():
    query = np.random.rand(5).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(5).astype(np.float32)) for i in range(5)]
    result = cosine_similarity_batch(query, candidates, k=0)
    assert result == []

# ------------------------------------------------------------
# Test 8
# Perfect match should yield score approximately 1.0
# ------------------------------------------------------------
def test_perfect_match_score_one():
    vec = np.random.rand(6).astype(np.float32)
    query = vec.copy()
    candidates = [
        ("match", vec),
        ("other", np.zeros_like(vec))
    ]
    result = cosine_similarity_batch(query, candidates, k=1)
    assert result[0]["id"] == "match"
    assert result[0]["score"] == pytest.approx(1.0)

# ------------------------------------------------------------
# Test 9
# Batch size greater than number of candidates uses single batch
# ------------------------------------------------------------
def test_batch_size_greater_than_candidates():
    query = np.random.rand(4).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(4).astype(np.float32)) for i in range(5)]
    # batch_size > len(candidates)
    result_large_batch = cosine_similarity_batch(query, candidates, k=3, batch_size=10)
    # should be identical to default (no batching)
    result_default = cosine_similarity_batch(query, candidates, k=3)
    assert result_large_batch == result_default

# ------------------------------------------------------------
# Test 10
# Batch size of zero treated as no batching
# ------------------------------------------------------------
def test_zero_batch_size_equivalent_to_default():
    query = np.random.rand(4).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(4).astype(np.float32)) for i in range(5)]
    result_zero_batch = cosine_similarity_batch(query, candidates, k=3, batch_size=0)
    result_default = cosine_similarity_batch(query, candidates, k=3)
    assert result_zero_batch == result_default

# ------------------------------------------------------------
# Test 11
# Batch size of one processes each vector individually
# ------------------------------------------------------------
def test_batch_size_one():
    # Use a simple dataset with predictable ordering
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = [
        ("a", [1.0, 0.0, 0.0]),  # similarity 1.0
        ("b", [0.5, 0.5, 0.0]),  # similarity ~0.707
        ("c", [0.0, 1.0, 0.0])   # similarity 0.0
    ]
    result = cosine_similarity_batch(query, candidates, k=3, batch_size=1)
    ids = [item["id"] for item in result]
    scores = [item["score"] for item in result]
    assert ids == ["a", "b", "c"]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(np.dot(query, np.array([0.5,0.5,0.0])) /
                                      (np.linalg.norm(query) * np.linalg.norm([0.5,0.5,0.0])))

# ------------------------------------------------------------
# Test 12
# Negative batch size raises an error
# ------------------------------------------------------------
def test_negative_batch_size_raises():
    query = np.random.rand(3).astype(np.float32)
    candidates = [(f"doc_{i}", np.random.rand(3).astype(np.float32)) for i in range(3)]
    with pytest.raises((ValueError, TypeError)):
        cosine_similarity_batch(query, candidates, k=2, batch_size=-5)

