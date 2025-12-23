__version__ = "0.2.0"
from .cosine_similarity import cosine_similarity
from .cosine_similarity_batch import cosine_similarity_batch
from .cosine_similarity_simd import cosine_similarity_simd
from .cosine_similarity_matrix import cosine_similarity_matrix

__all__ = [
    "cosine_similarity",
    "cosine_similarity_batch",
    "cosine_similarity_matrix",
    "cosine_similarity_simd",
]
