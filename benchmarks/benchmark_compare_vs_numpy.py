import time
import numpy as np
from symrank import compare
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine


def benchmark_sklearn_topk(query: np.ndarray, candidates: np.ndarray, top_k: int):
    sims = skl_cosine(query.reshape(1, -1), candidates)[0]
    topk_idx = np.argsort(sims)[-top_k:][::-1]
    return [(f"doc_{i}", sims[i]) for i in topk_idx]


def benchmark_numpy_topk(query: np.ndarray, candidates: np.ndarray, top_k: int):
    dot = candidates @ query
    norm_q = np.linalg.norm(query)
    norm_c = np.linalg.norm(candidates, axis=1)
    sims = dot / (norm_q * norm_c)
    topk_idx = np.argsort(sims)[-top_k:][::-1]
    return [(f"doc_{i}", sims[i]) for i in topk_idx]


def benchmark_symrank_topk(query: np.ndarray, candidates: np.ndarray, top_k: int):
    vecs = [(f"doc_{i}", v) for i, v in enumerate(candidates)]
    return compare(query, vecs, top_k=top_k)


def run():
    np.random.seed(42)
    dim = 1536
    n = 1000
    top_k = 5

    query = np.random.rand(dim).astype(np.float32)
    candidates = np.random.rand(n, dim).astype(np.float32)

    benchmarks = [
        ("SymRank_TopK", lambda: benchmark_symrank_topk(query, candidates, top_k)),
        ("sklearn_TopK", lambda: benchmark_sklearn_topk(query, candidates, top_k)),
        ("NumPy_TopK", lambda: benchmark_numpy_topk(query, candidates, top_k)),
    ]

    for name, fn in benchmarks:
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        print(f"{name:<15} | Time: {t1 - t0:.6f} seconds")


if __name__ == "__main__":
    run()
