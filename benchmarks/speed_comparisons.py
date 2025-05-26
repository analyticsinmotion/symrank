import numpy as np
import timeit
from typing import Optional
from symrank import compare
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine


def benchmark_sklearn_topk(query: np.ndarray, candidates: np.ndarray, k: int):
    sims = skl_cosine(query.reshape(1, -1), candidates)[0]
    topk_idx = np.argsort(sims)[-k:][::-1]
    return [(f"doc_{i}", sims[i]) for i in topk_idx]


def benchmark_numpy_topk(query: np.ndarray, candidates: np.ndarray, k: int):
    dot = candidates @ query
    norm_q = np.linalg.norm(query)
    norm_c = np.linalg.norm(candidates, axis=1)
    sims = dot / (norm_q * norm_c)
    topk_idx = np.argsort(sims)[-k:][::-1]
    return [(f"doc_{i}", sims[i]) for i in topk_idx]


def benchmark_symrank_topk(query: np.ndarray, candidates: np.ndarray, k: int, vector_size: int, batch_size: Optional[int] = None):
    vecs = [(f"doc_{i}", v) for i, v in enumerate(candidates)]
    return compare(query, vecs, k=k, vector_size=vector_size, batch_size=batch_size)


def run():
    np.random.seed(42)
    dim = 1536  # Change this to 3072 or any other dimension you want (384, 768, 1536, 3072)
    n = 1000
    k = 5
    batch_size = None  # Adjust this as needed

    query = np.random.rand(dim).astype(np.float32)
    candidates = np.random.rand(n, dim).astype(np.float32)

    benchmarks = [
        ("SymRank_TopK", lambda: benchmark_symrank_topk(query, candidates, k, dim, batch_size)),
        ("sklearn_TopK", lambda: benchmark_sklearn_topk(query, candidates, k)),
        ("NumPy_TopK", lambda: benchmark_numpy_topk(query, candidates, k)),
    ]

    number = 5    # Iterations per repeat
    repeat = 10   # Repeats

    # Header
    print(f"{'Benchmark':<15} | {'Best (ms)':<10} | {'Mean (ms)':<10} | {'Std Dev (ms)':<12} | Iter x Repeats")
    print("-" * 70)

    for name, fn in benchmarks:
        timer = timeit.Timer(fn)
        times = timer.repeat(repeat=repeat, number=number)
        times = np.array(times) / number  # Normalize per iteration

        best_ms = times.min() * 1000
        mean_ms = times.mean() * 1000
        std_ms = times.std() * 1000

        print(f"{name:<15} | {best_ms:<10.4f} | {mean_ms:<10.4f} | {std_ms:<12.4f} | {number} x {repeat}")


if __name__ == "__main__":
    run()
    