import numpy as np
import timeit
from symrank import compare
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine

def benchmark_sklearn_topk(query: np.ndarray, candidates: np.ndarray, top_k: int):
    sims = skl_cosine(query.reshape(1, -1), candidates)[0]
    topk_idx = np.argsort(sims)[-top_k:][::-1]
    return [sims[i] for i in topk_idx]

def benchmark_numpy_topk(query: np.ndarray, candidates: np.ndarray, top_k: int):
    dot = candidates @ query
    norm_q = np.linalg.norm(query)
    norm_c = np.linalg.norm(candidates, axis=1)
    sims = dot / (norm_q * norm_c)
    topk_idx = np.argsort(sims)[-top_k:][::-1]
    return [sims[i] for i in topk_idx]

def benchmark_symrank_topk(query: np.ndarray, candidates: np.ndarray, top_k: int):
    vecs = [(f"doc_{i}", v) for i, v in enumerate(candidates)]
    results = compare(query, vecs, top_k=top_k)
    return [r["score"] for r in results]

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

    number = 3
    repeat = 5

    print(f"{'Benchmark':<15} | {'Normalized (ms)':<15} | {'Similarity Scores'}")
    print("-" * 70)

    for name, fn in benchmarks:
        timer = timeit.Timer(fn)
        times = timer.repeat(repeat=repeat, number=number)
        times = np.array(times) / number  # Normalize per iteration
        norm_ms = times.mean() * 1000

        # Get the scores (run once, not timed)
        scores = fn()
        scores_str = ", ".join(f"{s:.5f}" for s in scores)

        print(f"{name:<15} | {norm_ms:<15.4f} | {scores_str}")

if __name__ == "__main__":
    run()