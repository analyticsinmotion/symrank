import numpy as np
import timeit
from symrank import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine

def benchmark_sklearn(query: np.ndarray, candidates: np.ndarray, k: int):
    sims = skl_cosine(query.reshape(1, -1), candidates)[0]
    topk_idx = np.argsort(sims)[-k:][::-1]
    return [sims[i] for i in topk_idx]

def benchmark_numpy(query: np.ndarray, candidates: np.ndarray, k: int):
    dot = candidates @ query
    norm_q = np.linalg.norm(query)
    norm_c = np.linalg.norm(candidates, axis=1)
    sims = dot / (norm_q * norm_c)
    topk_idx = np.argsort(sims)[-k:][::-1]
    return [sims[i] for i in topk_idx]

def benchmark_symrank(query: np.ndarray, candidates: np.ndarray, k: int):
    vecs = [(f"doc_{i}", v) for i, v in enumerate(candidates)]
    results = cosine_similarity(query, vecs, k=k)
    return [r["score"] for r in results]

def run():
    np.random.seed(42)
    dim = 1536
    n = 1000
    k = 5

    query = np.random.rand(dim).astype(np.float32)
    candidates = np.random.rand(n, dim).astype(np.float32)

    benchmarks = [
        ("SymRank", lambda: benchmark_symrank(query, candidates, k)),
        ("sklearn", lambda: benchmark_sklearn(query, candidates, k)),
        ("NumPy", lambda: benchmark_numpy(query, candidates, k)),
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