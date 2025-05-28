import numpy as np
import timeit
from symrank import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine

def bench(fn, number=10, repeat=20):
    """Run fn `number` times, repeat the measurement `repeat` times,
    and return (best_ms, mean_ms, std_ms)."""
    timer = timeit.Timer(fn)
    times = np.array(timer.repeat(number=number, repeat=repeat)) / number
    return times.min() * 1e3, times.mean() * 1e3, times.std() * 1e3

def run():
    # Configuration
    np.random.seed(42)
    dim = 1536
    n = 1000
    k = 5
    batch_size = None  # or an integer for batching

    # Data setup (done once)
    query = np.random.rand(dim).astype(np.float32)
    candidates = np.random.rand(n, dim).astype(np.float32)
    vecs = [(f"doc_{i}", candidates[i]) for i in range(n)]

    # Define the callable for each method
    def run_symrank():
        return cosine_similarity(query, vecs, k=k, batch_size=batch_size)

    def run_sklearn():
        sims = skl_cosine(query.reshape(1, -1), candidates)[0]
        topk_idx = np.argsort(sims)[-k:][::-1]
        return [(f"doc_{i}", float(sims[i])) for i in topk_idx]

    def run_numpy():
        dot = candidates @ query
        norm_q = np.linalg.norm(query)
        norm_c = np.linalg.norm(candidates, axis=1)
        sims = dot / (norm_q * norm_c)
        topk_idx = np.argsort(sims)[-k:][::-1]
        return [(f"doc_{i}", float(sims[i])) for i in topk_idx]

    # Benchmark all three
    methods = [
        ("SymRank", run_symrank),
        ("sklearn", run_sklearn),
        ("NumPy", run_numpy),
    ]

    # Header
    print(f"{'Method':<10} | {'Best (ms)':>9} | {'Mean (ms)':>9} | {'Std (ms)':>9}")
    print("-" * 46)
    for name, fn in methods:
        best, mean, std = bench(fn)
        print(f"{name:<10} | {best:9.3f} | {mean:9.3f} | {std:9.3f}")

if __name__ == "__main__":
    run()
