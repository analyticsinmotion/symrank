import numpy as np
import timeit
from symrank import cosine_similarity_batch
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine

def bench(fn, number=10, repeat=20):
    """
    Run fn() `number` times per repeat, repeat `repeat` times,
    and return (best_ms, mean_ms, std_ms).
    """
    timer = timeit.Timer(fn)
    times = np.array(timer.repeat(number=number, repeat=repeat)) / number
    return times.min() * 1e3, times.mean() * 1e3, times.std() * 1e3

def run():
    # Configuration
    np.random.seed(42)
    dim = 1536
    n = 50000
    k = 5
    batch_size = 256

    # Data setup (done once)
    query = np.random.rand(dim).astype(np.float32)
    candidates = np.random.rand(n, dim).astype(np.float32)

    # Build the list of (id, vector) once for SymRank
    vecs = [(f"doc_{i}", candidates[i]) for i in range(n)]

    # --- Define the zero-arg callables ---
    def run_symrank():
        # uses the Python wrapper that internally batches and calls Rust
        return cosine_similarity_batch(query, vecs, k=k, batch_size=batch_size)

    def run_sklearn_batch():
        # scikit-learn, batched manually in Python
        all_scores = []
        for start in range(0, n, batch_size):
            batch = candidates[start:start+batch_size]
            sims = skl_cosine(query.reshape(1, -1), batch)[0]
            all_scores.extend([(start + i, float(s)) for i, s in enumerate(sims)])
        # global top-k
        scores = np.array([s for _, s in all_scores])
        topk_idx = scores.argsort()[-k:][::-1]
        return [(f"doc_{all_scores[i][0]}", all_scores[i][1]) for i in topk_idx]

    def run_numpy_batch():
        # pure NumPy, batched manually
        all_scores = []
        norm_q = np.linalg.norm(query)
        for start in range(0, n, batch_size):
            batch = candidates[start:start+batch_size]
            dot = batch @ query
            norm_c = np.linalg.norm(batch, axis=1)
            sims = dot / (norm_q * norm_c)
            all_scores.extend([(start + i, float(s)) for i, s in enumerate(sims)])
        # global top-k
        scores = np.array([s for _, s in all_scores])
        topk_idx = scores.argsort()[-k:][::-1]
        return [(f"doc_{all_scores[i][0]}", all_scores[i][1]) for i in topk_idx]

    # Benchmark all three
    methods = [
        ("SymRank_TopK_Batch", run_symrank),
        ("sklearn_TopK_Batch", run_sklearn_batch),
        ("NumPy_TopK_Batch", run_numpy_batch),
    ]

    # Header
    print(f"{'Method':<20} | {'Best (ms)':>9} | {'Mean (ms)':>9} | {'Std Dev (ms)':>12}")
    print("-" * 60)
    for name, fn in methods:
        best, mean, std = bench(fn, number=5, repeat=10)
        print(f"{name:<20} | {best:9.3f} | {mean:9.3f} | {std:12.3f}")

if __name__ == "__main__":
    run()
