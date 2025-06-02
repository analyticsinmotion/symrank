import numpy as np
import timeit
from symrank import cosine_similarity
from symrank.cosine_similarity_simd import cosine_similarity_simd

def bench(fn, number=10, repeat=20):
    """Run fn `number` times, repeat the measurement `repeat` times,
    and return mean_ms."""
    timer = timeit.Timer(fn)
    times = np.array(timer.repeat(number=number, repeat=repeat)) / number
    return times.mean() * 1e3

def run():
    np.random.seed(42)
    dim = 1536
    k = 5
    batch_size = None
    ns = [1000, 2500, 5000, 10000, 25000, 50000, 100000]

    print(f"{'n':>8} | {'cosine_similarity':>18} | {'cosine_similarity_simd':>23}")
    print("-" * 55)

    for n in ns:
        query = np.random.rand(dim).astype(np.float32)
        candidates = np.random.rand(n, dim).astype(np.float32)
        vecs = [(f"doc_{i}", candidates[i]) for i in range(n)]

        def run_standard():
            return cosine_similarity(query, vecs, k=k, batch_size=batch_size)

        def run_simd():
            return cosine_similarity_simd(query, vecs, k=k)

        mean_standard = bench(run_standard)
        mean_simd = bench(run_simd)

        print(f"{n:8} | {mean_standard:18.3f} | {mean_simd:23.3f}")

if __name__ == "__main__":
    run()