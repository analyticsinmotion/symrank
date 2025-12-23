import numpy as np
import time
from statistics import mean, stdev
from symrank import cosine_similarity, cosine_similarity_matrix
from symrank.symrank import cosine_similarity as _cosine_similarity
import timeit

def time_ms(fn, repeat=25, number=50) -> tuple[float, float, float]:
    """Return (mean_ms, std_ms, median_ms) per call."""
    t = timeit.Timer(fn)
    samples = np.array(t.repeat(repeat=repeat, number=number)) / number * 1e3
    return float(samples.mean()), float(samples.std(ddof=1)), float(np.median(samples))

def benchmark(query, matrix, ids, k, num_runs=100, warmup=10):
    """Benchmark both APIs with warmup to avoid cold start overhead"""

    # Pre-build list input once
    list_input = [(ids[i], matrix[i]) for i in range(len(ids))]

    # Warmup: list API
    for _ in range(warmup):
        cosine_similarity(query, list_input, k=k)

    # Timed runs: list API
    times_list = []
    for _ in range(num_runs):
        start = time.perf_counter()
        cosine_similarity(query, list_input, k=k)
        times_list.append((time.perf_counter() - start) * 1000)

    # Warmup: matrix API
    for _ in range(warmup):
        cosine_similarity_matrix(query, matrix, ids, k=k)

    # Timed runs: matrix API
    times_matrix = []
    for _ in range(num_runs):
        start = time.perf_counter()
        cosine_similarity_matrix(query, matrix, ids, k=k)
        times_matrix.append((time.perf_counter() - start) * 1000)

    return {
        "list": {"mean": mean(times_list), "std": stdev(times_list), "min": min(times_list)},
        "matrix": {"mean": mean(times_matrix), "std": stdev(times_matrix), "min": min(times_matrix)},
        "speedup": mean(times_list) / mean(times_matrix)
    }


def main():
    print("=" * 70)
    print("Matrix API vs List API Benchmark")
    print("=" * 70)

    for N in [100, 500, 1000, 5000, 10000]:
        query = np.random.rand(1536).astype(np.float32)
        matrix = np.random.rand(N, 1536).astype(np.float32)
        ids = [f"doc_{i}" for i in range(N)]

        # DEBUG: validate matrix properties once per N
        print(
            "matrix:",
            matrix.flags["C_CONTIGUOUS"],
            matrix.dtype,
            matrix.shape
        )

        # # Measure Rust kernel only (single call, no Python overhead)
        # start = time.perf_counter()
        # _cosine_similarity(query, matrix, 5)
        # rust_elapsed = (time.perf_counter() - start) * 1000
        # print(f"  Rust kernel only: {rust_elapsed:.3f} ms")

        rust_mean, rust_std, rust_med = time_ms(
            lambda: _cosine_similarity(query, matrix, 5),
            repeat=20,
            number=20
        )
        print(f"  Rust kernel only: {rust_mean:.3f} ± {rust_std:.3f} ms (median {rust_med:.3f} ms)")

        results = benchmark(query, matrix, ids, k=5, num_runs=50, warmup=5)

        print(f"\nN={N:5d} candidates (k=5, dim=1536):")
        print(f"  List API:   {results['list']['mean']:6.2f} ± {results['list']['std']:5.2f} ms  (min: {results['list']['min']:5.2f} ms)")
        print(f"  Matrix API: {results['matrix']['mean']:6.2f} ± {results['matrix']['std']:5.2f} ms  (min: {results['matrix']['min']:5.2f} ms)")
        print(f"  Speedup:    {results['speedup']:.2f}x  ({(results['speedup']-1)*100:+.0f}% faster)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
