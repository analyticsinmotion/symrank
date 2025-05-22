import time
import timeit
import tracemalloc
import numpy as np
from typing import Optional
from symrank import compare


def generate_vectors(vector_size: int, num_candidates: int):
    """Generate a random query vector and list of candidate vectors."""
    query_vector = np.random.rand(vector_size).astype(np.float32)
    candidate_vectors = [
        (f"doc_{i}", np.random.rand(vector_size).astype(np.float32))
        for i in range(num_candidates)
    ]
    return query_vector, candidate_vectors


def run_benchmark(
    vector_size: int = 1536,
    num_candidates: int = 5000,
    top_k: int = 5,
    batch_size: Optional[int] = None,
    repeat: int = 5,
):
    """Run a benchmark on the compare function and report timing and memory usage."""

    # Generate synthetic data once (same vectors reused across repeats)
    query_vector, candidate_vectors = generate_vectors(vector_size, num_candidates)

    # Define the benchmark function
    def benchmark_function():
        compare(
            query_vector,
            candidate_vectors,
            method="cosine",
            top_k=top_k,
            vector_size=vector_size,
            batch_size=batch_size,
        )

    # Start memory tracking
    tracemalloc.start()

    total_elapsed_times = []

    for i in range(repeat):
        start_time = time.perf_counter()
        benchmark_function()
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        total_elapsed_times.append(elapsed)

    # Memory snapshot
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Use timeit to measure normalized time for a single full call
    normalized_batch_time = timeit.timeit(benchmark_function, number=1)
    normalized_candidate_time = normalized_batch_time / num_candidates if num_candidates else 0

    # Final reporting
    avg_total_time = sum(total_elapsed_times) / repeat

    print("=== Symrank Benchmark Results ===")
    print(f"Number of candidate vectors: {num_candidates}")
    print(f"Vector dimension: {vector_size}")
    print(f"Top-k: {top_k}")
    print(f"Batch size: {batch_size if batch_size else 'No batching'}")
    print(f"Total runs: {repeat}")
    print(f"Average total execution time (repeat runs): {avg_total_time:.6f} seconds")
    print(f"Normalized execution time (one batch): {normalized_batch_time:.6f} seconds")
    print(f"Normalized execution time per candidate: {normalized_candidate_time:.9f} seconds")
    print(f"Peak memory usage: {peak / 1024:.2f} KB")


if __name__ == "__main__":
    run_benchmark(
        vector_size=1536,
        num_candidates=1000,
        top_k=5,
        batch_size=256,
        repeat=5
    )
