import numpy as np
import timeit
from symrank import cosine_similarity, cosine_similarity_batch

def bench_once(fn, repeat=3):
    """Run fn() once per timing, repeat `repeat` times, return the average in ms."""
    timer = timeit.Timer(fn)
    times = timer.repeat(number=1, repeat=repeat)
    return (sum(times) / len(times)) * 1e3  # convert to ms

def run_batched_benchmark(
    n_values, batch_sizes, k=5, dim=128, repeat=5
):
    header = f"{'n':>8} | {'batch':>6} | {'cos_sim (ms)':>14} | {'cos_sim_batch (ms)':>19}"
    line   = "-" * len(header)
    print(header)
    print(line)

    for n in n_values:
        # 1) Generate data ONCE for this n
        rng = np.random.default_rng(42)
        query = rng.normal(size=dim).astype(np.float32)
        candidates = rng.normal(size=(n, dim)).astype(np.float32)

        # 2) Build the list of (id, vector) once
        vecs = [(f"id_{i}", candidates[i]) for i in range(n)]

        # 3) Prepare two zero-arg callables that capture `query` and `vecs`
        def cs_all():
            return cosine_similarity(query, vecs, k=k, batch_size=None)

        def csb_all():
            return cosine_similarity_batch(query, vecs, k=k, batch_size=n)

        # 4) Time the “all” case
        t_cs_all  = bench_once(cs_all,  repeat=repeat)
        t_csb_all = bench_once(csb_all, repeat=repeat)
        print(f"{n:8d} | {'all':>6} | {t_cs_all:14.3f} | {t_csb_all:19.3f}")

        # 5) Now per‐batch timings
        for bs in batch_sizes:
            if bs >= n:
                continue
            def cs_bs(bs=bs):
                return cosine_similarity(query, vecs, k=k, batch_size=bs)
            def csb_bs(bs=bs):
                return cosine_similarity_batch(query, vecs, k=k, batch_size=bs)

            t_cs  = bench_once(cs_bs,  repeat=repeat)
            t_csb = bench_once(csb_bs, repeat=repeat)
            print(f"{n:8d} | {bs:6d} | {t_cs:14.3f} | {t_csb:19.3f}")

if __name__ == "__main__":
    n_values   = [1000, 5000, 10000, 50000]
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    run_batched_benchmark(n_values, batch_sizes, k=5, dim=1536, repeat=5)
    