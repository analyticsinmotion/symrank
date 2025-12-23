# benchmarks/matrix_vs_numpy_sklearn.py
#
# Benchmark SymRank cosine_similarity_matrix against:
# 1) SymRank cosine_similarity (list-of-tuples API)
# 2) NumPy vectorized cosine similarity + top-k (naive, recomputes candidate norms each call)
# 3) NumPy vectorized cosine similarity + top-k (precomputed candidate norms, fair for many queries)
# 4) NumPy normalized-candidates dot + top-k (candidates normalized once, query normalized per call)
# 5) NumPy batch queries (best case for NumPy/BLAS, optional comparison)
# 6) scikit-learn cosine_similarity + top-k (optional, if installed)
#
# Notes:
# - NumPy and sklearn paths are fully vectorized (no Python loops over candidates).
# - Some baselines precompute corpus state outside the timed function, to reflect many-query usage.
#
# Run:
#   uv run benchmarks/matrix_vs_numpy_sklearn.py

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from symrank import cosine_similarity, cosine_similarity_matrix

try:
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity  # type: ignore
except Exception:
    sklearn_cosine_similarity = None


# ----------------------------
# Helpers
# ----------------------------

def ensure_float32_c_contiguous_2d(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected a NumPy array.")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}.")
    return np.ascontiguousarray(x, dtype=np.float32)


def ensure_float32_c_contiguous_1d(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError("Expected a NumPy array.")
    x = np.asarray(x, dtype=np.float32).ravel()
    return np.ascontiguousarray(x)


def topk_indices_desc_1d(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of top-k scores in descending order (1D).
    Uses argpartition for O(n) selection plus O(k log k) sort.
    """
    n = scores.shape[0]
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    if k >= n:
        return np.argsort(scores)[::-1]

    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx


def topk_indices_desc_2d(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return top-k indices per row in descending order (2D), fully vectorized.

    scores: (Q, N)
    returns: (Q, min(k, N)) indices into axis=1, sorted by descending score per row
    """
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D scores, got shape {scores.shape}")

    q, n = scores.shape
    if k <= 0:
        return np.empty((q, 0), dtype=np.int64)

    k_eff = min(k, n)

    if k_eff == n:
        return np.argsort(scores, axis=1)[:, ::-1]

    idx = np.argpartition(scores, -k_eff, axis=1)[:, -k_eff:]
    row = np.arange(q)[:, None]
    scores_k = scores[row, idx]
    order = np.argsort(scores_k, axis=1)[:, ::-1]
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    return idx_sorted


def time_ms(
    fn: Callable[[], object],
    warmup: int,
    repeat: int,
    number: int,
) -> tuple[float, float, float]:
    """
    Returns (mean_ms, std_ms, median_ms) for per-call time.
    """
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        t1 = time.perf_counter()
        per_call_ms = ((t1 - t0) / number) * 1e3
        samples.append(per_call_ms)

    samples_np = np.array(samples, dtype=np.float64)
    return float(samples_np.mean()), float(samples_np.std(ddof=0)), float(np.median(samples_np))


def maybe_print_flags(matrix: np.ndarray) -> None:
    print(f"matrix: {matrix.flags['C_CONTIGUOUS']} {matrix.dtype} {matrix.shape}")


# ----------------------------
# Baselines
# ----------------------------

def symrank_list_topk(
    query: np.ndarray,
    list_input: Sequence[tuple[str, np.ndarray]],
    k: int,
) -> list[dict]:
    q = ensure_float32_c_contiguous_1d(query)
    return cosine_similarity(q, list_input, k=k, batch_size=None)


def numpy_cosine_topk_naive(
    query: np.ndarray,
    candidates: np.ndarray,
    ids: Sequence[str],
    k: int,
) -> list[dict]:
    """
    Fully vectorized NumPy baseline (naive for repeated queries):
    sims = (C @ q) / (||q|| * ||C||)
    This recomputes ||C|| every call.
    """
    q = ensure_float32_c_contiguous_1d(query)
    C = ensure_float32_c_contiguous_2d(candidates)

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        sims = np.zeros((C.shape[0],), dtype=np.float32)
    else:
        dots = C @ q
        c_norms = np.linalg.norm(C, axis=1)
        denom = q_norm * c_norms
        sims = np.divide(
            dots,
            denom,
            out=np.zeros_like(dots, dtype=np.float32),
            where=denom != 0.0,
        )

    top_idx = topk_indices_desc_1d(sims, k)
    return [{"id": ids[int(i)], "score": float(sims[int(i)])} for i in top_idx]


def numpy_cosine_topk_precomputed_norms(
    query: np.ndarray,
    candidates: np.ndarray,
    candidate_norms: np.ndarray,
    ids: Sequence[str],
    k: int,
) -> list[dict]:
    """
    Vectorized NumPy baseline (fair for many queries):
    Precompute candidate_norms once, then per query compute dots and divide.
    """
    q = ensure_float32_c_contiguous_1d(query)
    C = ensure_float32_c_contiguous_2d(candidates)

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        sims = np.zeros((C.shape[0],), dtype=np.float32)
    else:
        dots = C @ q
        denom = (q_norm * candidate_norms).astype(np.float32, copy=False)
        sims = np.divide(
            dots,
            denom,
            out=np.zeros_like(dots, dtype=np.float32),
            where=denom != 0.0,
        )

    top_idx = topk_indices_desc_1d(sims, k)
    return [{"id": ids[int(i)], "score": float(sims[int(i)])} for i in top_idx]


def numpy_cosine_topk_normalized_candidates(
    query: np.ndarray,
    candidates_normed: np.ndarray,
    ids: Sequence[str],
    k: int,
) -> list[dict]:
    """
    Vectorized NumPy baseline (common production setup):
    Candidates are normalized once and stored.
    Per query: normalize q, then sims = C_normed @ q_normed.
    """
    q = ensure_float32_c_contiguous_1d(query)
    Cn = ensure_float32_c_contiguous_2d(candidates_normed)

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        sims = np.zeros((Cn.shape[0],), dtype=np.float32)
    else:
        qn = (q / q_norm).astype(np.float32, copy=False)
        sims = Cn @ qn

    top_idx = topk_indices_desc_1d(sims, k)
    return [{"id": ids[int(i)], "score": float(sims[int(i)])} for i in top_idx]


def numpy_cosine_topk_batch_queries(
    queries: np.ndarray,
    candidates_normed: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Batch query best case for NumPy:
    queries: (Q, D)
    candidates_normed: (N, D) already normalized
    returns: (Q, min(k, N)) indices into candidates, per query, descending

    Fully vectorized, no Python loop over Q.
    """
    Qm = ensure_float32_c_contiguous_2d(queries)
    Cn = ensure_float32_c_contiguous_2d(candidates_normed)

    q_norms = np.linalg.norm(Qm, axis=1)
    q_norms = np.where(q_norms == 0.0, 1.0, q_norms).astype(np.float32, copy=False)
    Qn = (Qm / q_norms[:, None]).astype(np.float32, copy=False)

    sims = Qn @ Cn.T
    return topk_indices_desc_2d(sims, k)


def sklearn_cosine_topk(
    query: np.ndarray,
    candidates: np.ndarray,
    ids: Sequence[str],
    k: int,
) -> list[dict]:
    """
    scikit-learn baseline:
    sims = cosine_similarity(query[None, :], candidates)[0]
    """
    if sklearn_cosine_similarity is None:
        raise RuntimeError("scikit-learn not installed.")

    q = ensure_float32_c_contiguous_1d(query)
    C = ensure_float32_c_contiguous_2d(candidates)

    sims = sklearn_cosine_similarity(q.reshape(1, -1), C)[0].astype(np.float32, copy=False)
    top_idx = topk_indices_desc_1d(sims, k)
    return [{"id": ids[int(i)], "score": float(sims[int(i)])} for i in top_idx]


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    np.random.seed(42)

    D = 1536
    k = 5

    # Added N=20 and N=50 to better reflect common re-ranking candidate set sizes.
    Ns = [20, 50, 100, 500, 1000, 5000, 10000]

    warmup = 5
    repeat = 20
    number = 20

    enable_batch_numpy = True
    batch_q = 64

    print("=" * 78)
    print("Matrix API vs SymRank list vs NumPy vs scikit-learn Benchmark")
    print("=" * 78)
    print(f"Config: D={D}, k={k}, warmup={warmup}, repeat={repeat}, number={number}")
    print("Notes: NumPy and sklearn paths are fully vectorized (no Python loops over candidates).")
    print("Notes: Some NumPy baselines precompute corpus state outside timing (norms, normalization).")
    print("Notes: NumPy batch is reported both per call (Q queries) and per query (divided by Q).")
    if sklearn_cosine_similarity is None:
        print("scikit-learn: NOT INSTALLED, skipping sklearn benchmarks.")
    print("=" * 78)

    for N in Ns:
        query = np.random.rand(D).astype(np.float32)
        candidates = np.random.rand(N, D).astype(np.float32)
        ids = [f"doc_{i}" for i in range(N)]

        maybe_print_flags(candidates)

        list_input = [(ids[i], candidates[i]) for i in range(N)]

        C = ensure_float32_c_contiguous_2d(candidates)
        candidate_norms = np.linalg.norm(C, axis=1).astype(np.float32, copy=False)

        denom = np.where(candidate_norms == 0.0, 1.0, candidate_norms).astype(np.float32, copy=False)
        candidates_normed = (C / denom[:, None]).astype(np.float32, copy=False)

        batch_queries = np.random.rand(batch_q, D).astype(np.float32)

        def run_symrank_matrix() -> list[dict]:
            return cosine_similarity_matrix(query, candidates, ids, k=k, batch_size=None)

        def run_symrank_list() -> list[dict]:
            return symrank_list_topk(query, list_input, k=k)

        def run_numpy_naive() -> list[dict]:
            return numpy_cosine_topk_naive(query, candidates, ids, k=k)

        def run_numpy_precomputed_norms() -> list[dict]:
            return numpy_cosine_topk_precomputed_norms(query, candidates, candidate_norms, ids, k=k)

        def run_numpy_normalized() -> list[dict]:
            return numpy_cosine_topk_normalized_candidates(query, candidates_normed, ids, k=k)

        def run_numpy_batch() -> np.ndarray:
            return numpy_cosine_topk_batch_queries(batch_queries, candidates_normed, k=k)

        def run_sklearn() -> list[dict]:
            return sklearn_cosine_topk(query, candidates, ids, k=k)

        sym_m = run_symrank_matrix()
        np_n = run_numpy_normalized()
        if sym_m and np_n and sym_m[0]["id"] != np_n[0]["id"]:
            print("WARNING: top-1 ID differs between SymRank matrix and NumPy normalized.")
            print(f"  SymRank top1: {sym_m[0]}")
            print(f"  NumPy   top1: {np_n[0]}")

        symm_mean, symm_std, symm_med = time_ms(run_symrank_matrix, warmup=warmup, repeat=repeat, number=number)
        syml_mean, syml_std, syml_med = time_ms(run_symrank_list, warmup=warmup, repeat=repeat, number=number)

        np0_mean, np0_std, np0_med = time_ms(run_numpy_naive, warmup=warmup, repeat=repeat, number=number)
        np1_mean, np1_std, np1_med = time_ms(run_numpy_precomputed_norms, warmup=warmup, repeat=repeat, number=number)
        np2_mean, np2_std, np2_med = time_ms(run_numpy_normalized, warmup=warmup, repeat=repeat, number=number)

        npb_mean: Optional[float] = None
        npb_std: Optional[float] = None
        npb_med: Optional[float] = None
        npb_mean_per_query: Optional[float] = None
        npb_std_per_query: Optional[float] = None
        npb_med_per_query: Optional[float] = None

        sk_mean: Optional[float] = None
        sk_std: Optional[float] = None
        sk_med: Optional[float] = None

        if enable_batch_numpy:
            npb_mean, npb_std, npb_med = time_ms(
                run_numpy_batch,
                warmup=warmup,
                repeat=repeat,
                number=max(1, number // 2),
            )
            npb_mean_per_query = npb_mean / batch_q
            npb_std_per_query = npb_std / batch_q
            npb_med_per_query = npb_med / batch_q

        if sklearn_cosine_similarity is not None:
            sk_mean, sk_std, sk_med = time_ms(
                run_sklearn,
                warmup=warmup,
                repeat=repeat,
                number=number,
            )

        print(f"\nN={N:,} candidates (D={D}, k={k})")
        print(f"  SymRank matrix:         {symm_mean:8.3f} ± {symm_std:6.3f} ms (median {symm_med:8.3f} ms)")
        print(f"  SymRank list:           {syml_mean:8.3f} ± {syml_std:6.3f} ms (median {syml_med:8.3f} ms)")

        print(f"  NumPy naive:            {np0_mean:8.3f} ± {np0_std:6.3f} ms (median {np0_med:8.3f} ms)")
        print(f"  NumPy precomp norms:    {np1_mean:8.3f} ± {np1_std:6.3f} ms (median {np1_med:8.3f} ms)")
        print(f"  NumPy normalized cand:  {np2_mean:8.3f} ± {np2_std:6.3f} ms (median {np2_med:8.3f} ms)")

        if enable_batch_numpy and npb_mean is not None and npb_std is not None and npb_med is not None:
            print(
                f"  NumPy batch Q={batch_q}:       {npb_mean:8.3f} ± {npb_std:6.3f} ms (median {npb_med:8.3f} ms) per call"
            )
            print(
                f"  NumPy batch Q={batch_q}:       {npb_mean_per_query:8.3f} ± {npb_std_per_query:6.3f} ms (median {npb_med_per_query:8.3f} ms) per query"
            )

        if sk_mean is not None and sk_std is not None and sk_med is not None:
            print(f"  sklearn:                {sk_mean:8.3f} ± {sk_std:6.3f} ms (median {sk_med:8.3f} ms)")

        print("  Speedups (baseline / SymRank matrix), single-query comparable:")
        print(f"    SymRank list / SymRank matrix:        {syml_mean / symm_mean:6.2f}x")
        print(f"    NumPy naive / SymRank matrix:         {np0_mean / symm_mean:6.2f}x")
        print(f"    NumPy precomp norms / SymRank matrix: {np1_mean / symm_mean:6.2f}x")
        print(f"    NumPy normalized / SymRank matrix:    {np2_mean / symm_mean:6.2f}x")

        if enable_batch_numpy and npb_mean_per_query is not None:
            print(f"    NumPy batch per query / SymRank matrix: {npb_mean_per_query / symm_mean:6.2f}x")

        if sk_mean is not None:
            print(f"    sklearn / SymRank matrix:             {sk_mean / symm_mean:6.2f}x")

    print("\n" + "=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
