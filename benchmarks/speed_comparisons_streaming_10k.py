# benchmarks/speed_comparisons_streaming_10k.py
from __future__ import annotations

import itertools
import timeit

import numpy as np
from datasets import load_dataset

from symrank import cosine_similarity, cosine_similarity_matrix

try:
    from sklearn.metrics.pairwise import cosine_similarity as skl_cosine  # type: ignore
except Exception:
    skl_cosine = None


def bench(fn, number=10, repeat=20):
    timer = timeit.Timer(fn)
    times = np.array(timer.repeat(number=number, repeat=repeat)) / number
    return times.min() * 1e3, times.mean() * 1e3, times.std() * 1e3


def main() -> None:
    # 1) Load & prepare data, stream and filter 100,001 records from HF dataset
    dataset = load_dataset(
        "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
        split="train",
        streaming=True,
    )

    def keep_short_id(example):
        return len(example["_id"]) <= 50

    filtered = filter(keep_short_id, dataset)
    sampled = itertools.islice(filtered, 100_001)

    result = [
        {"_id": rec["_id"], "embedding": rec["text-embedding-3-large-1536-embedding"]}
        for rec in sampled
    ]

    print(f"Retrieved {len(result)} records")
    print("Sample ID:", result[0]["_id"])
    print("Sample embedding slice:", result[0]["embedding"][:5], "…")

    # 2) Build query + candidates
    n_candidates = 10_000
    query = np.array(result[0]["embedding"], dtype=np.float32)
    subset = result[1 : 1 + n_candidates]
    labels = [r["_id"] for r in subset]
    candidates = np.stack([r["embedding"] for r in subset], axis=0).astype(np.float32)

    # SymRank list and matrix inputs
    vecs = list(zip(labels, candidates))
    ids = labels
    C = candidates

    k = 5

    def run_symrank_matrix():
        return cosine_similarity_matrix(query, C, ids, k=k, batch_size=None)

    def run_symrank_list():
        return cosine_similarity(query, vecs, k=k, batch_size=None)

    def run_numpy_normalized():
        # Candidates normalized once, query normalized per call (matches README baseline)
        Cn = C
        c_norms = np.linalg.norm(Cn, axis=1)
        c_norms = np.where(c_norms == 0.0, 1.0, c_norms).astype(np.float32, copy=False)
        C_normed = (Cn / c_norms[:, None]).astype(np.float32, copy=False)

        q = query
        qn = np.linalg.norm(q)
        if qn == 0.0:
            sims = np.zeros((C_normed.shape[0],), dtype=np.float32)
        else:
            sims = C_normed @ (q / qn).astype(np.float32, copy=False)

        topk = np.argsort(sims)[-k:][::-1]
        return [(labels[i], float(sims[i])) for i in topk]

    def run_sklearn():
        if skl_cosine is None:
            raise RuntimeError("scikit-learn not installed.")
        sims = skl_cosine(query.reshape(1, -1), candidates)[0]
        topk = np.argsort(sims)[-k:][::-1]
        return [(labels[i], float(sims[i])) for i in topk]

    methods = [
        ("SymRank matrix", run_symrank_matrix),
        ("SymRank list", run_symrank_list),
        ("NumPy normalized", run_numpy_normalized),
    ]
    if skl_cosine is not None:
        methods.append(("sklearn", run_sklearn))
    else:
        print("scikit-learn: NOT INSTALLED, skipping sklearn benchmark.")

    print(f"{'Method':<16} | {'Best (ms)':>9} | {'Mean (ms)':>9} | {'Std (ms)':>9}")
    print("-" * 54)
    for name, fn in methods:
        best, mean, std = bench(fn)
        print(f"{name:<16} | {best:9.3f} | {mean:9.3f} | {std:9.3f}")

    # Best-effort cleanup to avoid noisy generator shutdown errors on Windows
    try:
        dataset._ex_iterable = None  # type: ignore[attr-defined]
    except Exception:
        pass


if __name__ == "__main__":
    main()
