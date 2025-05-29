from datasets import load_dataset
import itertools
import numpy as np
import timeit

from symrank import cosine_similarity_batch
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine

# -----------------------------------------------------------------------------
# 1) Stream & filter 100,001 records from HF dataset
# -----------------------------------------------------------------------------
dataset = load_dataset(
    "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
    split="train",
    streaming=True,
    trust_remote_code=True
)

def keep_short_id(ex):
    return len(ex["_id"]) <= 50

filtered = filter(keep_short_id, dataset)
sampled = itertools.islice(filtered, 100_001)

# collect into memory
result = [
    {"_id": rec["_id"], "embedding": rec["text-embedding-3-large-1536-embedding"]}
    for rec in sampled
]

print(f"Retrieved {len(result)} records")
print("Sample ID:", result[0]["_id"])
print("Sample embedding slice:", result[0]["embedding"][:5], "…")

# -----------------------------------------------------------------------------
# 2) Build query + candidates
# -----------------------------------------------------------------------------
n_candidates = 10000
query = np.array(result[0]["embedding"], dtype=np.float32)
subset = result[1 : 1 + n_candidates]
labels = [r["_id"] for r in subset]
candidates = np.stack([r["embedding"] for r in subset], axis=0).astype(np.float32)
vecs = list(zip(labels, candidates))

# -----------------------------------------------------------------------------
# 3) Benchmark Setup
# -----------------------------------------------------------------------------
def bench(fn, number=5, repeat=10):
    """Run fn() `number` times per repeat, repeat `repeat` times,
    and return (best_ms, mean_ms, std_ms)."""
    timer = timeit.Timer(fn)
    times = np.array(timer.repeat(number=number, repeat=repeat)) / number
    return times.min() * 1000, times.mean() * 1000, times.std() * 1000

k = 5
batch_size = 256  # tune as needed

def run_symrank_batch():
    return cosine_similarity_batch(query, vecs, k=k, batch_size=batch_size)

def run_sklearn_batch():
    all_scores = []
    # scikit-learn only takes full batches, so we loop:
    for start in range(0, n_candidates, batch_size):
        batch = candidates[start:start+batch_size]
        sims = skl_cosine(query.reshape(1, -1), batch)[0]
        all_scores.extend(sims.tolist())
    topk_idx = np.argsort(all_scores)[-k:][::-1]
    return [(labels[i], float(all_scores[i])) for i in topk_idx]

def run_numpy_batch():
    all_scores = []
    norm_q = np.linalg.norm(query)
    for start in range(0, n_candidates, batch_size):
        batch = candidates[start:start+batch_size]
        dot = batch @ query
        norm_c = np.linalg.norm(batch, axis=1)
        sims = dot / (norm_q * norm_c)
        all_scores.extend(sims.tolist())
    topk_idx = np.argsort(all_scores)[-k:][::-1]
    return [(labels[i], float(all_scores[i])) for i in topk_idx]

# -----------------------------------------------------------------------------
# 4) Run benchmarks
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    methods = [
        ("SymRank_Batch", run_symrank_batch),
        ("sklearn_Batch", run_sklearn_batch),
        ("NumPy_Batch", run_numpy_batch),
    ]

    print(f"\n{'Method':<18} | {'Best (ms)':>9} | {'Mean (ms)':>9} | {'Std (ms)':>9}")
    print("-" *  60 )
    for name, fn in methods:
        best, mean, std = bench(fn)
        print(f"{name:<18} | {best:9.3f} | {mean:9.3f} | {std:9.3f}")
