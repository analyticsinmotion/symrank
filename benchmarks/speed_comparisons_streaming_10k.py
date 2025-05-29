from datasets import load_dataset
import itertools
import numpy as np
import timeit
from symrank import cosine_similarity  
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine

# 1. Load & prepare data - Stream & filter 100001 records from hugging face dataset
dataset = load_dataset(
    "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
    split="train",
    streaming=True,
    trust_remote_code=True
)

def filter_and_select(example):
    return len(example["_id"]) <= 50

filtered = filter(filter_and_select, dataset)
sampled = itertools.islice(filtered, 100_001)

result = [
    {"_id": rec["_id"], "embedding": rec["text-embedding-3-large-1536-embedding"]}
    for rec in sampled
]

print(f"Retrieved {len(result)} records")
print("Sample ID:", result[0]["_id"])
print("Sample embedding slice:", result[0]["embedding"][:5], "…")

# 2. Build query vector + first 10000 become candidates
n_candidates = 10000
query      = np.array(result[0]["embedding"], dtype=np.float32)
subset     = result[1 : 1 + n_candidates]
labels     = [r["_id"] for r in subset]
candidates = np.stack([r["embedding"] for r in subset], axis=0).astype(np.float32)

# 3) Pack for SymRank
vecs = list(zip(labels, candidates))

# At this point you have:
#   query      : np.ndarray of shape (1536,)
#   candidates : np.ndarray of shape (10000, 1536)
#   labels     : list of 10000 IDs
#   vecs       : list of (id, np.ndarray) for SymRank


# 3. Benchmarking
def bench(fn, number=10, repeat=20):
    timer = timeit.Timer(fn)
    times = np.array(timer.repeat(number=number, repeat=repeat)) / number
    return times.min() * 1e3, times.mean() * 1e3, times.std() * 1e3

k = 5

def run_symrank():
    return cosine_similarity(query, vecs, k=k)

def run_sklearn():
    sims = skl_cosine(query.reshape(1, -1), candidates)[0]
    topk = np.argsort(sims)[-k:][::-1]
    return [(labels[i], float(sims[i])) for i in topk]

def run_numpy():
    dot = candidates @ query
    norm_q = np.linalg.norm(query)
    norm_c = np.linalg.norm(candidates, axis=1)
    sims = dot / (norm_q * norm_c)
    topk = np.argsort(sims)[-k:][::-1]
    return [(labels[i], float(sims[i])) for i in topk]

methods = [
    ("SymRank", run_symrank),
    ("sklearn", run_sklearn),
    ("NumPy", run_numpy),
]

print(f"{'Method':<10} | {'Best (ms)':>9} | {'Mean (ms)':>9} | {'Std (ms)':>9}")
print("-" * 46)
for name, fn in methods:
    best, mean, std = bench(fn)
    print(f"{name:<10} | {best:9.3f} | {mean:9.3f} | {std:9.3f}")
