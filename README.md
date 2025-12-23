<!-- markdownlint-disable MD033 MD041 -->
![logo-symrank](https://github.com/user-attachments/assets/ce0b2224-d59a-4aab-a708-dcdc4968c54a)

<h1 align="center">Similarity ranking for Retrieval-Augmented Generation</h1>

<!-- badges: start -->

<div align="center">
  <table>
    <tr>
      <td><strong>Meta</strong></td>
      <td>
        <a href="https://pypi.org/project/symrank/"><img src="https://img.shields.io/pypi/v/symrank?label=PyPI&color=blue" alt="PyPI version"></a>&nbsp;
        <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13%7C3.14-blue?logo=python&logoColor=ffdd54" alt="Supports Python 3.10-3.14"></a>&nbsp;
        <a href="https://github.com/analyticsinmotion/symrank/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0 License"></a>&nbsp;
        <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>&nbsp;
        <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>&nbsp;
        <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/Powered%20by-Rust-black?logo=rust&logoColor=white" alt="Powered by Rust"></a>&nbsp;
        <a href="https://github.com/analyticsinmotion"><img src="https://raw.githubusercontent.com/analyticsinmotion/.github/main/assets/images/analytics-in-motion-github-badge-rounded.svg" alt="Analytics in Motion"></a>
        <!-- &nbsp;
        <a href="https://pypi.org/project/symrank/"><img src="https://img.shields.io/pypi/dm/symrank?label=PyPI%20downloads"></a>&nbsp;
        <a href="https://pepy.tech/project/symrank"><img src="https://static.pepy.tech/badge/symrank"></a>
        -->
      </td>
    </tr>
  </table>
</div>

<!-- badges: end -->

## ✨ What is SymRank?

**SymRank** is a blazing-fast Python library for top-k cosine similarity ranking, designed for vector search, retrieval-augmented generation (RAG), and embedding-based matching.

Built with a Rust + SIMD backend, it offers the speed of native code with the ease of Python.

<br/>

## 🚀 Why SymRank?

⚡ Fast: SIMD-accelerated cosine scoring with adaptive parallelism

🧠 Smart: Automatically selects serial or parallel mode based on workload

🔢 Top-K optimized: Efficient inlined heap selection (no full sort overhead)

🐍 Pythonic: Easy-to-use Python API

🦀 Powered by Rust: Safe, high-performance core engine

📉 Memory Efficient: Supports batching for speed and to reduce memory footprint

<br/>

## 📦 Installation

You can install SymRank with 'uv' or alternatively using 'pip'.

### Recommended (with uv)

```bash
uv pip install symrank
```

### Alternatively (using pip)

```bash
pip install symrank
```

<br/>

## 🧪 Usage

SymRank provides two APIs optimized for different workflows.

---

### Option 1: `cosine_similarity_matrix` (recommended for performance)

**Best when:**
- Candidate embeddings are already stored as a single 2D NumPy array
- Performance matters (about 10 to 14x faster for N=1,000 to 10,000 versus the list API)
- Running many queries against the same candidate set

```python
import numpy as np
from symrank import cosine_similarity_matrix

# Example data (dimension = 4 for readability)
query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

candidate_matrix = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # identical to query
        [0.0, 1.0, 0.0, 0.0],  # orthogonal
        [0.5, 0.5, 0.0, 0.0],  # partially aligned
        [0.2, 0.1, 0.0, 0.0],  # weakly aligned
    ],
    dtype=np.float32,
)

ids = ["doc_a", "doc_b", "doc_c", "doc_d"]

results = cosine_similarity_matrix(query, candidate_matrix, ids, k=3)
print(results)
```

**Output:**
```python
[
  {"id": "doc_a", "score": 1.0},
  {"id": "doc_d", "score": 0.8944272},
  {"id": "doc_c", "score": 0.70710677}
]
```

Notes:
- Scores are cosine similarity (range -1 to 1, higher = more similar)
- Results are sorted by descending similarity

**Typical production usage (1536-dimensional embeddings):**

```python
import numpy as np
from symrank import cosine_similarity_matrix

D = 1536
N = 10_000

query = np.random.rand(D).astype(np.float32)
candidate_matrix = np.random.rand(N, D).astype(np.float32)
ids = [f"doc_{i}" for i in range(N)]

top5 = cosine_similarity_matrix(query, candidate_matrix, ids, k=5)
for result in top5:
    print(f"{result['id']}: {result['score']:.4f}")
```

**Optional batching for memory control:**
```python
# Process 10k candidates in batches of 2000
results = cosine_similarity_matrix(
    query, candidate_matrix, ids, k=5, batch_size=2000
)
```

---

### Option 2: `cosine_similarity` (flexible and convenient)

**Best when:**
- Candidates come from mixed or streaming sources
- Vectors are naturally represented as (id, vector) pairs
- Simplicity is more important than maximum throughput

**Basic example using Python lists:**

```python
import symrank as sr

query = [0.1, 0.2, 0.3, 0.4]
candidates = [
    ("doc_1", [0.1, 0.2, 0.3, 0.5]),
    ("doc_2", [0.9, 0.1, 0.2, 0.1]),
    ("doc_3", [0.0, 0.0, 0.0, 1.0]),
]

results = sr.cosine_similarity(query, candidates, k=2)
print(results)
```

**Output:**
```python
[
  {"id": "doc_1", "score": 0.9939991235733032},
  {"id": "doc_3", "score": 0.7302967309951782}
]
```

**Basic example using NumPy arrays:**

```python
import symrank as sr
import numpy as np

query = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
candidates = [
    ("doc_1", np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float32)),
    ("doc_2", np.array([0.9, 0.1, 0.2, 0.1], dtype=np.float32)),
    ("doc_3", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
]

results = sr.cosine_similarity(query, candidates, k=2)
print(results)
```

**Output:**
```python
[
  {"id": "doc_1", "score": 0.9939991235733032},
  {"id": "doc_3", "score": 0.7302967309951782}
]
```

**Optional batching:**
```python
results = sr.cosine_similarity(query, candidates, k=5, batch_size=1000)
```

---

### Performance Comparison

| Dataset Size | Option 1 (matrix) | Option 2 (list) | Speedup |
|--------------|-------------------|-----------------|---------|
| N=100        | 0.02 ms           | 0.06 ms         | 3.3x    |
| N=1,000      | 0.18 ms           | 2.28 ms         | 12.7x   |
| N=10,000     | 1.50 ms           | 19.66 ms        | 13.1x   |

*Benchmark: 1536-dimensional embeddings, k=5, Python 3.14, Windows. Benchmark includes Python-side overhead for each API.*

---

### Quick Decision Guide

**Use `cosine_similarity_matrix` if:**
- ✅ You have a pre-built NumPy matrix of candidates
- ✅ Performance is critical
- ✅ Processing many queries against the same corpus

**Use `cosine_similarity` if:**
- ✅ Building candidates on-the-fly
- ✅ Mixed vector input types (lists or NumPy arrays)
- ✅ Flexibility > raw speed

Both functions return the same format: a list of dicts sorted by descending similarity score.


<br/>

## 🧩 API: cosine_similarity(...)

```python
cosine_similarity(
    query_vector,              # List[float] or np.ndarray
    candidate_vectors,         # List[Tuple[str, List[float] or np.ndarray]]
    k=5,                       # Number of top results to return
    batch_size=None            # Optional: set for memory-efficient batching
)
```

### 'cosine_similarity(...)' Parameters

| Parameter         | Type                                               | Default     | Description |
|-------------------|----------------------------------------------------|-------------|-------------|
| `query_vector`     | `list[float]` or `np.ndarray`                       | _required_  | The query vector you want to compare against the candidate vectors. |
| `candidate_vectors`| `list[tuple[str, list[float] or np.ndarray]]`          | _required_  | List of `(id, vector)` pairs. Each vector can be a list or NumPy array. |
| `k`                | `int`                                               | 5         | Number of top results to return, sorted by descending similarity. |
| `batch_size`       | `int` or `None`                                       | None      | Optional batch size to reduce memory usage. If None, uses SIMD directly. |

### Returns

List of dictionaries with `id` and `score` (cosine similarity), sorted by descending similarity:

```python
[{"id": "doc_42", "score": 0.8763}, {"id": "doc_17", "score": 0.8451}, ...]
```


<br/>

## 📄 License

This project is licensed under the Apache License 2.0.
