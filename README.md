![logo-symrank](https://github.com/user-attachments/assets/ce0b2224-d59a-4aab-a708-dcdc4968c54a)

<h1 align="center">Similarity ranking for Retrieval-Augmented Generation</h1>

<!-- badges: start -->

<div align="center">
  <table>
    <tr>
      <td><strong>Meta</strong></td>
      <td>
        <a href="https://pypi.org/project/symrank/"><img src="https://img.shields.io/pypi/v/symrank?label=PyPI&color=blue"></a>&nbsp;
        <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%7C3.11%7C3.12%7C3.13-blue?logo=python&logoColor=ffdd54"></a>&nbsp;
        <a href="https://github.com/analyticsinmotion/symrank/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>&nbsp;
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

### Recommended (with uv):
```bash
uv pip install symrank
```

### Alternatively (using pip):
```bash
pip install symrank
```

<br/>

## 🧪 Usage

### Basic Example (using python lists)

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

*Output*
```python
[{'id': 'doc_1', 'score': 0.9939991235733032}, {'id': 'doc_3', 'score': 0.7302967309951782}]
```

### Basic Example (using numpy arrays)

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

*Output*
```python
[{'id': 'doc_1', 'score': 0.9939991235733032}, {'id': 'doc_3', 'score': 0.7302967309951782}]
```

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





