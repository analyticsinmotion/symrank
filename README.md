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

<br/>

## 📦 Installation

You can install SymRank with 'uv' or alternatively using 'pip'.

### Recommended (with uv):
```bash
uv pip install symrank
```

### Using pip:
```bash
pip install symrank
```

<br/>

## 🧪 Usage

### Basic Example

```python
import symrank as sr

query = [0.1, 0.2, 0.3, 0.4]  # or np.array([...], dtype=np.float32)
candidates = [
    ("doc_1", [0.1, 0.2, 0.3, 0.5]),
    ("doc_2", [0.9, 0.1, 0.2, 0.1]),
    ("doc_3", [0.0, 0.0, 0.0, 1.0]),
]

results = sr.compare(query, candidates, top_k=2, vector_size=4)
print(results)
```

*Output*
```python
[{'id': 'doc_1', 'score': 0.9987}, {'id': 'doc_3', 'score': 0.8912}]
```

<br/>

## 🧩 API: compare(...)

```python
compare(
    query_vector,              # List[float] or np.ndarray
    candidate_vectors,         # List[Tuple[str, List[float] or np.ndarray]]
    method="cosine",           # Currently only "cosine" is supported
    top_k=5,                   # Number of top results to return
    vector_size=1536,          # Embedding dimension (default: OpenAI's)
    batch_size=None,           # Optional: split into batches for large sets
)
```

### 'compare(...)' Parameters

| Parameter         | Type                                               | Default     | Description |
|-------------------|----------------------------------------------------|-------------|-------------|
| `query_vector`     | List[float] or np.ndarray                      | _required_  | Vector to search with |
| `candidate_vectors`| List[Tuple[str, List[float] or np.ndarray]]      | _required_  | (id, vector) pairs to compare against |
| `method`           | str                                              | "cosine"  | Similarity method (E.g. "cosine") |
| `top_k`            | int                                              | 5         | Number of results to return |
| `vector_size`      | int                                              | 1536      | Dimensionality of all vectors |
| `batch_size`       | int or None                                      | None      | Optional batch size to reduce memory use |


### Returns

List of dictionaries with `id` and `score` (cosine similarity):

```python
[{"id": "doc_42", "score": 0.8763}, {"id": "doc_17", "score": 0.8451}, ...]
```




<!--
## Usage
**Import the SymRank package**

*Python Code:*
```python
from symrank import compare
```

### Examples:

#### 1. xxx
```python
# Example 1

```

#### 2. xxx
```python
# Example 2

```

#### 3. xxx
```python
# Example 3

```
-->

<br/>

## 📄 License

This project is licensed under the Apache License 2.0.





