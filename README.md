# SymRank

## What is SymRank?
**SymRank** is a blazing-fast Python library for top-k cosine similarity ranking, designed for vector search, retrieval-augmented generation (RAG), and embedding-based matching.

Built with a Rust + SIMD backend, it offers the speed of native code with the ease of Python.

<br/>

## 🚀 Why SymRank?

⚡ Fast: SIMD-accelerated cosine scoring with adaptive parallelism

🧠 Smart: Automatically selects serial or parallel mode based on workload

🔢 Top-K optimized: Efficient inlined heap selection (no full sort overhead)

🐍 Pythonic: Easy-to-use Python API

🦀 Powered by Rust: Safe, high-performance core engine


## 📦 Installation

You can install SymRank either with 'uv' or 'pip'.

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

results = sr.compare(query, candidates, top_k=2)
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
| 'query_vector'     | 'List[float]' or 'np.ndarray'                      | _required_  | Vector to search with |
| 'candidate_vectors'| 'List[Tuple[str, List[float] or np.ndarray]]'      | _required_  | `(id, vector)` pairs to compare against |
| 'method'           | 'str'                                              | '"cosine"'  | Similarity method (only "cosine" supported) |
| 'top_k'            | 'int'                                              | '5'         | Number of results to return |
| 'vector_size'      | 'int'                                              | '1536'      | Dimensionality of all vectors |
| 'batch_size'       | 'int or None'                                      | 'None'      | Optional batch size to reduce memory use |


### Returns

List of dictionaries with `id` and `score` (cosine similarity):

```python
[{"id": "doc_42", "score": 0.8763}, {"id": "doc_17", "score": 0.8451}, ...]
```




<!-->
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

## 📄 License

This project is licensed under the Apache License 2.0.





