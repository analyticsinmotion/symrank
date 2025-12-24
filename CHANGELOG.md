<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2025-12-24

### Added

- Added support for Python 3.14
- New `cosine_similarity_matrix` API for pre-built NumPy candidate matrices, achieving speedup over list-based API by eliminating Python object iteration overhead. Optimized for production RAG pipelines and vector search with large candidate sets.
- Comprehensive test suite for `cosine_similarity_matrix` with 13 test cases covering edge cases, type conversions, batching behavior, and equivalence with existing API.
- Benchmark script `matrix_vs_list_api.py` demonstrating performance comparison between matrix and list-based APIs across various dataset sizes.
- Comprehensive benchmark script matrix_vs_numpy_sklearn.py comparing SymRank against multiple NumPy baseline strategies (naive, precomputed norms, normalized candidates) and scikit-learn, demonstrating large speedup across N=100-10,000.

### Changed

- Optimized top-k selection in Rust core by replacing redundant sort_unstable_by with iterator reversal. The heap's into_sorted_vec() already returns sorted results, so reversing the iterator achieves descending order without an additional O(k log k) sort operation.
- Updated all Rust dependencies to latest stable versions:
  - `numpy` 0.25.0 → 0.27.1
  - `pyo3` 0.25.0 → 0.27.2
  - `ordered-float` 5.0.0 → 5.1.0
  - `rayon` 1.10.0 → 1.11.0
  - `rand` 0.9.1 → 0.9.2
  - `wide` 0.7.32 → 1.1.1
- Refactored speed_comparisons_streaming_10k.py to include matrix API comparison and align NumPy baseline with production-ready normalized candidates pattern.
- Optimized cosine similarity scoring loop with explicit zero-norm handling for query and candidate vectors.
- Stabilized SIMD-based cosine similarity implementation while preserving existing performance characteristics.

### Fixed

- Prevented division-by-zero in cosine similarity when encountering zero-norm candidate vectors.
- Resolved duplicate `ndarray` dependency versions by aligning all crates to `ndarray` 0.17.1.

### Removed

- Archived deprecated benchmark scripts:
  - `speed_comparison.py`
  - `speed_comparisons_with_batch.py`
  - `speed_standard_vs_batch.py`
  - `results_comparisons.py`
  - `speed_comparisons_streaming_10k_batch.py`
  - `simd_vs_standard.py`
- These scripts are superseded by the comprehensive matrix_vs_numpy_sklearn.py benchmark or test internal implementation details not relevant to users.

---

## [0.1.5] - 2025-06-02

### Added

- Unified `cosine_similarity` API: now automatically dispatches to SIMD or batch implementation based on `batch_size`.
- New benchmark script for streaming real embedding datasets from Hugging Face Hub.
- Benchmarks and compares cosine similarity performance across numpy, sklearn, and symrank APIs.
- Enables large-scale, realistic evaluation of similarity search performance.

### Changed

- Updated README with clearer usage examples and parameter descriptions.

### Fixed

- Improved type hints and docstrings for all public APIs.
- Cleaned up `__init__.py` to only expose the main APIs: `cosine_similarity`, `cosine_similarity_batch`, `cosine_similarity_simd`.
- Consistent handling of batch_size=None and batch_size=0 (both now mean "no batching").

---

## [0.1.4] - 2025-05-29

### Added

- Added support for python v3.10 to v3.13
- New benchmark scripts:
  - `speed_comparisons_streaming_10k.py`: Streams and filters 100,001 records from a Hugging Face dataset, then benchmarks SymRank, NumPy, and scikit-learn cosine similarity on 10,000 real embeddings.
  - `speed_comparisons_streaming_10k_batch.py`: Same data streaming and preparation, but benchmarks batch-mode performance for SymRank, NumPy, and scikit-learn.
- Both scripts provide realistic, large-scale performance comparisons using real-world embedding data.

---

## [0.1.3] - 2025-05-28

### Added

- `test_cosine_similarity_batch.py` with new pytest cases covering: batch size greater than the number of candidates, zero batch size treated as default (no batching), batch size of one (per‐vector processing), negative batch size error condition.
- New benchmark script `results_comparisons.py` that compares top-k cosine similarity results and execution times between SymRank, scikit-learn, and NumPy implementations. The script prints normalized timing (ms) and similarity scores for each method, aiding in both performance and correctness validation.
- New benchmark scripts:
  - `speed_comparison.py`: Compares speed of SymRank, scikit-learn, and NumPy cosine similarity implementations on the same dataset.
  - `speed_standard_vs_batch.py`: Benchmarks and compares standard vs. batch SymRank implementations across various dataset and batch sizes.
  - `speed_comparisons_with_batch.py`: Benchmarks batch-mode performance of SymRank, scikit-learn, and NumPy, including global top-k selection.
- All scripts output timing results in milliseconds, supporting robust performance analysis and regression testing.
- New module `cosine_similarity_batch.py` providing `cosine_similarity_batch`, a batching-optimized cosine similarity function. Supports processing large candidate sets in batches with a pre-allocated buffer for efficiency. Recommended for memory-constrained or large-scale similarity search scenarios.
- Expanded unit tests for `cosine_similarity`:
  - Tests for invalid input shapes, mismatched vector sizes, and empty candidate lists.
  - Checks for k=0, k greater than candidate count, identical vectors, integer sequence input, and negative/mixed values.
  - Improved assertions for output structure and correctness.

### Changed

- Updated `Cargo.toml` to optimize Rust release builds:
  - Enabled link-time optimization (`lto = true`), set `codegen-units = 1`, and `opt-level = 3` for improved runtime speed.
  - Clarified crate and module naming in `[lib]` section.
  - Ensured `ndarray` uses Rayon for parallelism and `wide` for SIMD acceleration.
  - Updated dependency versions for improved stability and performance.
- Updated `__init__.py` to expose both `cosine_similarity` and `cosine_similarity_batch` in the public API.
- Refactored Rust core (`lib.rs`)
- Improved input validation and error handling for candidate vector shapes in `cosine_similarity.py`

### Removed

- Removed the following legacy benchmark scripts (replaced by new, improved benchmarking suite):
  - `benchmark_compare_vs_numpy.py`
  - `benchmark_compare_vs_sklearn.py`
  - `benchmark_symrank.py`
  - `speed_comparisons_batch.py`
  - `speed_comparisons.py`
  - `threadpool_info.py`
- Deprecated diagnostics utilities for checking Rayon thread count, environment variables (e.g., OMP_NUM_THREADS, OPENBLAS_NUM_THREADS), and SIMD support have been removed from the Rust core. - This cleanup reduces maintenance burden and eliminates unused code paths.

---

## [0.1.2] - 2025-05-26

### Changed

- Changed all function arguments, variable names, and docstrings from `top_k` to `k` for clarity and consistency.
- `compare()` no longer requires the `vector_size` parameter. Vector dimensionality is now inferred from the `query_vector`.
- Candidate vectors are automatically validated against the inferred size.
- Removed the method parameter from the main similarity function (was previously intended for future extensibility).
- Renamed the main function from compare to cosine for clarity.
- Updated all internal imports, tests, and benchmarks to use cosine.
- API: The main similarity function is now cosine_similarity (was previously compare/cosine).
- File Structure: Renamed cosine.py to cosine_similarity.py for clarity.
- Tests: All tests now use cosine_similarity as the entry point.
- Public Interface: Only cosine_similarity is exposed in the package’s public API.
- Improved Rust cosine similarity kernel by moving the division for query_norm out of the hot loop. Now precomputes the reciprocal of query_norm and uses multiplication inside the scoring loop, reducing the number of divisions and improving throughput for large candidate sets.
- Improved Python wrapper for `cosine_similarity` that adds explicit check for empty `candidate_vectors` to provide a clear error message.
- Now checks that all candidate vectors have the correct dimension.
- Pre-allocates the batch buffer for better performance during batch processing.
- Refactored `speed_comparisons_batch.py` benchmark script to standardized batch processing logic across all benchmark functions

### Fixed

- Improved API clarity and reduced confusion by making the cosine similarity function explicit.

### Removed

- The `vector_size` argument from `compare()` and `_prepare_vector()`. This change simplifies the API and reduces the risk of mismatched vector sizes.
- The method parameter from the Python API.

---

## [0.1.1] - 2025-05-23

### Added

- Added `ci-check.yml` GitHub Actions workflow to validate wheel and sdist builds for all supported platforms and Python versions without uploading to PyPI. This workflow helps ensure all artifacts are correct before release.

### Fixed

- Updated CI workflow to use `--interpreter python` in maturin build arguments, ensuring each job builds only for its matrix Python version and preventing wheel overwriting during artifact merging.
- Fixed CI workflow for Linux wheels to use the correct Python interpreter path inside the manylinux container, ensuring successful builds for all targeted Python versions.

---

## [0.1.0] - 2025-05-23

### Added

- Initial implementation of high-performance brute-force cosine similarity search with a Rust backend and Python bindings via PyO3.
- `cosine_similarity()` function exposed to Python: computes cosine similarity between a query vector and multiple candidate vectors and returns the top-k most similar candidates.
- SIMD-accelerated (wide::f32x8) routines for dot product and norm calculations to maximize performance on modern CPUs.
- Adaptive parallel scoring of candidate vectors using Rayon for efficient multi-threaded computation.
- Inline, heap-based top-k selection algorithm for efficient retrieval of the k most similar candidates.
- Strict validation of C-contiguous memory layout for all vector inputs to ensure optimal performance.
- Python compare() interface: simple, type-safe wrapper that allows direct vector comparison and batching from Python.
- Utility functions for diagnostics (e.g., thread count, SIMD support) in the Rust backend.
- Benchmarks comparing package performance to NumPy and scikit-learn cosine similarity implementations.

### Fixed

- Fixed CI workflow to correctly merge all built wheel and sdist artifacts into the `dist/` directory before publishing to PyPI, preventing missing distribution errors.

---

## [Unreleased]

### Added
<!-- Add new features here -->

### Changed
<!-- Add changed behavior here -->

### Fixed
<!-- Add bug fixes here -->

### Removed
<!-- Add removals/deprecations here -->

---
