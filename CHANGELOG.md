# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
