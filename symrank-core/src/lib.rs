/// SymRank core library
/// Binds Rust functions to Python module using PyO3.

pub mod cosine_similarity;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction; // Import wrap_pyfunction manually

/// Initializes the Python module 'symrank'
#[pymodule]
fn symrank(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Explicitly wrap and assign before adding
    let cosine_fn = wrap_pyfunction!(cosine_similarity::cosine_similarity, m)?; // Wrap the function
    m.add_function(cosine_fn)?; // Add the wrapped function to the module

    Ok(())
}
