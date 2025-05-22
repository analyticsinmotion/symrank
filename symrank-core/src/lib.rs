/// SymRank core library
/// Binds Rust functions to Python module using PyO3.

//mod cosine;
pub mod cosine;
// pub mod heap;
// pub mod sort;
mod diagnostics;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction; // Import wrap_pyfunction manually

/// Initializes the Python module 'symrank_rust'
#[pymodule]
fn symrank(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Explicitly wrap and assign before adding
    let cosine_fn = wrap_pyfunction!(cosine::cosine_similarity, m)?; // Wrap the function
    m.add_function(cosine_fn)?; // Add the wrapped function to the module

    // Diagnostics utilities
    m.add_function(wrap_pyfunction!(diagnostics::rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::get_env_var, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::thread_env_summary, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::simd_supported, m)?)?;
    m.add_function(wrap_pyfunction!(diagnostics::simd_capabilities, m)?)?;

    Ok(())
}
