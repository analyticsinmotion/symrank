use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt; // Needed for into_py_any
use rayon;                 // Your parallelism library
use std::env;

/// Returns the number of Rayon threads.
#[pyfunction]
pub fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Returns an environment variable if it exists.
#[pyfunction]
pub fn get_env_var(name: String) -> Option<String> {
    env::var(&name).ok()
}

/// Returns a dictionary of selected environment variables relevant to threading.
#[pyfunction]
pub fn thread_env_summary(py: Python) -> PyResult<PyObject> {
    let vars = vec![
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ];

    let dict = PyDict::new(py);
    for var in vars {
        if let Ok(val) = env::var(var) {
            dict.set_item(var, val)?;
        }
    }

    dict.into_py_any(py).map(|obj| obj.into())
}

/// A basic compile-time SIMD check (does not catch all).
#[pyfunction]
pub fn simd_supported() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        false
    }
}

/// A runtime SIMD feature detector using is_x86_feature_detected.
#[pyfunction]
pub fn simd_capabilities() -> PyResult<Vec<String>> {
    let mut features = vec![];

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f") {
            features.push("avx512f".to_string());
        }
        if std::is_x86_feature_detected!("avx2") {
            features.push("avx2".to_string());
        }
        if std::is_x86_feature_detected!("avx") {
            features.push("avx".to_string());
        }
        if std::is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2".to_string());
        }
        if std::is_x86_feature_detected!("fma") {
            features.push("fma".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        features.push("neon".to_string()); // Almost always supported
    }

    Ok(features)
}
