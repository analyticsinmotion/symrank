use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use ndarray::parallel::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use ordered_float::NotNan;
use wide::f32x8; 

/// Compute cosine similarity between a query vector and candidate vectors.
/// Returns the top-k highest scoring candidates as (index, score).
/// Uses adaptive serial/parallel scoring and always inlines heap-based top-k selection.
///#[pyfunction]
#[pyfunction(text_signature = "(query, references, k, /)")]
pub fn cosine_similarity(
    query: PyReadonlyArray1<f32>,
    references: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if !query.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Query array must be C-contiguous. Use np.ascontiguousarray(query) in Python."
        ));
    }
    if !references.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Reference array must be C-contiguous. Use np.ascontiguousarray(candidates) in Python."
        ));
    }

    let query = query.as_slice().map_err(|_| PyValueError::new_err("Failed to convert query to slice"))?;

    // --- Ensure standard (C-contiguous) layout for SIMD ---
    let references = references.as_array();
    let references = references.as_standard_layout();

    if references.ndim() != 2 || references.shape()[1] != query.len() {
        return Err(PyValueError::new_err(format!(
            "Candidate vectors must have shape (N, {}). Got shape {:?}",
            query.len(),
            references.shape()
        )));
    }

    //let query_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let query_norm = fused_norm(query);
    let inv_query_norm = 1.0 / query_norm; // Precompute reciprocal

    // Adaptive serial/parallel scoring
    let n_refs = references.shape()[0];
    let threads = rayon::current_num_threads();
    let crossover = if threads >= 16 {
        300
    } else if threads >= 8 {
        600
    } else {
        1000
    };

    let scored: Vec<(usize, f32)> = if n_refs < crossover {
        // Serial
        references
            .outer_iter()
            .enumerate()
            .map(|(i, ref_vec)| {
                let ref_slice = ref_vec.as_slice().unwrap();
                let (dot, norm2) = fused_dot_and_norm(query, ref_slice);
                // let sim = dot / (query_norm * norm2);
                let inv_norm2 = 1.0 / norm2;
                let sim = dot * inv_query_norm * inv_norm2; // Use multiplication instead of division
                (i, sim)
            })
            .collect()
    } else {
        // Parallel
        references
            .axis_iter(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(i, ref_vec)| {
                let ref_slice = ref_vec.as_slice().unwrap();
                let (dot, norm2) = fused_dot_and_norm(query, ref_slice);
                // let sim = dot / (query_norm * norm2);
                let inv_norm2 = 1.0 / norm2;
                let sim = dot * inv_query_norm * inv_norm2; // Use multiplication instead of division
                (i, sim)
            })
            .collect()
    };

    // Inline heap-based top-k selection
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (i, score) in scored {
        let score = NotNan::new(score).unwrap_or(NotNan::new(0.0).unwrap());
        heap.push(Reverse((score, i)));
        if heap.len() > k {
            heap.pop();
        }
    }
    let mut result: Vec<_> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse((score, i))| (i, score.into_inner()))
        .collect();
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(result)
    
}

// #[inline(always)]
// fn fused_dot_and_norm(v1: &[f32], v2: &[f32]) -> (f32, f32) {
//     let mut dot = 0.0;
//     let mut norm2 = 0.0;
//     for (a, b) in v1.iter().zip(v2.iter()) {
//         dot += a * b;
//         norm2 += b * b;
//     }
//     (dot, norm2.sqrt())
// }

// SIMD-accelerated fused dot and norm using wide
#[inline(always)]
fn fused_dot_and_norm(v1: &[f32], v2: &[f32]) -> (f32, f32) {

    // Vectorized computation for main chunks
    let mut dot = f32x8::splat(0.0);
    let mut norm2 = f32x8::splat(0.0);

    let chunks = v1.chunks_exact(8);
    let remainder1 = chunks.remainder();
    let chunks2 = v2.chunks_exact(8);
    let remainder2 = chunks2.remainder();

    // for (a, b) in chunks.zip(chunks2) {
    //     // SAFETY: chunks_exact guarantees slices of length 8
    //     let a = f32x8::new([a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],]);
    //     let b = f32x8::new([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],]);
    //     dot += a * b;
    //     norm2 += b * b;
    // }

    for (a, b) in chunks.zip(chunks2) {
    // SAFETY: chunks_exact guarantees slices of length 8
    let a = f32x8::new(*<&[f32; 8]>::try_from(a).unwrap());
    let b = f32x8::new(*<&[f32; 8]>::try_from(b).unwrap());
    dot += a * b;
    norm2 += b * b;
    //dot = a.mul_add(b, dot);
    //norm2 = b.mul_add(b, norm2);
}

    // Scalar fallback for remainder
    let mut dot_scalar = dot.reduce_add();
    let mut norm2_scalar = norm2.reduce_add();

    for (&a, &b) in remainder1.iter().zip(remainder2.iter()) {
        dot_scalar += a * b;
        norm2_scalar += b * b;
    }
    (dot_scalar, norm2_scalar.sqrt())
}

#[inline(always)]
fn fused_norm(v: &[f32]) -> f32 {

    let mut acc = f32x8::splat(0.0);

    let chunks = v.chunks_exact(8);
    let remainder = chunks.remainder();

    // for a in chunks {
    //     let a = f32x8::new([
    //         a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
    //     ]);
    //     acc += a * a;
    // }

    for a in chunks {
    let a = f32x8::new(*<&[f32; 8]>::try_from(a).unwrap());
    acc += a * a;
}

    let mut sum = acc.reduce_add();
    for &x in remainder {
        sum += x * x;
    }
    sum.sqrt()
}
