//! Convert DOCX and PPTX files to PDF and compute MinHash minima natively.

use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const PRIME: u64 = (1 << 61) - 1;

/// Convert a DOCX or PPTX file to PDF bytes.
#[pyfunction]
fn office_to_pdf(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyBytes>> {
    let result = py
        .detach(|| office2pdf::convert(&path))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result.pdf))
}

// Reduce by folding, since 2^61 is congruent to 1 modulo the Mersenne prime.
fn mod_prime(value: u128) -> u64 {
    let prime = u128::from(PRIME);
    let folded = (value & prime) + (value >> 61);
    let folded = (folded & prime) + (folded >> 61);
    (if folded >= prime { folded - prime } else { folded }) as u64
}

/// Return the minimum of each `(a * x + b) mod 2^61 - 1` hash over the values.
#[pyfunction]
fn minhash(py: Python<'_>, values: Vec<u64>, coefficients: Vec<(u64, u64)>) -> Vec<u64> {
    py.detach(|| {
        coefficients
            .iter()
            .map(|&(a, b)| {
                values
                    .iter()
                    .map(|&x| mod_prime(u128::from(a) * u128::from(x) + u128::from(b)))
                    .min()
                    .unwrap_or(PRIME)
            })
            .collect()
    })
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(office_to_pdf, module)?)?;
    module.add_function(wrap_pyfunction!(minhash, module)?)
}
