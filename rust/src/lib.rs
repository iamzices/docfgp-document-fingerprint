//! Convert DOCX and PPTX files to PDF with office2pdf.

use std::path::PathBuf;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Convert a DOCX or PPTX file to PDF bytes.
#[pyfunction]
fn office_to_pdf(py: Python<'_>, path: PathBuf) -> PyResult<Bound<'_, PyBytes>> {
    let result = py
        .detach(|| office2pdf::convert(&path))
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(PyBytes::new(py, &result.pdf))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(office_to_pdf, module)?)
}
