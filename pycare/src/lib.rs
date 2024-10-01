use pyo3::{prelude::*, types::PyTuple};

pub mod expr;
pub mod vm;

use std::fs::File;
use wecare::*;

/// Calculate and save the preprocessing
#[pyfunction]
#[pyo3(signature = (num_shares, num_triplets, *paths_to_pre, scheme="spdz-25519"))]
fn preproc(
    num_shares: usize,
    num_triplets: usize,
    paths_to_pre: &Bound<'_, PyTuple>,
    scheme: &str,
) -> PyResult<()> {
    let mut files: Vec<File> = paths_to_pre
        .iter()
        .map(|x| {
            x.extract::<String>()
                .and_then(|name| File::create(name).map_err(|e| e.into()))
        })
        .collect::<PyResult<_>>()?;

    match scheme {
        "spdz-25519" => do_preproc(&mut files, &[num_shares, num_shares], num_triplets, false),
        "spdz-32" => do_preproc(&mut files, &[num_shares, num_shares], num_triplets, true),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid scheme: '{scheme}'"
            )));
        }
    }
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn caring(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO: enable this
    // pyo3_log::init();

    // TODO: disable this
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::from_default_env();
    tracing_subscriber::fmt().with_env_filter(filter).init();

    m.add_function(wrap_pyfunction!(preproc, m)?)?;
    m.add_class::<vm::Engine>()?;
    m.add_class::<vm::Computed>()?;
    m.add_class::<expr::Expr>()?;
    m.add_class::<expr::Id>()?;
    m.add_class::<expr::Opened>()?;
    Ok(())
}
