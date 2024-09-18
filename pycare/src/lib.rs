use pyo3::{exceptions::PyIOError, prelude::*, types::PyTuple};

pub mod expr;
pub mod vm;

use std::fs::File;
use wecare::*;

#[pyclass]
struct OldEngine(Option<wecare::Engine>);

/// Setup a MPC addition engine connected to the given sockets using SPDZ.
#[pyfunction]
#[pyo3(signature = (path_to_pre, my_addr, *others))]
fn spdz(path_to_pre: &str, my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<OldEngine> {
    let others: Vec<_> = others
        .iter()
        .map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    let mut file = File::open(path_to_pre).unwrap();
    match wecare::Engine::setup(my_addr)
        .add_participants(&others)
        .file_to_preprocessed(&mut file)
        .build_spdz()
    {
        Ok(e) => Ok(OldEngine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0)),
    }
}

/// Setup a MPC addition engine connected to the given sockets using shamir secret sharing.
#[pyfunction]
#[pyo3(signature = (threshold, my_addr, *others))]
fn shamir(threshold: u32, my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<OldEngine> {
    let others: Vec<_> = others
        .iter()
        .map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    match wecare::Engine::setup(my_addr)
        .add_participants(&others)
        .threshold(threshold as u64)
        .build_shamir()
    {
        Ok(e) => Ok(OldEngine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0)),
    }
}

/// Setup a MPC addition engine connected to the given sockets using shamir secret sharing.
#[pyfunction]
#[pyo3(signature = (threshold, my_addr, *others))]
fn feldman(threshold: u32, my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<OldEngine> {
    let others: Vec<_> = others
        .iter()
        .map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    match wecare::Engine::setup(my_addr)
        .add_participants(&others)
        .threshold(threshold as u64)
        .build_feldman()
    {
        Ok(e) => Ok(OldEngine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0)),
    }
}

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

#[pymethods]
impl OldEngine {
    /// Run a sum procedure in which each party supplies a double floating point
    fn sum(&mut self, a: f64) -> f64 {
        self.0.as_mut().unwrap().mpc_sum(&[a]).unwrap()[0]
    }

    /// Run a sum procedure in which each party supplies a double floating point
    fn sum_many(&mut self, a: Vec<f64>) -> Vec<f64> {
        self.0.as_mut().unwrap().mpc_sum(&a).unwrap()
    }

    /// takedown engine
    fn takedown(&mut self) {
        let engine = self.0.take().unwrap();
        engine.shutdown();
    }
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

    m.add_function(wrap_pyfunction!(spdz, m)?)?;
    m.add_function(wrap_pyfunction!(shamir, m)?)?;
    m.add_function(wrap_pyfunction!(feldman, m)?)?;
    m.add_function(wrap_pyfunction!(preproc, m)?)?;
    m.add_class::<OldEngine>()?;
    m.add_class::<vm::Engine>()?;
    m.add_class::<vm::Computed>()?;
    m.add_class::<expr::Expr>()?;
    m.add_class::<expr::Id>()?;
    m.add_class::<expr::Opened>()?;
    Ok(())
}
