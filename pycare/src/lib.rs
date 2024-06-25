use pyo3::{exceptions::PyIOError, prelude::*, types::PyTuple};

use std::fs::File;
use wecare::*;

#[pyclass]
struct Engine(Option<wecare::Engine>);

/// Setup a MPC addition engine connected to the given sockets using SPDZ.
#[pyfunction]
#[pyo3(signature = (path_to_pre, my_addr, *others))]
fn spdz(path_to_pre: &str, my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<Engine> {
    let others: Vec<_> = others
        .iter()
        .map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    let mut file = File::open(path_to_pre).unwrap();
    match wecare::Engine::setup(my_addr)
        .add_participants(&others)
        .file_to_preprocessed(&mut file)
        .build_spdz() {
        Ok(e) => Ok(Engine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0)),
    }
}


/// Setup a MPC addition engine connected to the given sockets using shamir secret sharing.
#[pyfunction]
#[pyo3(signature = (threshold, my_addr, *others))]
fn shamir(threshold: u32, my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<Engine> {
    let others: Vec<_> = others
        .iter()
        .map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    match wecare::Engine::setup(my_addr)
        .add_participants(&others)
        .threshold(threshold as u64)
        .build_shamir() {
        Ok(e) => Ok(Engine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0)),
    }
}

/// Calculate and save the preprocessing
#[pyfunction]
#[pyo3(signature = (number_of_shares, paths_to_pre))]
fn preproc(number_of_shares: usize, paths_to_pre: &str) {
    let mut files: Vec<File> = paths_to_pre
        .split(",")
        .map(|p| File::create(p).unwrap())
        .collect();
    do_preproc(&mut files, vec![number_of_shares, number_of_shares]);
}

#[pymethods]
impl Engine {
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
    m.add_function(wrap_pyfunction!(spdz, m)?)?;
    m.add_function(wrap_pyfunction!(shamir, m)?)?;
    m.add_function(wrap_pyfunction!(preproc, m)?)?;
    m.add_class::<Engine>()?;
    Ok(())
}
