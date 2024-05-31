use pyo3::{prelude::*, types::PyTuple, exceptions::PyIOError};

use wecare::*;
use std::path::Path;

#[pyclass]
struct Engine(Option<AdderEngine>);

/// Setup a MPC addition engine connected to the given sockets.
#[pyfunction]
#[pyo3(signature = (path_to_pre, my_addr, *others))]

fn setup(path_to_pre: &str, my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<Engine> {
    let others : Vec<_> = others.iter().map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    let file_name = Path::new(path_to_pre);
    match setup_engine(my_addr, &others, file_name) {
        Ok(e) => Ok(Engine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0))
    }
}

/// Calculate and save the preprocessing
#[pyfunction]
#[pyo3(signature = (number_of_shares, paths_to_pre))]
fn preproc( number_of_shares: usize, paths_to_pre: &str){
    let paths_to_pre = paths_to_pre.split(",").collect();
    do_preproc(paths_to_pre, vec![number_of_shares, number_of_shares]);
}


#[pymethods]
impl Engine {

    /// Run a sum procedure in which each party supplies a double floating point
    fn sum(&mut self, a: f64) -> f64 {
        mpc_sum(self.0.as_mut().unwrap(), &[a]).unwrap()[0]
    }

    /// Run a sum procedure in which each party supplies a double floating point
    fn sum_many(&mut self, a: Vec<f64>) -> Vec<f64> {
        mpc_sum(self.0.as_mut().unwrap(), &a).unwrap()
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
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(preproc, m)?)?;
    m.add_class::<Engine>()?;
    Ok(())
}

