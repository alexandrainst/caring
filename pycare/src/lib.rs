use pyo3::{prelude::*, types::PyTuple, exceptions::PyIOError};

use wecare::*;

#[pyclass]
struct Engine(Option<AdderEngine>);

/// Setup a MPC addition engine connected to the given sockets.
#[pyfunction]
#[pyo3(signature = (my_addr, *others))]
fn setup(my_addr: &str, others: &Bound<'_, PyTuple>) -> PyResult<Engine> {
    let others : Vec<_> = others.iter().map(|x| x.extract::<String>().unwrap().clone())
        .collect();
    match setup_engine(my_addr, &others) {
        Ok(e) => Ok(Engine(Some(e))),
        Err(e) => Err(PyIOError::new_err(e.0))
    }
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
    m.add_class::<Engine>()?;
    Ok(())
}

