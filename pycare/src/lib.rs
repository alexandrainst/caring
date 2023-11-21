use pyo3::{prelude::*, types::PyTuple, exceptions::PyIOError};

use wecare::*;

/// Setup a MPC addition engine connected to the given sockets.
#[pyfunction]
#[pyo3(signature = (my_addr, *others))]
fn setup(my_addr: &str, others: &PyTuple) -> PyResult<()> {
    let others : Vec<_> = others.iter().map(|x| x.extract().unwrap())
        .collect();
    match setup_engine(my_addr, &others) {
        Ok(_) => Ok(()),
        Err(e) => Err(PyIOError::new_err(e.0))
    }

}

/// Run a sum procedure in which each party supplies a double floating point
#[pyfunction]
fn sum(a: f64) -> f64 {
    mpc_sum(&[a]).unwrap()[0]
}


#[pyfunction]
fn sum_many(a: Vec<f64>) -> Vec<f64> {
    mpc_sum(&a).unwrap()
}

/// Takedown the MPC engine, freeing the memory and dropping the connections
#[pyfunction]
fn takedown() {
    ENGINE.lock().unwrap().as_mut().take();
}

/// A Python module implemented in Rust.
#[pymodule]
fn pycare(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(sum_many, m)?)?;
    m.add_function(wrap_pyfunction!(takedown, m)?)?;
    Ok(())
}

