use std::{future::Future, ops::DerefMut, sync::Mutex, time::Duration};

use crate::expr::{Id, Opened};
use pyo3::{exceptions::{PyTypeError, PyValueError}, prelude::*, types::PyList};
use wecare::vm;

#[pyclass(frozen)]
pub struct Engine(Mutex<EngineInner>);

struct EngineInner {
    engine: vm::Engine,
    runtime: tokio::runtime::Runtime,
}

#[pyclass(frozen)]
pub struct Computed(vm::Value<vm::UnknownNumber>);

#[pymethods]
impl Computed {
    /// Parse the computed result as a float
    fn as_float(&self) -> Vec<f64> {
        self.0.clone().map(|s| s.to_f64()).to_vec()
    }

    /// Parse the computed result as an integer
    fn as_int(&self) -> Vec<u64> {
        self.0.clone().map(|s| s.to_u64()).to_vec()
    }
}

#[pymethods]
impl Engine {
    /// Construct a new engine connected to the other parties
    ///
    /// * `scheme`: one of {'spdz-25519', 'shamir-25519', 'feldman-25519', 'spdz-32', 'shamir-32'}
    /// * `address`: the address to bind to
    /// * `peers`: the adresses of the other peers to connect to
    /// * `multithreaded`: use a multithreaded runtime
    /// * `threshold`: (optional) threshold if using a threshold scheme
    /// * `preprocessed`: (optional) path to preprocessed material
    #[new]
    #[pyo3(signature = (scheme, address, peers, multithreaded=false, threshold=None, preprocessed_path=None))]
    fn new(
        py: Python<'_>,
        scheme: &str,
        address: &str,
        peers: &Bound<'_, PyList>,
        multithreaded: bool,
        threshold: Option<u64>,
        preprocessed_path: Option<&str>,
    ) -> PyResult<Self> {
        let peers = peers.iter().map(|x| x.extract::<String>().unwrap().clone());

        let (scheme, field) = match scheme {
            "spdz-25519" => (vm::SchemeKind::Spdz, vm::FieldKind::Curve25519),
            "shamir-25519" => (vm::SchemeKind::Shamir, vm::FieldKind::Curve25519),
            "feldman-25519" => (vm::SchemeKind::Shamir, vm::FieldKind::Curve25519),
            "spdz-32" => (vm::SchemeKind::Spdz, vm::FieldKind::Element32),
            "shamir-32" => (vm::SchemeKind::Shamir, vm::FieldKind::Element32),
            _ => return Err(PyValueError::new_err("Unknown scheme")),
        };

        let builder = vm::Engine::builder()
            .address(address)
            .participants_from(peers)
            .scheme(scheme)
            .field(field);

        let builder = builder.threshold(threshold.unwrap_or_default());
        let builder = match preprocessed_path {
            Some(path) => {
                let file = std::fs::File::open(path)?;
                builder.preprocessed(file)
            }
            None => builder,
        };

        let runtime = tokio::runtime::Runtime::new().unwrap();

        let engine = runtime.block_on(async {
            check_signals(py, async {
                builder
                    .connect()
                    .await
                    .map_err(pyo3::exceptions::PyBrokenPipeError::new_err)?
                    .build()
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            })
            .await
        })??;

        let engine = EngineInner { engine, runtime };
        Ok(Self(Mutex::new(engine)))
    }

    /// Execute a script
    ///
    /// * `script`: list of expressions to evaluate
    fn execute(&self, py: Python<'_>, script: &Opened) -> PyResult<Computed> {
        let res = {
            let mut this = self.0.lock().expect("Lock poisoned");
            let script: vm::Opened = script.0.clone();
            let EngineInner { engine, runtime } = this.deref_mut();
            runtime.block_on(check_signals(py, async {
                engine
                    .execute(script)
                    .await
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
            }))??
        };
        Ok(Computed(res))
    }

    /// Your own Id
    fn id(&self) -> Id {
        Id(self.0.lock().unwrap().engine.id())
    }


    /// Sum
    fn sum(&self, py: Python<'_>, num: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let mut this = self.0.lock().expect("Lock poisoned");
        let EngineInner { engine, runtime } = this.deref_mut();
        if let Ok(num) = num.extract::<f64>() {
            runtime.block_on(check_signals(py, async {
                engine
                    .sum(&[num])
                    .await
            }))
        } else if let Ok(nums) = num.extract::<Vec<f64>>() {
            runtime.block_on(check_signals(py, async {
                engine
                    .sum(&nums)
                    .await
            }))
        } else {
            Err(PyTypeError::new_err("num is not a number"))
        }
    }


    /// Your own Id
    fn peers(&self) -> Vec<Id> {
        self.0
            .lock()
            .unwrap()
            .engine
            .peers()
            .into_iter()
            .map(Id)
            .collect()
    }
}


/// Check signals from python routinely while running other future
async fn check_signals<F: Future>(py: Python<'_>, f: F) -> Result<F::Output, PyErr> {
    let signals = async {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            match py.check_signals() {
                Ok(_) => continue,
                Err(err) => break err,
            }
        }
    };

    tokio::select! {
        err = signals => {
            Err(err)
        },
        res = f => {
            Ok(res)
        },
    }
}
