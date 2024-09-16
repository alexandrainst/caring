use std::sync::Mutex;

use crate::expr::{Id, Opened};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use wecare::vm;

#[pyclass(frozen)]
pub struct Engine(Mutex<vm::blocking::Engine>);

#[pyclass(frozen)]
pub struct Computed(vm::Value<vm::UnknownNumber>);

#[pymethods]
impl Computed {
    /// Parse the computed result as a float
    fn as_float(&self) -> Vec<f64> {
        self.0.clone().map(|s| s.to_f64()).to_vec()
    }

    /// Parse the computed result as an integer
    fn as_integer(&self) -> Vec<u64> {
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
    #[pyo3(signature = (scheme, address, peers, multithreaded=false, threshold=None, preprocessed=None))]
    fn new(
        scheme: &str,
        address: &str,
        peers: &Bound<'_, PyList>,
        multithreaded: bool,
        threshold: Option<u64>,
        preprocessed: Option<&str>,
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
        let builder = match preprocessed {
            Some(path) => {
                let file = std::fs::File::open(path)?;
                builder.preprocessed(file)
            }
            None => builder,
        };

        let builder = if multithreaded {
            builder.multi_threaded_runtime()
        } else {
            builder.single_threaded_runtime()
        };

        let builder = builder.connect_blocking().unwrap();
        let engine = builder.build();
        Ok(Self(Mutex::new(engine)))
    }

    /// Execute a script
    ///
    /// * `script`: list of expressions to evaluate
    fn execute(&self, script: &Opened) -> PyResult<Computed> {
        let res = {
            let mut engine = self.0.lock().expect("Lock poisoned");
            let script: vm::Opened = script.0.clone();
            engine
                .execute(script)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        };
        Ok(Computed(res))
    }

    /// Your own Id
    fn id(&self) -> Id {
        Id(self.0.lock().unwrap().id())
    }

    /// Your own Id
    fn peers(&self) -> Vec<Id> {
        self.0.lock().unwrap().peers().into_iter().map(Id).collect()
    }
}
