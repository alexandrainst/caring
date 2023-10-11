use std::{sync::Mutex, net::{SocketAddrV4, SocketAddr}};

use fixed::traits::Fixed;
use pyo3::{prelude::*, types::{PyDict, PyTuple}};

use caring::{shamir, connection::TcpNetwork};
use rand::thread_rng;

struct AdderEngine {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime
}

static ENGINE : Mutex<Option<Box<AdderEngine>>> = Mutex::new(None);
fn mpc_sum(num: f64) -> f64 {
    let mut engine = ENGINE.lock().unwrap();
    let engine = engine.as_mut().unwrap().as_mut();
    let AdderEngine { network, runtime } = engine;

    type Fix = fixed::FixedU64<32>;
    let num = Fix::from_num(num).to_bits();
    let res = runtime.block_on(async {
        let num = curve25519_dalek::Scalar::from(num);
        
        // construct
        let parties: Vec<_> = network
            .participants()
            .map(|id| (id + 1))
            .map(curve25519_dalek::Scalar::from)
            .collect();

        let mut rng = thread_rng();
        let shares = shamir::share::<curve25519_dalek::Scalar>(num, &parties, 2, &mut rng);

        // share my shares.
        let shares = network.symmetric_unicast(shares).await;

        // compute
        let my_result = shares.into_iter().sum();
        let open_shares = network.symmetric_broadcast(my_result).await;

        // reconstruct
        let res = shamir::reconstruct(&open_shares);
        let res: [u8; 8] = res.as_bytes()[0..8].try_into().unwrap();
        u64::from_le_bytes(res)
    });

    let res = Fix::from_bits(res);
    res.to_num()
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
#[pyo3(signature = (my_addr, *others))]
fn setup(my_addr: &str, others: &PyTuple) -> PyResult<()> {
    let my_addr : SocketAddr = my_addr.parse().unwrap();
    let others : Vec<SocketAddr> = others.iter().map(|x| x.extract().unwrap())
        .map(|s: &str| s.parse().unwrap())
        .collect();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build().unwrap();
    let network = TcpNetwork::connect(my_addr, &others);
    let network = runtime.block_on(network);
    let engine =  AdderEngine { network, runtime };
    ENGINE.lock().unwrap().replace(Box::new(engine));
    Ok(())
}

#[pyfunction]
fn sum(a: f64) -> f64 {
    mpc_sum(a)
}

/// A Python module implemented in Rust.
#[pymodule]
fn pycare(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    Ok(())
}
