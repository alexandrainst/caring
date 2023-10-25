use std::{sync::Mutex, net::SocketAddr};
use curve25519_dalek::Scalar;
use pyo3::{prelude::*, types::{PyTuple, PyList}};

use caring::{schemes::feldman, connection::TcpNetwork};
use rand::thread_rng;

struct AdderEngine {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime,
    threshold: u64
}

type SignedFix = fixed::FixedI128<64>;
fn to_offset(num: f64) -> u128 {
    let num : i128 = SignedFix::from_num(num).to_bits(); // convert to signed fixed point
    // Okay this a some voodoo why this 'just works', but the idea is that
    // twos complement is actually just an offset from the max value.
    // https://en.wikipedia.org/wiki/Offset_binary
    num as u128
}

fn from_offset(num: u128) -> f64 {
    // Same applies as above
    let num = SignedFix::from_bits(num as i128);
    num.to_num()
}


#[test]
fn offset_binary() {
    use curve25519_dalek::Scalar;
    let a : Scalar  =  to_offset(1.1).into();
    let b : Scalar = to_offset(2.2).into();
    let c : Scalar  = to_offset(3.3).into();
    let sum = a + b + c;
    let sum: [u8; 16] = sum.as_bytes()[0..16].try_into().unwrap();
    let sum = u128::from_le_bytes(sum);
    let sum = from_offset(sum);
    assert!(sum - 6.6 < 0.01);
}

static ENGINE : Mutex<Option<Box<AdderEngine>>> = Mutex::new(None);
fn mpc_sum(nums: &[f64]) -> Vec<f64> {
    let mut engine = ENGINE.lock().unwrap();
    let engine = engine.as_mut().unwrap().as_mut();
    let AdderEngine { network, runtime, threshold } = engine;
    let nums : Vec<_> = nums.into_iter().map(|&num| {
        let num = to_offset(num);
        
        curve25519_dalek::Scalar::from(num)
    }).collect();

    let res = runtime.block_on(async {

        // construct
        let parties: Vec<_> = network
            .participants()
            .map(|id| (id + 1))
            .map(curve25519_dalek::Scalar::from)
            .collect();

        let mut rng = thread_rng();
        let shares = feldman::share_many::<curve25519_dalek::Scalar, curve25519_dalek::RistrettoPoint>(&nums, &parties, *threshold, &mut rng);

        // share my shares.
        let shares = network.symmetric_unicast(shares).await;

        // compute
        let my_result = shares.into_iter().sum();
        let open_shares : Vec<feldman::VecVerifiableShare<_,_>> = network.symmetric_broadcast(my_result).await;

        // reconstruct
        let res = feldman::reconstruct_many(&open_shares).unwrap();
        // NOTE: Since we are only using half of this space, we have
        // a possibility of 'checking' for computation failures.
        res.iter().map(|res : &Scalar| {
            let res =  res.as_bytes()[0..16].try_into().unwrap();
            let res = u128::from_le_bytes(res);
            from_offset(res)
        }).collect()
    });
    res
}

/// Setup a MPC addition engine connected to the given sockets.
#[pyfunction]
#[pyo3(signature = (my_addr, *others))]
fn setup(my_addr: &str, others: &PyTuple) -> PyResult<()> {
    let my_addr : SocketAddr = my_addr.parse().unwrap();
    let others : Vec<SocketAddr> = others.iter().map(|x| x.extract().unwrap())
        .map(|s: &str| s.parse().unwrap())
        .collect();

    let threshold = ((others.len() + 1) / 2 + 1) as u64; 
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build().unwrap();
    let network = TcpNetwork::connect(my_addr, &others);
    let network = runtime.block_on(network);
    let engine =  AdderEngine { network, runtime, threshold };
    ENGINE.lock().unwrap().replace(Box::new(engine));
    Ok(())
}

/// Run a sum procedure in which each party supplies a double floating point
#[pyfunction]
fn sum(a: f64) -> f64 {
    mpc_sum(&[a])[0]
}


#[pyfunction]
fn sum_many(a: Vec<f64>) -> Vec<f64> {
    mpc_sum(&a)
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

