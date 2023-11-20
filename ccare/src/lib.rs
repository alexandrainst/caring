use core::slice::{self};
use std::{
    ffi::{CStr, c_char},
    net::SocketAddr,
    sync::Mutex, panic,
};

use caring::{network::TcpNetwork, schemes::feldman};
use rand::thread_rng;

struct AdderEngine {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime,
    threshold: u64,
}

type SignedFix = fixed::FixedI128<64>;
fn to_offset(num: f64) -> u128 {
    let num: i128 = SignedFix::from_num(num).to_bits(); // convert to signed fixed point
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
    let a: Scalar = to_offset(1.1).into();
    let b: Scalar = to_offset(2.2).into();
    let c: Scalar = to_offset(3.3).into();
    let sum = a + b + c;
    let sum: [u8; 16] = sum.as_bytes()[0..16].try_into().unwrap();
    let sum = u128::from_le_bytes(sum);
    let sum = from_offset(sum);
    assert!(sum - 6.6 < 0.01);
}

static ENGINE: Mutex<Option<Box<AdderEngine>>> = Mutex::new(None);
fn mpc_sum(nums: &[f64]) -> Option<Vec<f64>> {
    let mut engine = ENGINE.lock().unwrap();
    let engine = engine.as_mut().unwrap().as_mut();
    let AdderEngine {
        network,
        runtime,
        threshold,
    } = engine;
    let nums: Vec<_> = nums
        .iter()
        .map(|&num| {
            let num = to_offset(num);

            curve25519_dalek::Scalar::from(num)
        })
        .collect();

    let res : Option<_> = runtime.block_on(async {
        // construct
        let parties: Vec<_> = network
            .participants()
            .map(|id| (id + 1))
            .map(curve25519_dalek::Scalar::from)
            .collect();

        let mut rng = thread_rng();
        let shares = feldman::share_many::<
            curve25519_dalek::Scalar,
            curve25519_dalek::RistrettoPoint,
        >(&nums, &parties, *threshold, &mut rng);

        // share my shares.
        let shares = network.symmetric_unicast(shares).await.expect("Sharing shares");

        // compute
        let my_result = shares.into_iter().sum();
        let open_shares: Vec<feldman::VecVerifiableShare<_, _>> =
            network.symmetric_broadcast(my_result).await.expect("Publishing shares");

        // reconstruct
        let res = feldman::reconstruct_many(&open_shares)?
            .into_iter()
            .map(|x| x.as_bytes()[0..128/8].try_into().expect("Should be infalliable"))
            .map(u128::from_le_bytes)
            .map(from_offset)
            .collect();
        // NOTE: Since we are only using half of this space, we have
        // a possibility of 'checking' for computation failures.
        Some(res)
    });
    res
}

fn setup_engine(my_addr: &str, others: &[&str]) -> Result<(), ()> {
    let my_addr: SocketAddr = my_addr.parse().unwrap();
    let others: Vec<SocketAddr> = others.iter().map(|s| s.parse().unwrap()).collect();

    let threshold = ((others.len() + 1) / 2 + 1) as u64;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let network = TcpNetwork::connect(my_addr, &others);
    let network = runtime.block_on(network).expect("Failed to set up network");
    let engine = AdderEngine {
        network,
        runtime,
        threshold,
    };
    ENGINE.lock().unwrap().replace(Box::new(engine));
    Ok(())
}


/// # Safety
/// Strings must null terminated and the array have length `len`
#[no_mangle]
pub unsafe extern "C" fn care_setup(my_addr: *const c_char, others: *const *const c_char, len: usize) -> i32 {
    let my_addr = unsafe { CStr::from_ptr(my_addr) };
    let Ok(my_addr) = my_addr.to_str() else {
        return 1;
    };

    let Ok(others) = (unsafe {
        let others = slice::from_raw_parts(others, len);
        let others: Result<Vec<_>, _> = others
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr))
            .map(|cstr| cstr.to_str())
            .collect();
        others
    }) else {
        return 2;
    };
    let Ok(_) = setup_engine(my_addr, &others) else {
        return 3;
    };
    0
}

/// Run a sum procedure in which each party supplies a double floating point
///
/// Returns `NaN` on unknown panics.
///
#[no_mangle]
pub extern "C" fn care_sum(a: f64) -> f64 {
    let result = panic::catch_unwind(|| {
        mpc_sum(&[a])
    });
    let Ok(result) = result else {
        return f64::NAN;
    };
    let Some(result) = result else {
        return -1.0;
    };
    result[0]
}

/// Compute many sums (vectorized)
///
/// # Safety
/// The `buf` pointer must have length `len`.
///
/// Returns `-1` on unknown panics.
///
#[no_mangle]
pub unsafe extern "C" fn care_sum_many(buf: *mut f64, len: usize) -> i32 {
    let input: &[_] = unsafe { slice::from_raw_parts(buf, len) };
    let result = panic::catch_unwind(|| {
        mpc_sum(input)
    });
    let Ok(res) = result else {
        return -1;
    };
    let Some(res) = res else {
        return 1;
    };
    let res = res.as_slice().as_ptr();
    unsafe { std::ptr::copy(res, buf, len) };
    0
}

/// Takedown the MPC engine, freeing the memory and dropping the connections
#[no_mangle]
pub extern "C" fn care_takedown() {
    ENGINE.lock().unwrap().as_mut().take();
}
