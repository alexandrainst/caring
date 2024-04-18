use std::{net::SocketAddr, time::Duration};

use caring::{net::network::TcpNetwork, schemes::{feldman, shamir::ShamirParams}};
use rand::thread_rng;

pub struct AdderEngine {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime,
    threshold: u64,
}

impl AdderEngine {
    pub fn shutdown(self) {
        let AdderEngine { network, runtime, .. } = self;
        runtime.spawn(network.shutdown());
        runtime.shutdown_timeout(Duration::from_secs(5));
    }
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



pub fn mpc_sum(engine: &mut AdderEngine, nums: &[f64]) -> Option<Vec<f64>> {
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

        let my_id = curve25519_dalek::Scalar::from(network.index as u32 + 1);
        let ctx = ShamirParams { ids: parties, threshold: *threshold };
        for share in shares.iter() {
            assert_eq!(share.x, my_id);
        }

        // compute
        let my_result = shares.into_iter().sum();

        let open_shares: Vec<feldman::VecVerifiableShare<_,_>> =
            network.symmetric_broadcast(my_result).await.expect("Publishing shares");

        for (share, id) in open_shares.iter().zip(&ctx.ids) {
            assert_eq!(share.x, *id);
        }

        // reconstruct
        let res = feldman::reconstruct_many(&ctx, &open_shares).expect("Failed to validate")
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


#[derive(Debug)]
pub struct MpcError(pub &'static str);

impl std::fmt::Display for MpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = self.0;
        write!(f, "Bad thing happened: {msg}")
    }
}

impl std::error::Error for MpcError {}

pub fn setup_engine(my_addr: &str, others: &[impl AsRef<str>]) -> Result<AdderEngine, MpcError> {
    let my_addr: SocketAddr = my_addr.parse().unwrap();
    let others: Vec<SocketAddr> = others.iter().map(|s| s.as_ref().parse().unwrap()).collect();

    let threshold = ((others.len() + 1) / 2 + 1) as u64;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let network = TcpNetwork::connect(my_addr, &others);
    let network = runtime.block_on(network).map_err(|_| MpcError("Failed to setup network"))?;
    let engine = AdderEngine {
        network,
        runtime,
        threshold,
    };
    Ok(engine)
}


#[cfg(test)]
mod test {
    use std::time::Duration;

    use super::*;

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

    #[test]
    fn sunshine() {
        use std::thread;
        let t1 = thread::spawn(|| {
            println!("[1] Setting up...");
            let mut engine = setup_engine("127.0.0.1:1234", &["127.0.0.1:1235"]).unwrap();
            println!("[1] Ready");
            let res = mpc_sum(&mut engine, &[32.0]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(|| {
            println!("[2] Setting up...");
            let mut engine = setup_engine("127.0.0.1:1235", &["127.0.0.1:1234"]).unwrap();
            println!("[2] Ready");
            let res = mpc_sum(&mut engine, &[32.0]).unwrap();
            println!("[2] Done");
            drop(engine);
            res
        });
        let a = t1.join().unwrap()[0];
        let b = t2.join().unwrap()[0];
        assert_eq!(a, 64.0);
        assert_eq!(b, 64.0);
    }
}
