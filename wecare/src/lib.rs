use caring::{
    algebra::math::Vector,
    net::network::TcpNetwork,
    schemes::{
        interactive::InteractiveSharedMany,
        shamir,
        spdz::{self, preprocessing},
    },
};
use rand::{thread_rng, SeedableRng};
use std::{fs::File, mem, net::SocketAddr, time::Duration};

type F = curve25519_dalek::Scalar;

pub struct AdderEngine<S: InteractiveSharedMany> {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime,
    context: S::Context,
}

impl<S> AdderEngine<S>
where
    S: InteractiveSharedMany<Value = curve25519_dalek::Scalar>,
{
    //pub fn setup_engine(my_addr: &str, others: &[impl AsRef<str>], file_name: String) -> Result<AdderEngine<F>, MpcError> {
    pub fn setup(
        my_addr: &str,
        others: &[impl AsRef<str>],
        context: S::Context,
    ) -> Result<Self, MpcError> {
        let my_addr: SocketAddr = my_addr.parse().unwrap();
        let others: Vec<SocketAddr> = others.iter().map(|s| s.as_ref().parse().unwrap()).collect();

        //let threshold = ((others.len() + 1) / 2 + 1) as u64;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let network = TcpNetwork::connect(my_addr, &others);
        let network = runtime
            .block_on(network)
            .map_err(|_| MpcError("Failed to setup network"))?;
        let engine = AdderEngine {
            network,
            runtime,
            context,
        };
        Ok(engine)
    }

    pub fn shutdown(self) {
        let AdderEngine {
            network, runtime, ..
        } = self;
        runtime.spawn(network.shutdown());
        runtime.shutdown_timeout(Duration::from_secs(5));
    }

    pub fn mpc_sum(&mut self, nums: &[f64]) -> Option<Vec<f64>>
    where
        <S as InteractiveSharedMany>::VectorShare: std::iter::Sum,
    {
        let AdderEngine {
            network,
            runtime,
            context,
        } = self;

        let nums: Vec<_> = nums
            .iter()
            .map(|&num| {
                let num = to_offset(num);
                F::from(num)
            })
            .collect();
        let rng = rand::rngs::StdRng::from_rng(thread_rng()).unwrap();
        let res: Option<_> = runtime.block_on(async move {
            let ctx = context;
            let mut network = network;
            // construct
            let shares: Vec<S::VectorShare> =
                S::symmetric_share_many(ctx, &nums, rng, &mut network)
                    .await
                    .unwrap();
            let sum: S::VectorShare = shares.into_iter().sum();
            let res: Vector<F> = S::recombine_many(ctx, sum, network).await.unwrap();

            let res = res
                .into_iter()
                .map(|x| u128::from_le_bytes(x.as_bytes()[0..128 / 8].try_into().unwrap()))
                .map(from_offset)
                .collect();
            Some(res)
        });
        res
    }
}

impl AdderEngine<spdz::Share<curve25519_dalek::Scalar>> {
    pub fn from_preprocess(
        my_addr: &str,
        others: &[impl AsRef<str>],
        file: &mut File,
    ) -> Result<Self, MpcError> {
        let my_addr: SocketAddr = my_addr.parse().unwrap();
        let others: Vec<SocketAddr> = others.iter().map(|s| s.as_ref().parse().unwrap()).collect();
        let mut context = preprocessing::read_preproc_from_file(file);

        //let threshold = ((others.len() + 1) / 2 + 1) as u64;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let network = TcpNetwork::connect(my_addr, &others);
        let network = runtime
            .block_on(network)
            .map_err(|_| MpcError("Failed to setup network"))?;
        context.params.who_am_i = network.index;
        let engine = AdderEngine {
            network,
            runtime,
            context,
        };
        Ok(engine)
    }
}

impl AdderEngine<shamir::Share<curve25519_dalek::Scalar>> {
    pub fn shamir(my_addr: &str, others: &[impl AsRef<str>]) -> Result<Self, MpcError> {
        let my_addr: SocketAddr = my_addr.parse().unwrap();
        let others: Vec<SocketAddr> = others.iter().map(|s| s.as_ref().parse().unwrap()).collect();

        let threshold = ((others.len() + 1) / 2 + 1) as u64;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let network = TcpNetwork::connect(my_addr, &others);
        let network = runtime
            .block_on(network)
            .map_err(|_| MpcError("Failed to setup network"))?;

        let ids = network
            .participants()
            .map(|id| (id + 1u32).into())
            .collect();
        let context = shamir::ShamirParams { threshold, ids };
        let engine = AdderEngine {
            network,
            runtime,
            context,
        };
        Ok(engine)
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

// We start allowing just one element.

#[derive(Debug)]
pub struct MpcError(pub &'static str);

impl std::fmt::Display for MpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = self.0;
        write!(f, "Bad thing happened: {msg}")
    }
}

impl std::error::Error for MpcError {}

pub fn do_preproc(files: &mut [File], number_of_shares: Vec<usize>) {
    assert_eq!(files.len(), number_of_shares.len());
    let known_to_each = vec![number_of_shares[0], number_of_shares[1]];
    let number_of_triplets = 0;
    let num = to_offset(0.0);
    preprocessing::write_preproc_to_file(
        files,
        known_to_each,
        number_of_triplets,
        curve25519_dalek::Scalar::from(num),
    )
    .unwrap();
}

#[cfg(test)]
mod test {
    use std::{io::Seek, time::Duration};

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
    fn sunshine_shamir() {
        use std::thread;
        let t1 = thread::spawn(move || {
            println!("[1] Setting up...");
            let mut engine = AdderEngine::shamir("127.0.0.1:1232", &["127.0.0.1:1233"]).unwrap();
            println!("[1] Ready");
            let res = engine.mpc_sum(&[32.0]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(move || {
            println!("[2] Setting up...");
            let mut engine = AdderEngine::shamir("127.0.0.1:1233", &["127.0.0.1:1232"]).unwrap();
            println!("[2] Ready");
            let res = engine.mpc_sum(&[32.0]).unwrap();
            println!("[2] Done");
            drop(engine);
            res
        });
        let a = t1.join().expect("joining")[0];
        let b = t2.join().expect("joining")[0];
        assert_eq!(a, 64.0);
        assert_eq!(b, 64.0);
    }

    #[test]
    fn sunshine_spdz() {
        use std::thread;
        let ctx1 = tempfile::tempfile().unwrap();
        let ctx2 = tempfile::tempfile().unwrap();
        let mut files = [ctx1, ctx2];
        do_preproc(&mut files, vec![1, 1]);
        let [mut ctx1, mut ctx2] = files;
        ctx1.rewind().unwrap();
        ctx2.rewind().unwrap();
        let t1 = thread::spawn(move || {
            println!("[1] Setting up...");
            let mut engine =
                AdderEngine::from_preprocess("127.0.0.1:1234", &["127.0.0.1:1235"], &mut ctx1)
                    .unwrap();
            println!("[1] Ready");
            let res = engine.mpc_sum(&[32.0]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(move || {
            println!("[2] Setting up...");
            let mut engine =
                AdderEngine::from_preprocess("127.0.0.1:1235", &["127.0.0.1:1234"], &mut ctx2)
                    .unwrap();
            println!("[2] Ready");
            let res = engine.mpc_sum(&[32.0]).unwrap();
            println!("[2] Done");
            drop(engine);
            res
        });
        let a = t1.join().expect("joining")[0];
        let b = t2.join().expect("joining")[0];
        assert_eq!(a, 64.0);
        assert_eq!(b, 64.0);
    }

    #[test]
    fn sunshine_spdz_for_two() {
        use std::thread;

        let ctx1 = tempfile::tempfile().unwrap();
        let ctx2 = tempfile::tempfile().unwrap();
        let mut files = [ctx1, ctx2];
        do_preproc(&mut files, vec![2, 2]);
        let [mut ctx1, mut ctx2] = files;
        ctx1.rewind().unwrap();
        ctx2.rewind().unwrap();
        let t1 = thread::spawn(move || {
            println!("[1] Setting up...");
            let mut engine =
                AdderEngine::from_preprocess("127.0.0.1:2234", &["127.0.0.1:2235"], &mut ctx1)
                    .unwrap();
            println!("[1] Ready");
            let res = engine.mpc_sum(&[32.0, 11.9]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(move || {
            println!("[2] Setting up...");
            let mut engine =
                AdderEngine::from_preprocess("127.0.0.1:2235", &["127.0.0.1:2234"], &mut ctx2)
                    .unwrap();
            println!("[2] Ready");
            let res = engine.mpc_sum(&[32.0, 24.1]).unwrap();
            println!("[2] Done");
            drop(engine);
            res
        });
        let a = t1.join().expect("joining");
        let a1 = a[0];
        let a2 = a[1];
        let b = t2.join().expect("joining");
        let b1 = b[0];
        let b2 = b[1];
        assert_eq!(a1, 64.0);
        assert_eq!(b1, 64.0);
        assert_eq!(a2, 36.0);
        assert_eq!(b2, 36.0);
        assert_eq!(a, [64.0, 36.0]);
    }
}
