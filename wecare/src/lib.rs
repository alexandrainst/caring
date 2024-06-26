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
use std::{fs::File, net::SocketAddr, time::Duration};
use tokio::runtime::Runtime;

type F = curve25519_dalek::Scalar;

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
    fn new(network: TcpNetwork, runtime: Runtime, context: S::Context) -> Self {
        Self {
            network,
            runtime,
            context,
        }
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

pub type SpdzEngine = AdderEngine<spdz::Share<curve25519_dalek::Scalar>>;

pub type ShamirEngine = AdderEngine<shamir::Share<curve25519_dalek::Scalar>>;

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

pub type Engine = generic::AdderEngine;

mod generic {
    use tokio::runtime::Runtime;

    use super::*;

    pub enum AdderEngine {
        Spdz(SpdzEngine),
        Shamir(ShamirEngine),
    }

    pub struct EngineBuilder<'a> {
        my_addr: &'a str,
        other_addr: Vec<&'a str>,
        threshold: Option<u64>,
        preprocessed: Option<&'a mut File>,
    }
    impl<'a> EngineBuilder<'a> {
        pub fn build_spdz(self) -> Result<AdderEngine, MpcError> {
            let (network, runtime) = self.semi_build()?;
            let file = self
                .preprocessed
                .ok_or(MpcError("No proccesing file found"))?;
            let mut context = preprocessing::read_preproc_from_file(file);
            context.params.who_am_i = network.index;
            let engine = SpdzEngine::new(network, runtime, context);
            Ok(AdderEngine::Spdz(engine))
        }

        pub fn build_shamir(self) -> Result<AdderEngine, MpcError> {
            let threshold = self.threshold.ok_or(MpcError("No threshold found"))?;
            let (network, runtime) = self.semi_build()?;
            let ids = network
                .participants()
                .map(|id| (id + 1u32).into())
                .collect();
            let context = shamir::ShamirParams { threshold, ids };
            let engine = ShamirEngine::new(network, runtime, context);
            Ok(AdderEngine::Shamir(engine))
        }

        fn semi_build(&self) -> Result<(TcpNetwork, Runtime), MpcError> {
            let my_addr: SocketAddr = self.my_addr.parse().unwrap();
            let others: Vec<SocketAddr> =
                self.other_addr.iter().map(|s| s.parse().unwrap()).collect();

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let network = TcpNetwork::connect(my_addr, &others);
            let network = runtime
                .block_on(network)
                .map_err(|_| MpcError("Failed to setup network"))?;

            Ok((network, runtime))
        }

        pub fn add_participant(mut self, addr: &'a str) -> Self {
            self.other_addr.push(addr);
            self
        }

        pub fn add_participants(mut self, addrs: &'a [impl AsRef<str>]) -> Self {
            let addrs = addrs.iter().map(|s| s.as_ref::<'a>());

            self.other_addr.extend(addrs);
            self
        }

        pub fn threshold(mut self, t: u64) -> Self {
            self.threshold = Some(t);
            self
        }

        pub fn file_to_preprocessed(mut self, file: &'a mut File) -> Self {
            self.preprocessed = Some(file);
            self
        }
    }

    impl AdderEngine {
        pub fn setup(addr: &str) -> EngineBuilder<'_> {
            EngineBuilder {
                my_addr: addr,
                other_addr: vec![],
                threshold: None,
                preprocessed: None,
            }
        }

        pub fn mpc_sum(&mut self, nums: &[f64]) -> Option<Vec<f64>> {
            match self {
                AdderEngine::Spdz(e) => e.mpc_sum(nums),
                AdderEngine::Shamir(e) => e.mpc_sum(nums),
            }
        }

        pub fn shutdown(self) {
            match self {
                AdderEngine::Spdz(e) => e.shutdown(),
                AdderEngine::Shamir(e) => e.shutdown(),
            }
        }
    }
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
            let mut engine = Engine::setup("127.0.0.1:1232")
                .add_participant("127.0.0.1:1233")
                .threshold(2)
                .build_shamir()
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
            let mut engine = Engine::setup("127.0.0.1:1233")
                .add_participant("127.0.0.1:1232")
                .threshold(2)
                .build_shamir()
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

            let mut engine = Engine::setup("127.0.0.1:1234")
                .add_participant("127.0.0.1:1235")
                .file_to_preprocessed(&mut ctx1)
                .build_spdz()
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
            let mut engine = Engine::setup("127.0.0.1:1235")
                .add_participant("127.0.0.1:1234")
                .file_to_preprocessed(&mut ctx2)
                .build_spdz()
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
            let mut engine = Engine::setup("127.0.0.1:2234")
                .add_participant("127.0.0.1:2235")
                .file_to_preprocessed(&mut ctx1)
                .build_spdz()
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
            let mut engine = Engine::setup("127.0.0.1:2235")
                .add_participant("127.0.0.1:2234")
                .file_to_preprocessed(&mut ctx2)
                .build_spdz()
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
