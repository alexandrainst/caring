pub mod vm;

use caring::{
    algebra::math::Vector,
    net::network::TcpNetwork,
    schemes::{
        feldman,
        interactive::InteractiveSharedMany,
        shamir,
        spdz::{self, preprocessing},
    },
};
use curve25519_dalek::RistrettoPoint;
use fixed::FixedI32;
use rand::{thread_rng, SeedableRng};
use std::{error::Error, fs::File, net::SocketAddr, time::Duration};
use tokio::runtime::Runtime;

pub trait Mapping {
    fn from_f64(val: f64) -> Self;
    fn into_f64(val: Self) -> f64;
}

pub type S25519 = curve25519_dalek::Scalar;
pub type S32 = caring::algebra::element::Element32;

type SignedFix = fixed::FixedI128<64>;
impl Mapping for S25519 {
    fn from_f64(val: f64) -> Self {
        Self::from({
            // convert to signed fixed point
            let num: i128 = SignedFix::from_num(val).to_bits();
            // Okay this a some voodoo why this 'just works', but the idea is that
            // twos complement is actually just an offset from the max value.
            // https://en.wikipedia.org/wiki/Offset_binary
            num as u128
        })
    }

    fn into_f64(val: Self) -> f64 {
        let val = u128::from_le_bytes(val.as_bytes()[0..128 / 8].try_into().unwrap());
        {
            // Same applies as above
            let num = SignedFix::from_bits(val as i128);
            num.to_num()
        }
    }
}

impl Mapping for S32 {
    fn from_f64(val: f64) -> Self {
        let num: i32 = FixedI32::<16>::from_num(val).to_bits();
        S32::from(num as u32)
    }

    fn into_f64(val: Self) -> f64 {
        let val: u32 = val.into();
        let val = FixedI32::<16>::from_bits(val as i32);
        val.to_num()
    }
}

pub struct AdderEngine<S: InteractiveSharedMany> {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime,
    context: S::Context,
}

impl<S, F> AdderEngine<S>
where
    S: InteractiveSharedMany<Value = F>,
    F: Mapping,
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

        let nums: Vec<_> = nums.iter().map(|&num| F::from_f64(num)).collect();
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

            let res = res.into_iter().map(|x| F::into_f64(x)).collect();
            Some(res)
        });
        res
    }
}

pub type SpdzEngine = AdderEngine<spdz::Share<S25519>>;
pub type ShamirEngine = AdderEngine<shamir::Share<S25519>>;
pub type FeldmanEngine = AdderEngine<feldman::VerifiableShare<S25519, RistrettoPoint>>;
pub type SpdzEngine32 = AdderEngine<spdz::Share<S32>>;
pub type ShamirEngine32 = AdderEngine<shamir::Share<S32>>;

#[derive(Debug)]
pub struct MpcError(pub String);

impl std::fmt::Display for MpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = &self.0;
        write!(f, "Bad thing happened: {msg}")
    }
}

impl std::error::Error for MpcError {}

pub fn do_preproc(
    files: &mut [File],
    num_preshares: &[usize],
    num_triplets: usize,
    use_32: bool,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(files.len(), num_preshares.len());
    if use_32 {
        let num = S32::from_f64(0.0);
        preprocessing::write_context(files, num_preshares, num_triplets, num)
    } else {
        let num = S25519::from_f64(0.0);
        preprocessing::write_context(files, num_preshares, num_triplets, num)
    }
}

pub type Engine = generic::AdderEngine;

mod generic {

    use super::*;

    pub enum AdderEngine {
        Spdz(SpdzEngine),
        Spdz32(SpdzEngine32),
        Shamir(ShamirEngine),
        Shamir32(ShamirEngine32),
        Feldman(FeldmanEngine),
    }

    pub struct EngineBuilder<'a> {
        my_addr: &'a str,
        other_addr: Vec<&'a str>,
        threshold: Option<u64>,
        preprocessed: Option<&'a mut File>,
        use_32bit_field: bool,
    }

    impl<'a> EngineBuilder<'a> {
        pub fn build_spdz(self) -> Result<AdderEngine, MpcError> {
            let (mut network, runtime) = self.connect_network()?;
            let file = self
                .preprocessed
                .ok_or(MpcError("No proccesing file found".to_string()))?;
            if self.use_32bit_field {
                let context =
                    preprocessing::load_context(file).map_err(|e| MpcError(e.to_string()))?;
                network.set_id(context.params.who_am_i);
                let engine = SpdzEngine32::new(network, runtime, context);
                Ok(AdderEngine::Spdz32(engine))
            } else {
                let context =
                    preprocessing::load_context(file).map_err(|e| MpcError(e.to_string()))?;
                network.set_id(context.params.who_am_i);
                let engine = SpdzEngine::new(network, runtime, context);
                Ok(AdderEngine::Spdz(engine))
            }
        }

        pub fn build_shamir(self) -> Result<AdderEngine, MpcError> {
            let threshold = self
                .threshold
                .ok_or_else(|| MpcError("No threshold found".to_owned()))?;
            let (network, runtime) = self.connect_network()?;
            if self.use_32bit_field {
                let ids = network
                    .participants()
                    .map(|id| (id + 1u32).into())
                    .collect();
                let context = shamir::ShamirParams { threshold, ids };
                let engine = ShamirEngine32::new(network, runtime, context);
                Ok(AdderEngine::Shamir32(engine))
            } else {
                let ids = network
                    .participants()
                    .map(|id| (id + 1u32).into())
                    .collect();
                let context = shamir::ShamirParams { threshold, ids };
                let engine = ShamirEngine::new(network, runtime, context);
                Ok(AdderEngine::Shamir(engine))
            }
        }

        pub fn build_feldman(self) -> Result<AdderEngine, MpcError> {
            let threshold = self
                .threshold
                .ok_or_else(|| MpcError("No threshold found".to_string()))?;
            let (network, runtime) = self.connect_network()?;
            let ids = network
                .participants()
                .map(|id| (id + 1u32).into())
                .collect();
            let context = shamir::ShamirParams { threshold, ids };
            let engine = FeldmanEngine::new(network, runtime, context);
            Ok(AdderEngine::Feldman(engine))
        }

        fn connect_network(&self) -> Result<(TcpNetwork, Runtime), MpcError> {
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
                .map_err(|e| MpcError(format!("Failed to setup network: {e}")))?;

            Ok((network, runtime))
        }

        pub fn add_participant(mut self, addr: &'a str) -> Self {
            self.other_addr.push(addr);
            self
        }

        pub fn add_participants(mut self, addrs: &'a [impl AsRef<str>]) -> Self {
            let addrs = addrs.iter().map(|s| s.as_ref());

            self.other_addr.extend(addrs);
            self
        }

        pub fn threshold(mut self, t: u64) -> Self {
            self.threshold = Some(t);
            self
        }

        pub fn use_32bit_field(mut self) -> Self {
            self.use_32bit_field = true;
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
                use_32bit_field: false,
            }
        }

        pub fn mpc_sum(&mut self, nums: &[f64]) -> Option<Vec<f64>> {
            match self {
                AdderEngine::Spdz(e) => e.mpc_sum(nums),
                AdderEngine::Spdz32(e) => e.mpc_sum(nums),
                AdderEngine::Shamir(e) => e.mpc_sum(nums),
                AdderEngine::Shamir32(e) => e.mpc_sum(nums),
                AdderEngine::Feldman(e) => e.mpc_sum(nums),
            }
        }

        pub fn shutdown(self) {
            match self {
                AdderEngine::Spdz(e) => e.shutdown(),
                AdderEngine::Shamir(e) => e.shutdown(),
                AdderEngine::Spdz32(e) => e.shutdown(),
                AdderEngine::Shamir32(e) => e.shutdown(),
                AdderEngine::Feldman(e) => e.shutdown(),
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
        let a = Scalar::from_f64(1.1);
        let b = Scalar::from_f64(2.2);
        let c = Scalar::from_f64(3.3);
        let sum = a + b + c;
        let sum = Scalar::into_f64(sum);
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
        do_preproc(&mut files, &[1, 1], 0, false).unwrap();
        let [mut ctx1, mut ctx2] = files;
        ctx1.rewind().unwrap();
        ctx2.rewind().unwrap();
        let t1 = thread::spawn(move || {
            let span = tracing::info_span!("Player-0");
            let _enter = span.enter();
            tracing::info!("Setting up...");
            let mut engine = Engine::setup("127.0.0.1:1234")
                .add_participant("127.0.0.1:1235")
                .file_to_preprocessed(&mut ctx1)
                .build_spdz()
                .unwrap();
            tracing::info!("Ready");
            let res = engine.mpc_sum(&[32.0]).unwrap();
            tracing::info!("Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(move || {
            let span = tracing::info_span!("Player-1");
            let _enter = span.enter();
            tracing::info!("Setting up...");
            let mut engine = Engine::setup("127.0.0.1:1235")
                .add_participant("127.0.0.1:1234")
                .file_to_preprocessed(&mut ctx2)
                .build_spdz()
                .unwrap();
            tracing::info!("Ready");
            let res = engine.mpc_sum(&[32.0]).unwrap();
            tracing::info!("Done");
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
        do_preproc(&mut files, &[2, 2], 0, false).unwrap();
        let [mut ctx1, mut ctx2] = files;
        ctx1.rewind().unwrap();
        ctx2.rewind().unwrap();
        let t1 = thread::spawn(move || {
            let span = tracing::info_span!("Player-0");
            let _enter = span.enter();
            tracing::info!("Setting up...");
            let mut engine = Engine::setup("127.0.0.1:2234")
                .add_participant("127.0.0.1:2235")
                .file_to_preprocessed(&mut ctx1)
                .build_spdz()
                .unwrap();
            tracing::info!("Ready");
            let res = engine.mpc_sum(&[32.0, 11.9]).unwrap();
            tracing::info!("Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(200));
        let t2 = thread::spawn(move || {
            let span = tracing::info_span!("Player-1");
            let _enter = span.enter();
            tracing::info!("Setting up...");
            let mut engine = Engine::setup("127.0.0.1:2235")
                .add_participant("127.0.0.1:2234")
                .file_to_preprocessed(&mut ctx2)
                .build_spdz()
                .unwrap();
            tracing::info!("Ready");
            let res = engine.mpc_sum(&[32.0, 24.1]).unwrap();
            tracing::info!("Done");
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
