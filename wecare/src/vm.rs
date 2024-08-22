use std::{fs::File, net::SocketAddr, path::Path};

use curve25519_dalek::{RistrettoPoint, Scalar};
use fixed::{FixedI128, FixedI32};
use rand::{rngs::StdRng, SeedableRng};

use caring::{
    algebra::{element::Element32, math::Vector},
    net::{agency::Broadcast, connection::TcpConnection, network::TcpNetwork},
    schemes::{
        feldman,
        interactive::{InteractiveShared, InteractiveSharedMany},
        shamir, spdz,
    },
    vm::{self, parsing::Exp},
};

use crate::Mapping;

pub enum Number {
    Float(f64),
    Integer(u64),
    SignedInteger(i64),
}

impl TryFrom<Number> for Element32 {
    type Error = String;
    fn try_from(value: Number) -> Result<Self, Self::Error> {
        match value {
            Number::Float(float) => {
                let num: i32 = FixedI32::<16>::from_num(float).to_bits();
                Ok(Element32::from(num as u32))
            }
            Number::Integer(uint) => {
                let uint: u32 = uint
                    .try_into()
                    .map_err(|_| format!("Could not parse fit {uint} into an u32"))?;
                Ok(Element32::from(uint))
            }
            Number::SignedInteger(int) => {
                let int: i32 = int
                    .try_into()
                    .map_err(|_| format!("Could not parse fit {int} into an u32"))?;
                let uint: u32 = if int.is_negative() {
                    u32::MAX / 2 - int.unsigned_abs()
                } else {
                    u32::MAX / 2 + int as u32
                };
                Ok(Element32::from(uint))
            }
        }
    }
}

impl TryFrom<Number> for Scalar {
    type Error = String;
    fn try_from(value: Number) -> Result<Self, Self::Error> {
        match value {
            Number::Float(float) => {
                let num: i128 = FixedI128::<64>::from_num(float).to_bits();
                Ok(Scalar::from(num as u128))
            }
            Number::Integer(uint) => {
                let uint: u128 = uint.into();
                Ok(Scalar::from(uint))
            }
            Number::SignedInteger(int) => {
                let int: i128 = int.into();
                let uint: u128 = if int.is_negative() {
                    u128::MAX / 2 - int.unsigned_abs()
                } else {
                    u128::MAX / 2 + int as u128
                };
                Ok(Scalar::from(uint))
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum UnknownNumber {
    U32(u32),
    U64(u64),
    U128(u128),
}

impl UnknownNumber {
    pub fn to_u64(self) -> u64 {
        match self {
            UnknownNumber::U32(a) => a.into(),
            UnknownNumber::U64(a) => a,
            UnknownNumber::U128(a) => a as u64,
        }
    }

    pub fn to_i64(self) -> i64 {
        todo!()
    }

    pub fn to_f64(self) -> u64 {
        match self {
            UnknownNumber::U32(val) => {
                let val = FixedI32::<16>::from_bits(val as i32);
                val.to_num()
            }
            UnknownNumber::U64(_) => todo!(),
            UnknownNumber::U128(val) => {
                let num = FixedI128::<64>::from_bits(val as i128);
                num.to_num()
            }
        }
    }
}

impl From<Element32> for UnknownNumber {
    fn from(value: Element32) -> Self {
        let value: u32 = value.into();
        Self::U32(value)
    }
}

impl From<Scalar> for UnknownNumber {
    fn from(value: Scalar) -> Self {
        let val = u128::from_le_bytes(value.as_bytes()[0..128 / 8].try_into().unwrap());
        Self::U128(val)
    }
}

type ShamirEngine<F> = vm::Engine<TcpConnection, shamir::Share<F>, StdRng>;
type SpdzEngine<F> = vm::Engine<TcpConnection, spdz::Share<F>, StdRng>;
type FeldmanEngine<F, G> = vm::Engine<TcpConnection, feldman::VerifiableShare<F, G>, StdRng>;

type ShamirCurve25519Engine = ShamirEngine<curve25519_dalek::Scalar>;
type ShamirElement32Engine = ShamirEngine<caring::algebra::element::Element32>;
type SpdzCurve25519Engine = SpdzEngine<curve25519_dalek::Scalar>;
type SpdzElement32Engine = SpdzEngine<caring::algebra::element::Element32>;

pub type Expr = Exp<Number>;

pub enum Engine {
    Spdz25519(SpdzCurve25519Engine),
    Spdz32(SpdzElement32Engine),
    Shamir25519(ShamirCurve25519Engine),
    Shamir32(ShamirElement32Engine),
    Feldman25519(FeldmanEngine<Scalar, RistrettoPoint>),
}

impl Engine {
    pub fn builder<'a>() -> EngineBuilder<'a> {
        EngineBuilder::default()
    }

    pub async fn execute(&mut self, expr: Expr) -> UnknownNumber {
        let res: UnknownNumber = match self {
            Engine::Spdz25519(engine) => {
                let res = engine.execute(&expr.open().try_finalize().unwrap()).await;
                res.into()
            }
            Engine::Shamir32(engine) => engine
                .execute(&expr.open().try_finalize().unwrap())
                .await
                .into(),
            Engine::Spdz32(engine) => {
                let res = engine.execute(&expr.open().try_finalize().unwrap()).await;
                res.into()
            }
            Engine::Shamir25519(engine) => {
                let res = engine.execute(&expr.open().try_finalize().unwrap()).await;
                res.into()
            }
            Engine::Feldman25519(engine) => {
                let res = engine.execute(&expr.open().try_finalize().unwrap()).await;
                res.into()
            }
        };
        res
    }

    // pub async fn raw<S, Func, O>(&mut self, routine: Func) -> O
    //     where S: InteractiveShared,
    //           Func: async Fn(&mut TcpNetwork, &mut S::Context, &mut StdRng) -> O
    // {
    //     match self {
    //         Engine::Spdz25519(e) => { e.raw(routine).await },
    //         Engine::Spdz32(e) => e.raw(routine).await,
    //         Engine::Shamir25519(_) => todo!(),
    //         Engine::Shamir32(_) => todo!(),
    //         Engine::Feldman25519(_) => todo!(),
    //     }
    // }

    // compatilbity.
    // pub async fn mpc_sum<F: Mapping, S: InteractiveSharedMany<Value = F>>(&mut self, nums: &[f64]) -> Option<Vec<f64>>
    // where
    //     <S as InteractiveSharedMany>::VectorShare: std::iter::Sum,
    // {
    //     self.raw(async |net, ctx, rng| {
    //         let nums: Vec<_> = nums.iter().map(|&num| F::from_f64(num)).collect();
    //         let shares: Vec<S::VectorShare> =
    //             S::symmetric_share_many(ctx, &nums, rng, net) .await .unwrap();
    //         let sum: S::VectorShare = shares.into_iter().sum();
    //         let res: Vector<F> = S::recombine_many(ctx, sum, net).await.unwrap();
    //
    //         let res = res.into_iter().map(|x| F::into_f64(x)).collect();
    //         Some(res)
    //     }).await
    // }
}

pub enum FieldKind {
    Curve25519,
    Element32,
}

pub enum SchemeKind {
    Shamir,
    Spdz,
    Feldman,
}

#[derive(Default)]
pub struct EngineBuilder<'a> {
    own: Option<SocketAddr>,
    peers: Vec<SocketAddr>,
    network: Option<TcpNetwork>,
    threshold: Option<u64>,
    preprocesing: Option<&'a Path>,
    field: Option<FieldKind>,
    scheme: Option<SchemeKind>,
}

impl<'a> EngineBuilder<'a> {
    pub fn address(mut self, addr: impl Into<SocketAddr>) -> Self {
        self.own.replace(addr.into());
        self
    }

    pub fn participant(mut self, addr: impl Into<SocketAddr>) -> Self {
        self.peers.push(addr.into());
        self
    }

    pub fn participants<I, T>(mut self, addrs: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<SocketAddr>,
    {
        let addrs = addrs.into_iter().map(|s| s.into());
        self.peers.extend(addrs);
        self
    }

    pub fn threshold(mut self, t: u64) -> Self {
        self.threshold = Some(t);
        self
    }

    pub fn preprocessed(mut self, path: &'a Path) -> Self {
        self.preprocesing = Some(path);
        self
    }

    pub fn scheme(mut self, scheme: SchemeKind) -> Self {
        self.scheme = Some(scheme);
        self
    }

    pub fn field(mut self, field: FieldKind) -> Self {
        self.field = Some(field);
        self
    }

    pub async fn connect(mut self) -> Result<Self, &'static str> {
        let network = TcpNetwork::connect(self.own.unwrap(), &self.peers)
            .await
            .map_err(|e| "Bad thing happened")?;

        self.network = Some(network);
        Ok(self)
    }

    pub fn build(self) -> Engine {
        let network = self.network.expect("No network installed!");
        let party_count = network.size();
        let scheme = self.scheme.unwrap_or(SchemeKind::Shamir);
        let threshold = self.threshold.unwrap_or(party_count as u64);
        let field = self.field.unwrap_or(FieldKind::Curve25519);
        let rng = rand::rngs::StdRng::from_entropy();

        match (scheme, field) {
            (SchemeKind::Shamir, FieldKind::Curve25519) => {
                let ids = network
                    .participants()
                    .map(|id| (id + 1u32).into())
                    .collect();
                let context = shamir::ShamirParams { threshold, ids };
                Engine::Shamir25519(vm::Engine::new(context, network, rng))
            }
            (SchemeKind::Shamir, FieldKind::Element32) => {
                let ids = network
                    .participants()
                    .map(|id| (id + 1u32).into())
                    .collect();
                let context = shamir::ShamirParams { threshold, ids };
                Engine::Shamir32(vm::Engine::new(context, network, rng))
            }
            (SchemeKind::Spdz, FieldKind::Curve25519) => {
                let path = self.preprocesing.expect("Missing preproc!");
                let mut file = File::open(path).unwrap();
                let context = spdz::preprocessing::load_context(&mut file);
                Engine::Spdz25519(vm::Engine::new(context, network, rng))
            }
            (SchemeKind::Spdz, FieldKind::Element32) => {
                let path = self.preprocesing.expect("Missing preproc!");
                let mut file = File::open(path).unwrap();
                let context = spdz::preprocessing::load_context(&mut file);
                Engine::Spdz32(vm::Engine::new(context, network, rng))
            }
            (SchemeKind::Feldman, FieldKind::Curve25519) => {
                let ids = network
                    .participants()
                    .map(|id| (id + 1u32).into())
                    .collect();
                let context = shamir::ShamirParams { threshold, ids };
                Engine::Feldman25519(vm::Engine::new(context, network, rng))
            }
            (SchemeKind::Feldman, FieldKind::Element32) => {
                panic!("Can't construct feldman from this field element. Missing group!")
            }
        }
    }
}

pub mod blocking {
    use crate::vm::{Expr, UnknownNumber};

    pub struct Engine {
        parent: super::Engine,
        runtime: tokio::runtime::Runtime,
    }

    pub struct EngineBuilder<'a> {
        parent: super::EngineBuilder<'a>,
        runtime: tokio::runtime::Runtime,
    }

    impl<'a> super::EngineBuilder<'a> {
        pub fn single_threaded_runtime(self) -> EngineBuilder<'a> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            EngineBuilder {
                parent: self,
                runtime,
            }
        }
        pub fn multi_threaded_runtime(self) -> EngineBuilder<'a> {
            let runtime = tokio::runtime::Builder::new_multi_thread().build().unwrap();
            EngineBuilder {
                parent: self,
                runtime,
            }
        }
    }

    impl<'a> EngineBuilder<'a> {
        pub fn connect_blocking(mut self) -> Result<Self, &'static str> {
            let runtime = &mut self.runtime;
            let mut parent = self.parent;
            parent = runtime.block_on(async move { parent.connect().await })?;
            self.parent = parent;
            Ok(self)
        }
    }

    impl Engine {
        pub fn execute(&mut self, expr: Expr) -> UnknownNumber {
            self.runtime
                .block_on(async { self.parent.execute(expr).await })
        }
    }
}
