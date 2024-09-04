use std::{
    fs::File,
    net::{SocketAddr, ToSocketAddrs},
};

use curve25519_dalek::{RistrettoPoint, Scalar};
use fixed::{FixedI128, FixedI32};
use rand::{rngs::StdRng, SeedableRng};

use caring::{
    algebra::{element::Element32, math::Vector},
    net::{agency::Broadcast, connection::TcpConnection, network::TcpNetwork},
    schemes::{feldman, shamir, spdz},
    vm::{self, parsing::Exp},
};

pub use caring::net::Id;
pub use caring::vm::Value;

#[derive(Clone, Copy)]
pub enum Number {
    Float(f64),
    Integer(u64),
    SignedInteger(i64),
}

impl From<f64> for Number {
    fn from(value: f64) -> Self {
        Number::Float(value)
    }
}

impl From<u64> for Number {
    fn from(value: u64) -> Self {
        Number::Integer(value)
    }
}

impl From<i64> for Number {
    fn from(value: i64) -> Self {
        Number::SignedInteger(value)
    }
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
                    .map_err(|_| format!("Could not fit {uint} into an u32"))?;
                Ok(Element32::from(uint))
            }
            Number::SignedInteger(int) => {
                let int: i32 = int
                    .try_into()
                    .map_err(|_| format!("Could not fit {int} into an u32"))?;
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

    pub fn to_f64(self) -> f64 {
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
pub type Opened = vm::parsing::Opened<Number>;

pub enum Engine {
    Spdz25519(SpdzCurve25519Engine),
    Spdz32(SpdzElement32Engine),
    Shamir25519(ShamirCurve25519Engine),
    Shamir32(ShamirElement32Engine),
    Feldman25519(FeldmanEngine<Scalar, RistrettoPoint>),
}

macro_rules! delegate {
    ($self:expr, $func:ident) => {
        match $self {
            Engine::Spdz25519(e) => e.$func().into(),
            Engine::Spdz32(e) => e.$func().into(),
            Engine::Shamir25519(e) => e.$func().into(),
            Engine::Shamir32(e) => e.$func().into(),
            Engine::Feldman25519(e) => e.$func().into(),
        }
    };
}

macro_rules! delegate_await {
    ($self:expr, $func:ident) => {
        match $self {
            Engine::Spdz25519(e) => e.$func().await.into(),
            Engine::Spdz32(e) => e.$func().await.into(),
            Engine::Shamir25519(e) => e.$func().await.into(),
            Engine::Shamir32(e) => e.$func().await.into(),
            Engine::Feldman25519(e) => e.$func().await.into(),
        }
    };
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    pub async fn execute(&mut self, expr: Opened) -> Value<UnknownNumber> {
        let res: Value<UnknownNumber> = match self {
            Engine::Spdz25519(engine) => {
                let res = engine.execute(&expr.try_finalize().unwrap()).await;
                res.map(|x| x.into())
            }
            Engine::Shamir32(engine) => engine
                .execute(&expr.try_finalize().unwrap())
                .await
                .map(|x| x.into()),
            Engine::Spdz32(engine) => {
                let res = engine.execute(&expr.try_finalize().unwrap()).await;
                res.map(|x| x.into())
            }
            Engine::Shamir25519(engine) => {
                let res = engine.execute(&expr.try_finalize().unwrap()).await;
                res.map(|x| x.into())
            }
            Engine::Feldman25519(engine) => {
                let res = engine.execute(&expr.try_finalize().unwrap()).await;
                res.map(|x| x.into())
            }
        };
        res
    }

    pub fn id(&self) -> Id {
        delegate!(self, id)
    }

    pub fn peers(&self) -> Vec<Id> {
        delegate!(self, peers)
    }

    pub async fn sum(&mut self, nums: &[f64]) -> Vec<f64> {
        let nums: Vector<_> = nums.iter().map(|v| Number::Float(*v)).collect();
        let program = {
            let explist = Expr::symmetric_share_vec(nums);
            explist.sum().open()
        };
        self.execute(program)
            .await
            .unwrap_vector()
            .into_iter()
            .map(|x| x.to_f64())
            .collect()
    }

    pub async fn shutdown(self) -> Result<(), std::io::Error> {
        delegate_await!(self, shutdown)
    }
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
pub struct EngineBuilder {
    own: Option<SocketAddr>,
    peers: Vec<SocketAddr>,
    network: Option<TcpNetwork>,
    threshold: Option<u64>,
    preprocesing: Option<File>,
    field: Option<FieldKind>,
    scheme: Option<SchemeKind>,
}

impl EngineBuilder {
    pub fn address(mut self, addr: impl ToSocketAddrs) -> Self {
        // TODO: Handle this better
        self.own
            .replace(addr.to_socket_addrs().unwrap().next().unwrap());
        self
    }

    pub fn participant(mut self, addr: impl ToSocketAddrs) -> Self {
        // TODO: Handle this better
        self.peers
            .push(addr.to_socket_addrs().unwrap().next().unwrap());
        self
    }

    pub fn participants(mut self, addrs: impl ToSocketAddrs) -> Self {
        let addrs = addrs.to_socket_addrs().unwrap();
        self.peers.extend(addrs);
        self
    }

    pub fn participants_from<A: ToSocketAddrs>(
        mut self,
        addrs: impl IntoIterator<Item = A>,
    ) -> Self {
        let addrs = addrs
            .into_iter()
            .map(|a| a.to_socket_addrs().unwrap().next().unwrap());
        self.peers.extend(addrs);
        self
    }

    pub fn threshold(mut self, t: u64) -> Self {
        self.threshold = Some(t);
        self
    }

    pub fn preprocessed(mut self, file: File) -> Self {
        self.preprocesing = Some(file);
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
            .map_err(|_| "Bad thing happened")?;

        self.network = Some(network);
        Ok(self)
    }

    pub fn build(self) -> Engine {
        let mut network = self.network.expect("No network installed!");
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
                let mut file = self.preprocesing.expect("Missing preproc!");
                let context = spdz::preprocessing::load_context(&mut file);
                network.set_id(context.params.who_am_i);
                Engine::Spdz25519(vm::Engine::new(context, network, rng))
            }
            (SchemeKind::Spdz, FieldKind::Element32) => {
                let mut file = self.preprocesing.expect("Missing preproc!");
                let context = spdz::preprocessing::load_context(&mut file);
                network.set_id(context.params.who_am_i);
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
    use caring::{net::Id, vm::Value};

    use crate::vm::UnknownNumber;

    pub struct Engine {
        parent: super::Engine,
        runtime: tokio::runtime::Runtime,
    }

    pub struct EngineBuilder {
        parent: super::EngineBuilder,
        runtime: tokio::runtime::Runtime,
    }

    impl super::EngineBuilder {
        pub fn single_threaded_runtime(self) -> EngineBuilder {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            EngineBuilder {
                parent: self,
                runtime,
            }
        }
        pub fn multi_threaded_runtime(self) -> EngineBuilder {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();
            EngineBuilder {
                parent: self,
                runtime,
            }
        }
    }

    impl EngineBuilder {
        pub fn connect_blocking(mut self) -> Result<Self, &'static str> {
            let runtime = &mut self.runtime;
            let mut parent = self.parent;
            parent = runtime.block_on(async move { parent.connect().await })?;
            self.parent = parent;
            Ok(self)
        }

        pub fn build(self) -> Engine {
            let parent = self.parent.build();
            Engine {
                runtime: self.runtime,
                parent,
            }
        }
    }

    impl Engine {
        pub fn execute(&mut self, expr: super::Opened) -> Value<UnknownNumber> {
            self.runtime.block_on(self.parent.execute(expr))
        }

        pub fn id(&self) -> Id {
            self.parent.id()
        }

        pub fn peers(&self) -> Vec<Id> {
            self.parent.peers()
        }

        pub fn sum(&mut self, nums: &[f64]) -> Vec<f64> {
            self.runtime.block_on(self.parent.sum(nums))
        }

        pub fn shutdown(self) -> Result<(), std::io::Error> {
            self.runtime.block_on(self.parent.shutdown())
        }
    }
}

#[cfg(test)]
mod test {
    use std::{thread, time::Duration};

    use crate::vm::{blocking, Engine, Expr, FieldKind, SchemeKind};

    #[test]
    fn addition() {
        fn engine(addr: &str, peers: [&str; 2]) -> blocking::Engine {
            Engine::builder()
                .address(addr)
                .participants_from(peers)
                .scheme(SchemeKind::Shamir)
                .field(FieldKind::Curve25519)
                .single_threaded_runtime()
                .connect_blocking()
                .unwrap()
                .build()
        }
        let addrs = ["127.0.0.1:3235", "127.0.0.1:3236", "127.0.0.1:3237"];
        let res = thread::scope(|scope| {
            [
                scope.spawn(|| {
                    println!("Party 0: Starting");
                    let mut engine = engine(addrs[0], [addrs[1], addrs[2]]);
                    let me = engine.id();

                    let num = 3.0;
                    let [a, b, c] = Expr::share_and_receive(num, me);
                    let sum = a + b + c;
                    let res = sum.open();
                    let script = res;

                    println!("Party 0: Executing");
                    let res = engine.execute(script);
                    res.unwrap_single().to_f64()
                }),
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(50));
                    println!("Party 1: Starting");
                    let mut engine = engine(addrs[1], [addrs[0], addrs[2]]);
                    let me = engine.id();

                    let num = 7.0;
                    let [a, b, c] = Expr::share_and_receive(num, me);
                    let sum = a + b + c;
                    let res = sum.open();
                    let script = res;

                    println!("Party 1: Executing");
                    let res = engine.execute(script);
                    res.unwrap_single().to_f64()
                }),
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(100));
                    println!("Party 2: Starting");
                    let mut engine = engine(addrs[2], [addrs[0], addrs[1]]);
                    let me = engine.id();

                    let num = 13.0;
                    let [a, b, c] = Expr::share_and_receive(num, me);
                    let sum = a + b + c;
                    let res = sum.open();
                    let script = res;

                    println!("Party 2: Executing");
                    let res = engine.execute(script);
                    res.unwrap_single().to_f64()
                }),
            ]
            .map(|t| t.join().unwrap())
        });

        assert_eq!(&res, &[23.0, 23.0, 23.0])
    }
}
