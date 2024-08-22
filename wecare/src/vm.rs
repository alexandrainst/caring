use curve25519_dalek::Scalar;
use fixed::{FixedI128, FixedI32};
use rand::rngs::StdRng;


use caring::{
    algebra::element::Element32,
    net::connection::TcpConnection,
    schemes::{feldman, shamir, spdz},
    vm::{self, parsing::Exp},
};

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
pub enum UnknownNumber{
    U32(u32),
    U64(u64),
    U128(u128),
}


impl UnknownNumber {

    pub fn to_u64(self) -> u64 {
        match self {
            UnknownNumber::U32(a) => a.into(),
            UnknownNumber::U64(a) => a,
            UnknownNumber::U128(a) => a as u64
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
            },
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
    Spdz(SpdzCurve25519Engine),
    Shamir(ShamirElement32Engine),
}

impl Engine {
    pub async fn execute(&mut self, expr: Expr) -> UnknownNumber {
        let res: UnknownNumber = match self {
            Engine::Spdz(engine) => {
                let res = engine.execute(&expr.open().try_finalize().unwrap()).await;
                res.into()
            }
            Engine::Shamir(engine) => {
                engine.execute(&expr.open().try_finalize().unwrap()).await.into()
            }
            _ => panic!("unsupported!"),
        };

        res
    }
}
