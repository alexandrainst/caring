use curve25519_dalek::{constants, Scalar};
use ff::PrimeField;
use fixed::{FixedI128, FixedI32};
use rand::rngs::StdRng;

use std::{
    error::Error,
    ops::{Add, Sub},
};

use caring::{
    algebra::element::Element32,
    net::connection::{Connection, TcpConnection},
    schemes::{feldman, shamir, spdz},
    vm::{self, parsing::Exp, Instruction, Value},
};

pub enum Expr {
    Curve25519(vm::parsing::Exp<curve25519_dalek::Scalar>),
    Element32(vm::parsing::Exp<Element32>),
}

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

type ShamirEngine<F> = vm::Engine<TcpConnection, shamir::Share<F>, StdRng>;
type SpdzEngine<F> = vm::Engine<TcpConnection, spdz::Share<F>, StdRng>;
type FeldmanEngine<F, G> = vm::Engine<TcpConnection, feldman::VerifiableShare<F, G>, StdRng>;

type ShamirCurve25519Engine = ShamirEngine<curve25519_dalek::Scalar>;
type ShamirElement32Engine = ShamirEngine<caring::algebra::element::Element32>;
type SpdzCurve25519Engine = SpdzEngine<curve25519_dalek::Scalar>;
type SpdzElement32Engine = SpdzEngine<caring::algebra::element::Element32>;

pub struct Expression {
    constants: Vec<Number>,
    instructions: Vec<Instruction>,
}

impl Add for Expr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expr::Curve25519(a), Expr::Curve25519(b)) => Expr::Curve25519(a + b),
            (Expr::Element32(a), Expr::Element32(b)) => Expr::Element32(a + b),
            _ => {
                panic!("Incompatible")
            }
        }
    }
}

impl Sub for Expr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expr::Curve25519(a), Expr::Curve25519(b)) => Self::Curve25519(a - b),
            (Expr::Element32(a), Expr::Element32(b)) => Self::Element32(a - b),
            _ => {
                panic!("Incompatible")
            }
        }
    }
}

pub enum Engine {
    Spdz(SpdzCurve25519Engine),
    Shamir(ShamirElement32Engine),
}

impl Engine {
    pub async fn execute(&mut self, expr: Expression) {
        match self {
            Engine::Spdz(engine) => {
                let constants = expr
                    .constants
                    .into_iter()
                    .map(|x| Value::Single(x.try_into().unwrap()))
                    .collect();
                let exp = Exp::from_parts(constants, expr.instructions);
                engine.execute(&exp.open().finalize()).await;
            }
            Engine::Shamir(engine) => {
                let constants = expr
                    .constants
                    .into_iter()
                    .map(|x| Value::Single(x.try_into().unwrap()))
                    .collect();
                let exp = Exp::from_parts(constants, expr.instructions);
                engine.execute(&exp.open().finalize()).await;
            }
            _ => panic!("unsupported!"),
        }
    }
}
