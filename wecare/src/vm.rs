use enum_dispatch::enum_dispatch;
use ff::Field;
use rand::rngs::StdRng;

use std::{
    any::Any,
    error::Error,
    ops::{Add, Sub},
    pin::Pin,
    sync::Arc,
};

use caring::{
    algebra::element::Element32,
    net::connection::{Connection, TcpConnection},
    schemes::{feldman, shamir, spdz},
    vm::{self, parsing::Exp},
};

type ShamirEngine<F> = vm::Engine<TcpConnection, shamir::Share<F>, StdRng>;
type SpdzEngine<F> = vm::Engine<TcpConnection, spdz::Share<F>, StdRng>;
type FeldmanEngine<F, G> = vm::Engine<TcpConnection, feldman::VerifiableShare<F, G>, StdRng>;

type ShamirCurve25519Engine = ShamirEngine<curve25519_dalek::Scalar>;
type ShamirElement32Engine = ShamirEngine<caring::algebra::element::Element32>;
type SpdzCurve25519Engine = SpdzEngine<curve25519_dalek::Scalar>;
type SpdzElement32Engine = SpdzEngine<caring::algebra::element::Element32>;

impl Expr {}

enum Expr {
    Curve25519(vm::parsing::Exp<curve25519_dalek::Scalar>),
    Element32(vm::parsing::Exp<Element32>),
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
    pub async fn execute(&mut self, expr: Expr) {
        match (self, expr) {
            (Engine::Spdz(engine), Expr::Curve25519(exp)) => {
                engine.execute(&exp.finalize());
            }
            (Engine::Shamir(engine), Expr::Element32(exp)) => {
                engine.execute(&exp.finalize());
            }
            _ => panic!("unsupported!"),
        }
    }
}
