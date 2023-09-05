#![feature(split_array)]

mod field;
mod ideal;
pub mod shamir;

use std::{error::Error, process::abort};

use ff::{Field, PrimeField};
use field::Element;

use crate::ideal::cointoss;

#[derive(Copy, Clone)]
struct Share<F: Field> {
    val: F,
    mac: F,
}

impl<F: Field> std::ops::Add for Share<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            val: self.val + rhs.val,
            mac: self.mac + rhs.mac,
        }
    }
}

impl<F: Field> std::ops::Sub for Share<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            val: self.val + rhs.val,
            mac: self.mac + rhs.mac,
        }
    }
}

impl<F: Field> std::ops::Mul<F> for Share<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self::Output {
        Self {
            val: self.val * rhs,
            mac: self.mac * rhs,
        }
    }
}

impl<F: Field> std::ops::Mul for Share<F> {
    type Output = Self;

    fn mul(self, _rhs: Self) -> Self::Output {
        let xy: F = todo!();
        let rxy: F = todo!();
        Self { val: xy, mac: rxy }
    }
}

fn share<F: Field>(num: Element) -> Vec<Share<F>> {
    todo!()
}

fn verify<F: Field>(input: &[Share<F>], output: &[Share<F>]) -> bool {
    let (alphas, betas): (&[F], &[F]) = cointoss();

    let (u0, w0): (F, F) = input
        .iter()
        .zip(alphas)
        .map(|(z, a)| (*a * z.mac, *a * z.val))
        .fold((F::ZERO, F::ZERO), |(mac, val), (u, w)| (u + mac, w + val));
    let (u1, w1): (F, F) = output
        .iter()
        .zip(betas)
        .map(|(z, b)| (*b * z.mac, *b * z.val))
        .fold((F::ZERO, F::ZERO), |(mac, val), (u, w)| (u + mac, w + val));

    let u = u0 + u1;
    let w = w0 + w1;

    let r: F = todo!("open 'r'");

    let t = u - r * w;
    todo!("check zero for 't'");
}

fn reconstruct<F: Field>(shares: &[Share<F>], wire: usize) -> Element {
    todo!()
}

fn main() {
    println!("Hello, world!");
}
