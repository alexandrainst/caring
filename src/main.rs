#![feature(split_array)]

mod field;
mod ideal;
pub mod shamir;
mod float;

// TODO: Redo this. Probably just delete it.

use std::{error::Error, process::abort};

use ff::{Field, PrimeField};
use field::Element32;

use crate::ideal::cointoss;

#[derive(Copy, Clone)]
pub struct VerifiableShare<F: Field> {
    share: shamir::Share<F>,
    mac: F,
}

impl<F: Field> std::ops::Add for VerifiableShare<F> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share + rhs.share,
            mac: self.mac + rhs.mac,
        }
    }
}

impl<F: Field> std::ops::Sub for VerifiableShare<F> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share + rhs.share,
            mac: self.mac + rhs.mac,
        }
    }
}

impl<F: Field> std::ops::Mul<F> for VerifiableShare<F> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self::Output {
        Self {
            share: self.share * rhs,
            mac: self.mac * rhs,
        }
    }
}

impl<F: Field> std::ops::Mul for VerifiableShare<F> {
    type Output = Self;

    fn mul(self, _rhs: Self) -> Self::Output {
        let xy: shamir::Share<F> = todo!();
        let rxy: F = todo!();
        Self { share: xy, mac: rxy }
    }
}

pub fn share<F: Field>(num: Element32) -> Vec<VerifiableShare<F>> {
    todo!()
}

pub fn verify<F: Field>(input: &[VerifiableShare<F>], output: &[VerifiableShare<F>]) -> bool {
    let (alphas, betas): (&[F], &[F]) = cointoss();

    let (u0, w0): (F, F) = input
        .iter()
        .zip(alphas)
        .map(|(z, a)| (*a * z.mac, *a * z.share.y))
        .fold((F::ZERO, F::ZERO), |(mac, val), (u, w)| (u + mac, w + val));
    let (u1, w1): (F, F) = output
        .iter()
        .zip(betas)
        .map(|(z, b)| (*b * z.mac, *b * z.share.y))
        .fold((F::ZERO, F::ZERO), |(mac, val), (u, w)| (u + mac, w + val));

    let u = u0 + u1;
    let w = w0 + w1;

    let r: F = todo!("open 'r'");

    let t = u - r * w;
    todo!("check zero for 't'");
}

pub fn reconstruct<F: Field>(shares: &[VerifiableShare<F>], wire: usize) -> Element32 {
    todo!()
}

fn main() {
    println!("Hello, world!");
}
