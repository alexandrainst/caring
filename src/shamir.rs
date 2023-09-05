use core::fmt;
use std::{fmt::Display, ops};

use ff::{derive::rand_core::RngCore, Field};
use num_traits::Num;

// pub trait Field: num_traits::Num + num_traits::NumAssign + fmt::Debug + ops::Neg<Output = Self> + Copy + num_traits::Pow<i32, Output = Self> {
//     fn random(rng: &mut impl RngCore) -> Self;
//     fn invert(self) -> Option<Self>;
// }

// impl Field for f64 {
//     fn random(rng: &mut impl RngCore) -> Self {
//         let num = rng.next_u32();
//         num as f64
//     }

//     fn invert(self) -> Option<Self> {
//         Some(1.0 / self)
//     }
// }

#[derive(Clone, Copy, Debug)]
pub struct Share<F: Field> {
    x: F,
    y: F,
}

impl<F: Field> std::ops::Add for Share<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.x == rhs.x);
        Self {
            x: self.x,
            y: self.y + rhs.y,
        }
    }
}

pub fn share<F: Field>(v: F, ids: &[F], threshold: u64, rng: &mut impl RngCore) -> Vec<Share<F>> {
    let n = ids.len();
    assert!(
        n >= threshold as usize,
        "Threshold should be less-than-equal to the number of shares"
    );

    // Sample random t-degree polynomial
    let mut polynomial: Vec<F> = Vec::with_capacity(threshold as usize);
    polynomial.push(v);
    for _ in 1..threshold {
        let a = F::random(&mut *rng);
        polynomial.push(a);
    }

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<Share<F>> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let share = polynomial
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let exp: [u64; 1] = [i as u64];
                (*a) * x.pow(exp)
            })
            .fold(F::ZERO, |sum, x| sum + x);
        shares.push(Share::<F> { x, y: share });
    }

    shares
}

pub fn reconstruct<F: Field>(shares: &[Share<F>]) -> F {
    // Lagrange interpolation
    let mut sum = F::ZERO;
    for share in shares.iter() {
        let xi = share.x;
        let yi = share.y;

        let mut prod = F::ONE;
        for Share { x: xk, y: _ } in shares.iter() {
            let xk = *xk;
            if xk == xi {
                continue;
            }
            prod *= -xk * (xi - xk).invert().unwrap();
        }
        sum += yi * prod;
        dbg!(sum);
    }

    sum
}

#[cfg(test)]
mod test {

    use std::collections::BTreeMap;

    use crate::field::Element;
    use ff::PrimeField;

    use super::*;

    #[test]
    fn sunshine() {
        let mut rng = rand::thread_rng();
        let v = Element::from(42);
        dbg!(v);
        let ids: Vec<_> = (1..=5).map(Element::from).collect();
        let shares = share(v, &ids, 6, &mut rng);
        dbg!(&shares);
        // assert!(shares.len() == 5);
        let v = reconstruct(&shares);
        dbg!(&v);
        assert_eq!(v, Element::from(42));
    }
}
