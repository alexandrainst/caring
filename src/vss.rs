use std::{ops, iter};

use ff::Field;
use rand::RngCore;
use crate::shamir::{Share, self};

#[derive(Clone, Copy)]
pub struct VerifiableShare<F : Field, G : Field>{
    share: Share<F>,
    // Should probably be a different field than F,
    // but supports a commit operation commit(F) -> G
    // or a 'product' G = F * G with F.
    // As such we can support any pair of fields that provide these
    // options. Note that the discrete log problem for the second
    // should be hard.
    mac: Mac<G>, 
}

#[derive(Clone, Copy)]
pub struct Mac<F: Field>(F);


impl<F: Field, G: Field> ops::Add for VerifiableShare<F,G> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share + rhs.share,
            mac: Mac(self.mac.0 + rhs.mac.0),
        }
    }
}


impl<F: Field, G: Field> ops::Sub for VerifiableShare<F,G> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share - rhs.share,
            mac: Mac(self.mac.0 - rhs.mac.0),
        }
    }
}

pub struct Polynomial<F: Field>(Box<[F]>);

pub fn share<F: Field, G: Field>(
    val: F,
    ids: &[F],
    threshold: u64,
    rng: &mut impl RngCore,
) -> (Vec<VerifiableShare<F,G>>, Polynomial<G>)
    where G: std::ops::Mul<F, Output = G>
{
    // let shares = shamir::share(val, threshold, rng);
    // 1. We need to get the polynomial.
    // 2. We then need to do commitments to it.
    // 3. We need to provide commitments/macs to the corresponding shares.
    // Then pack these macs with the shares and output them.

    // there are some code-duplication with `shamri.rs` currently.
    // that will probably be fixed.

    // Sample random t-degree polynomial
    let n = ids.len();
    let poly = (1..threshold).map(|_| F::random(&mut *rng));
    // I want to avoid this allocation :(
    let poly : Box<[F]> = iter::once(val).chain(poly).collect();

    let g : G = G::ONE;
    let mac_poly : Box<[G]> = poly.iter().map(|a| g * *a).collect();

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<_> = Vec::with_capacity(n);

    for x in ids {
        let x = *x;
        let share = poly.iter()
            .enumerate()
            .map(|(i, a)| -> F {
                // evaluate: a * x^i
                *a * x.pow([i as u64])
            }) // sum: s + a1 x + a2 x^2 + ...
            .fold(F::ZERO, |sum, x| sum + x);
        let mac = mac_poly.iter()
            .enumerate()
            .map(|(i, a)| {
                // evaluate: a * x^i
                *a * x.pow([i as u64])
            }) // sum: s + a1 x + a2 x^2 + ...
            .fold(G::ZERO, |sum, x| sum + x);
        let share = Share::<F> { x, y: share };
        let mac = Mac(mac);
        shares.push(VerifiableShare{share, mac});
    }

    let mac_poly = Polynomial(mac_poly);
    (shares, mac_poly)
}
