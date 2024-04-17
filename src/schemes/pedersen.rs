//! Implementation of Pedersen Secret Sharing
//! <https://www.cs.cornell.edu/courses/cs754/2001fa/129.PDF>
use std::ops::{Add, Mul, MulAssign, Sub};

use ff::Field;
use group::Group;
use rand::RngCore;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    algebra::{
        math::{lagrange_interpolation, Vector},
        poly::Polynomial,
    },
    schemes::Shared,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct VerifiableShare<F: Field, G: Group> {
    secret: F,
    blindness: F,
    // PERF: Wrap in a Arc or something like it?
    commitment: Polynomial<G>,
}

/// Generators for Pedersen Secret Sharing (and Pedersen Commitments)
///
/// Consists of two group elements (g,h), usally elliptic curve points.
///
/// These are cosntant public values, and should have a default implementation for ease of use.
/// NOTE: We can't really provide general default implementation for any Group,
/// but we should possibly do it for some groups e.g., curve25519.
#[derive(Clone)]
pub struct PedersenGenParams<G: Group>(G, G);

pub fn share<F, G>(
    secret: F,
    ids: &[F],
    threshold: usize,
    rng: &mut impl RngCore,
    PedersenGenParams(g, h): &PedersenGenParams<G>,
) -> Vec<VerifiableShare<F, G>>
where
    G: std::ops::Mul<F, Output = G>,
    F: Field,
    G: Group,
{
    let mut p1: Polynomial<F> = Polynomial::random(threshold, rng);
    let p2: Polynomial<F> = Polynomial::random(threshold, rng);
    // secret `s`
    p1.0[0] = secret;
    // random secret `t` (left random)

    let commitments: Polynomial<G> =
        p1.0.iter()
            .zip(p2.0.iter())
            .map(|(&a, &b)| *g * a + *h * b)
            .collect();

    let shares: Vec<_> = ids
        .iter()
        .map(|i| {
            // Sample the shares
            VerifiableShare {
                secret: p1.eval(i),
                blindness: p2.eval(i),
                commitment: commitments.clone(),
            }
        })
        .collect();

    shares
}

pub fn verify<F, G>(
    id: &F,
    share: &VerifiableShare<F, G>,
    PedersenGenParams(g, h): &PedersenGenParams<G>,
) -> bool
where
    F: Field,
    G: Group + std::ops::Mul<F, Output = G>,
{
    let VerifiableShare {
        secret,
        blindness,
        commitment,
    } = share;
    // C0^(i^0) * C1^(i^1) * C1^(i^2) + ...
    let mut check = G::identity();
    for (i, &a) in commitment.0.iter().enumerate() {
        check += a * id.pow([i as u64]);
    }

    // g^s * h^t =?= ...
    *g * *secret + *h * *blindness == check
}

#[derive(Clone)]
pub struct PedersenContext<F: Field, G: Group> {
    ids: Vec<F>,
    threshold: usize,
    pedersen_params: PedersenGenParams<G>,
}

impl<F, G> Shared for VerifiableShare<F, G>
where
    F: Field + Serialize + DeserializeOwned,
    G: Group + Serialize + DeserializeOwned + std::ops::Mul<F, Output = G>,
{
    type Context = PedersenContext<F, G>;

    type Value = F;

    fn share(ctx: &Self::Context, secret: Self::Value, rng: &mut impl RngCore) -> Vec<Self> {
        share(secret, &ctx.ids, ctx.threshold, rng, &ctx.pedersen_params)
    }

    fn recombine(ctx: &Self::Context, shares: &[Self]) -> Option<Self::Value> {
        let cheating: bool = shares
            .iter()
            .zip(ctx.ids.iter())
            .map(|(s, id)| verify(id, s, &ctx.pedersen_params))
            .any(|c| !c);
        if cheating {
            return None;
        }
        Some(reconstruct(shares, &ctx.ids))
    }
}

/// Reconstruct shares
///
/// Warning: Does not check them!
pub fn reconstruct<F: Field, G: Group>(shares: &[VerifiableShare<F, G>], ids: &[F]) -> F {
    // TODO: Maybe verify that the shares are all correct.

    // HACK: Parse as shamir, since we just use lagrange interpolation.
    let secrets: Vec<_> = shares.iter().map(|s| s.secret).collect();

    lagrange_interpolation(F::ZERO, ids, &secrets)
}

impl<F: Field, G: Group> Add<&Self> for VerifiableShare<F, G> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.secret += &rhs.secret;
        self.blindness += &rhs.blindness;
        self.commitment.0 += &rhs.commitment.0;
        self
    }
}

impl<F: Field, G: Group> Add for VerifiableShare<F, G> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<F: Field, G: Group> Sub<&Self> for VerifiableShare<F, G> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.secret -= &rhs.secret;
        self.blindness -= &rhs.blindness;
        self.commitment.0 -= &rhs.commitment.0;
        self
    }
}

impl<F: Field, G: Group> Sub for VerifiableShare<F, G> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<F: Field, G: Group> Mul<F> for VerifiableShare<F, G>
where
    Vector<G>: for<'a> MulAssign<&'a F>,
{
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self.secret *= rhs;
        self.blindness *= rhs;
        self.commitment.0 *= &rhs;
        self
    }
}

#[cfg(test)]
mod tests {

    use curve25519_dalek::{RistrettoPoint, Scalar};
    use ff::PrimeField;
    use rand::thread_rng;

    use super::*;

    #[test]
    fn simple() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let mut rng = thread_rng();
        let v = Scalar::random(&mut rng);

        let gens = PedersenGenParams(
            RistrettoPoint::generator(),
            RistrettoPoint::random(&mut rng),
        );
        let parties: Vec<_> = PARTIES.map(Scalar::from).collect();
        let shares = share::<Scalar, RistrettoPoint>(v, &parties, 2, &mut rng, &gens);

        for (i, share) in parties.iter().zip(shares.iter()) {
            assert!(verify(i, share, &gens));
        }

        let v2 = reconstruct(&shares, &parties);
        assert_eq!(v, v2);
    }

    #[test]
    fn cheaters() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let mut rng = thread_rng();
        let v = Scalar::random(&mut rng);

        let gens = PedersenGenParams(
            RistrettoPoint::generator(),
            RistrettoPoint::random(&mut rng),
        );
        let parties: Vec<_> = PARTIES.map(Scalar::from).collect();
        let mut shares = share::<Scalar, RistrettoPoint>(v, &parties, 2, &mut rng, &gens);

        for share in &mut shares {
            share.secret *= Scalar::from_u128(2);
        }

        for (i, share) in parties.iter().zip(shares.iter()) {
            assert!(!verify(i, share, &gens));
        }
    }
}
