//! Implementation of Pedersen Secret Sharing
//! <https://www.cs.cornell.edu/courses/cs754/2001fa/129.PDF>
use ff::Field;
use group::Group;
use rand::RngCore;

use crate::algebra::poly::Polynomial;


pub struct VerifiableShare<F: Field, G: Group> {
    secret: F,
    blindness: F,
    commitment: Polynomial<G>,
}

/// Generators for Pedersen Secret Sharing (and Pedersen Commitments)
///
/// Consists of two group elements (g,h), usally elliptic curve points.
///
/// These are cosntant public values, and should have a default implementation for ease of use.
/// NOTE: We can't really provide general default implementation for any Group,
/// but we should possibly do it for some groups e.g., curve25519.
pub struct PedersenGenParams<G: Group>(G, G);

pub fn share<F: Field, G: Group>(
    secret: F,
    ids: &[F],
    threshold: usize,
    rng: &mut impl RngCore,
    PedersenGenParams(g, h): &PedersenGenParams<G>,
) -> Vec<VerifiableShare<F, G>>
where
    G: std::ops::Mul<F, Output = G>,
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
            VerifiableShare{secret: p1.eval(i), blindness: p2.eval(i), commitment: commitments.clone()}
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
    let VerifiableShare { secret, blindness, commitment } = share;
    // C0^(i^0) * C1^(i^1) * C1^(i^2) + ...
    let mut check = G::identity();
    for (i, &a) in commitment.0.iter().enumerate() {
        check += a * id.pow([i as u64]);
    }

    // g^s * h^t =?= ...
    *g * *secret + *h * *blindness == check
}

pub fn reconstruct<F: Field, G: Group>(shares: &[VerifiableShare<F, G>], ids: &[F]) -> F {
    // TODO: Maybe verify that the shares are all correct.

    // HACK: Parse as shamir, since we just use lagrange interpolation.
    let shares: Vec<_> = shares
        .iter()
        .zip(ids)
        .map(|(share, id)| super::shamir::Share { x: *id, y: share.secret })
        .collect();

    super::shamir::reconstruct(&shares)
}

#[cfg(test)]
mod tests {

    use curve25519_dalek::{RistrettoPoint, Scalar};
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
}
