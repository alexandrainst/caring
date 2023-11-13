//! Implementation of Pedersen Secret Sharing
//! <https://www.cs.cornell.edu/courses/cs754/2001fa/129.PDF>
use ff::Field;
use group::Group;
use rand::RngCore;

use crate::poly::Polynomial;

use derive_more::*;
#[derive(Add, Sub, Neg)]
pub struct VerifiableShare<F: Field>(F, F);
pub struct Commitment<G>(Box<[G]>);

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
) -> (Vec<VerifiableShare<F>>, Commitment<G>)
where
    G: std::ops::Mul<F, Output = G>,
{
    let mut p1: Polynomial<F> = Polynomial::random(threshold, rng);
    let p2: Polynomial<F> = Polynomial::random(threshold, rng);
    // secret `s`
    p1.0[0] = secret;
    // random secret `t` (left random)

    let shares: Vec<VerifiableShare<F>> = ids
        .iter()
        .map(|i| {
            // Sample the shares
            VerifiableShare(p1.eval(i), p2.eval(i))
        })
        .collect();

    let commitments: Box<[G]> =
        p1.0.iter()
            .zip(p2.0.iter())
            .map(|(&a, &b)| *g * a + *h * b)
            .collect();
    let commitments = Commitment(commitments);

    (shares, commitments)
}

pub fn verify<F, G>(
    id: &F,
    share: &VerifiableShare<F>,
    commit: &Commitment<G>,
    PedersenGenParams(g, h): &PedersenGenParams<G>,
) -> bool
where
    F: Field,
    G: Group + std::ops::Mul<F, Output = G>,
{
    // C0^(i^0) * C1^(i^1) * C1^(i^2) + ...
    let mut check = G::identity();
    for (i, &a) in commit.0.iter().enumerate() {
        check += a * id.pow([i as u64]);
    }

    // g^s * h^t =?= ...
    *g * share.0 + *h * share.1 == check
}

pub fn reconstruct<F: Field>(shares: &[VerifiableShare<F>], ids: &[F]) -> F {
    // TODO: Maybe verify that the shares are all correct.

    // HACK: Parse as shamir, since we just use lagrange interpolation.
    let shares: Vec<_> = shares
        .iter()
        .zip(ids)
        .map(|(share, id)| super::shamir::Share { x: *id, y: share.0 })
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
        let (shares, commit) = share::<Scalar, RistrettoPoint>(v, &parties, 2, &mut rng, &gens);

        for (i, share) in parties.iter().zip(shares.iter()) {
            assert!(verify(i, share, &commit, &gens));
        }

        let v2 = reconstruct(&shares, &parties);
        assert_eq!(v, v2);
    }
}
