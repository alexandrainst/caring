//! This is an implementation of verifiable secret sharing using Feldman's scheme
//! see https://en.wikipedia.org/wiki/Verifiable_secret_sharing#Feldman's_scheme
//! The scheme can be instansiated with any field F and a corresponding group G
//! for which there exists a mapping F -> G using a generator `g`.
//! It should also be noted that the discrete log problem in G should be *hard*.
//!
//! So we probably could use a 'automated' way to verify shares received.
//! Now, we never need to verify our own shares, and the easy way is just to do it
//! before each operation and reconstruction. However, these verifications could be done eagerly
//! when receiving a share, parallel to everything else, and just 'awaited' before sending
//! anything based on that.
//!
use std::{iter, ops, sync::Arc};

use crate::{shamir::{self, Share}, poly::Polynomial};

use ff::Field;
use group::Group;
use rand::RngCore;


#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifiableShare<F: Field, G: Group> {
    share: Share<F>,
    poly: Arc<Polynomial<G>>,
}
// Consider adding the polynomial here for ease of use,
// since they are always coupled. Note however that we
// probably don't want to perform clones of the polynomial
// when generating shares, sooooo we probably need to use
// something like `Cow`, `Rc`, `Arc` or the like.

impl<F: Field,G> VerifiableShare<F, G> where G: Group + std::ops::Mul<F, Output = G> {
    pub fn verify(&self) -> bool
    {
        let VerifiableShare { share, poly } = self;
        let mut check = G::identity();
        for (i, &a) in poly.0.iter().enumerate() {
            check += a * share.x.pow([i as u64]);
        }
        check == G::generator() * share.y
    }
}

impl<F: Field, G: Group> ops::Add for VerifiableShare<F,G> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        let poly = Arc::make_mut(&mut self.poly);
        poly.add_self(&rhs.poly);
        Self {
            share: self.share + rhs.share,
            poly: self.poly
        }
    }
}


impl<F: Field, G: Group> ops::Sub for VerifiableShare<F,G> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let poly = Arc::make_mut(&mut self.poly);
        poly.sub_self(&rhs.poly);
        Self {
            share: self.share - rhs.share,
            poly: self.poly
        }
    }
}

impl<F: Field, G: Group> std::iter::Sum for VerifiableShare<F,G> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut fst = iter.next().unwrap();
        let mut share = fst.share;
        let poly_ref = Arc::make_mut(&mut fst.poly);
        for vs in iter {
            share += vs.share;
            poly_ref.add_self(&vs.poly);
        }
        VerifiableShare { share, poly: fst.poly }
    }
}

pub fn share<F: Field, G: Group>(
    val: F,
    ids: &[F],
    threshold: u64,
    rng: &mut impl RngCore,
) -> Vec<VerifiableShare<F, G>>
where
    G: std::ops::Mul<F, Output = G>,
{
    // let shares = shamir::share(val, threshold, rng);
    // 1. We need to get the polynomial.
    // 2. We then need to do commitments to it.
    // 3. We need to provide commitments/macs to the corresponding shares.
    // Then pack these macs with the shares and output them.

    // there are some code-duplication with `shamir.rs` currently.
    // that will probably be fixed.

    // Sample random t-degree polynomial
    let n = ids.len();
    let poly = (1..threshold).map(|_| F::random(&mut *rng));
    // I want to avoid this allocation :(
    let poly: Box<[F]> = iter::once(val).chain(poly).collect();
    let mac_poly: Box<[G]> = poly.iter().map(|a| G::generator() * *a).collect();

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<_> = Vec::with_capacity(n);

    for x in ids {
        let x = *x;
        let share = poly
            .iter()
            .enumerate()
            .map(|(i, a)| -> F {
                // evaluate: a * x^i
                *a * x.pow([i as u64])
            }) // sum: s + a1 x + a2 x^2 + ...
            .fold(F::ZERO, |sum, x| sum + x);
        let share = Share::<F> { x, y: share };
        let poly = Polynomial(mac_poly.clone());
        let poly = Arc::new(poly);
        shares.push(VerifiableShare { share, poly });
    }

    shares
}

pub fn reconstruct<F: Field, G: Group>(
    shares: &[VerifiableShare<F, G>],
) -> Option<F>
where G: Group + std::ops::Mul<F, Output = G> {
    // let (shares, macs) : (Vec<_>, Vec<_>) = shares.iter().map(|s| (s.share)).unzip();
    for share in shares {
        if !share.verify() {
            return None;
        }
    }
    let shares: Vec<_> = shares.iter().map(|s| s.share).collect();
    let res = shamir::reconstruct(&shares);

    Some(res)
}

#[cfg(test)]
mod test {
    use curve25519_dalek::{RistrettoPoint, Scalar};
    use rand::thread_rng;

    use super::*;

    #[test]
    fn sharing() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let mut rng = thread_rng();
        let v = Scalar::random(&mut rng);

        let parties: Vec<_> = PARTIES.map(Scalar::from).collect();
        let shares = share::<Scalar, RistrettoPoint>(v, &parties, 2, &mut rng);
        for share in &shares {
            assert!(share.verify());
        }
        let v2 = reconstruct(&shares).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn addition() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let mut rng = thread_rng();
        let v1 = Scalar::from_bytes_mod_order([7; 32]);
        let v2 = Scalar::from_bytes_mod_order([10; 32]);

        let parties: Vec<_> = PARTIES.map(Scalar::from).collect();
        let shares1 = share::<Scalar, RistrettoPoint>(v1, &parties, 2, &mut rng);
        let shares2 = share::<Scalar, RistrettoPoint>(v2, &parties, 2, &mut rng);
        for share in &shares1 {
            assert!(share.verify());
        }
        for share in &shares2 {
            assert!(share.verify());
        }
        let shares: Vec<_> = shares1
            .into_iter()
            .zip(shares2)
            .map(|(s1, s2)| s1 + s2)
            .collect();

        for share in &shares {
            assert!(share.verify());
        }

        let vsum = reconstruct(&shares).unwrap();
        assert_eq!(v1 + v2, vsum);
    }

    #[test]
    fn addition_fixpoint() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let a = 1.0;
        let b = 3.0;
        type Fix = fixed::FixedU32<16>;
        // Function to pad a u32 to a [u8; 32]
        fn pad(num: u32) -> [u8; 32] {
            let num = num.to_le_bytes();
            let mut arr = [0; 32];
            arr[0] = num[0];
            arr[1] = num[1];
            arr[2] = num[2];
            arr[3] = num[3];
            arr
        }

        let a = Fix::from_num(a);
        let b = Fix::from_num(b);
        let v1 = Scalar::from_bytes_mod_order(pad(a.to_bits()));
        let v2 = Scalar::from_bytes_mod_order(pad(b.to_bits()));
        v2.invert();

        let mut rng = thread_rng();
        let parties: Vec<_> = PARTIES.map(Scalar::from).collect();
        let shares1 = share::<Scalar, RistrettoPoint>(v1, &parties, 2, &mut rng);
        let shares2 = share::<Scalar, RistrettoPoint>(v2, &parties, 2, &mut rng);
        for share in &shares1 {
            assert!(share.verify());
        }
        for share in &shares2 {
            assert!(share.verify());
        }
        let shares: Vec<_> = shares1
            .into_iter()
            .zip(shares2)
            .map(|(s1, s2)| s1 + s2)
            .collect();

        for share in &shares {
            assert!(share.verify());
        }

        let vsum = reconstruct(&shares).unwrap();
        let sum = &vsum.as_bytes()[0..4];
        let sum: [u8; 4] = sum.try_into().unwrap();
        let sum = Fix::from_le_bytes(sum);
        assert_eq!(a + b, sum);
    }
}
