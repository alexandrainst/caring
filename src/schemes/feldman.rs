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
use std::{borrow::Borrow, iter, ops, sync::Arc};

use crate::{
    poly::Polynomial,
    schemes::shamir::{self},
};

use ff::Field;
use group::Group;
use rand::RngCore;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifiableShare<F: Field, G: Group> {
    share: shamir::Share<F>,
    poly: Arc<Polynomial<G>>,
}

impl<F: Field, G> VerifiableShare<F, G>
where
    G: Group + std::ops::Mul<F, Output = G>,
{
    pub fn verify(&self) -> bool {
        let VerifiableShare { share, poly } = self;
        let mut check = G::identity();
        for (i, &a) in poly.0.iter().enumerate() {
            check += a * share.x.pow([i as u64]);
        }
        check == G::generator() * share.y
    }
}

impl<F: Field, G: Group> ops::Add for VerifiableShare<F, G> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        let mut poly = Arc::make_mut(&mut self.poly);
        poly += &rhs.poly;
        Self {
            share: self.share + rhs.share,
            poly: self.poly,
        }
    }
}

impl<F: Field, G: Group> ops::Sub for VerifiableShare<F, G> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        let mut poly = Arc::make_mut(&mut self.poly);
        poly -= &rhs.poly;
        Self {
            share: self.share - rhs.share,
            poly: self.poly,
        }
    }
}

impl<'a, 'b, F: Field, G: Group> ops::Mul<F> for VerifiableShare<F, G> where &'a mut G: ops::MulAssign<&'b F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        let mut poly = Arc::make_mut(&mut self.poly);
        poly *= &rhs;
        Self {
            share: self.share * rhs,
            poly: self.poly,
        }
    }
}

impl<F: Field, G: Group> std::iter::Sum for VerifiableShare<F, G> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut fst = iter.next().unwrap();
        let mut share = fst.share;
        let mut poly_ref = Arc::make_mut(&mut fst.poly);
        for vs in iter {
            share += vs.share;
            poly_ref += &vs.poly;
        }
        VerifiableShare {
            share,
            poly: fst.poly,
        }
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
        let share = shamir::Share::<F> { x, y: share };
        let poly = Polynomial(mac_poly.clone());
        let poly = Arc::new(poly);
        shares.push(VerifiableShare { share, poly });
    }
    shares
}

pub fn reconstruct<F: Field, G: Group>(shares: &[VerifiableShare<F, G>]) -> Option<F>
where
    G: Group + std::ops::Mul<F, Output = G>,
{
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

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct VecVerifiableShare<F: Field, G: Group> {
    shares: shamir::VecShare<F>,
    polys: Arc<[Polynomial<G>]>,
}

impl<F: Field, G: Group> std::ops::Add for &VecVerifiableShare<F, G> {
    type Output = VecVerifiableShare<F, G>;

    fn add(self, rhs: Self) -> Self::Output {
        let shares = &self.shares + &rhs.shares;
        let polys: Arc<[Polynomial<_>]> = self
            .polys
            .iter()
            .cloned()
            .zip(rhs.polys.iter())
            .map(|(mut a, b)| {
                a += b;
                a
            })
            .collect();
        VecVerifiableShare { shares, polys }
    }
}

impl<F: Field, G: Group> std::iter::Sum for VecVerifiableShare<F, G> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let fst = iter.next().unwrap();
        let mut shares = fst.shares;
        // let polys : &mut [Polynomial<_>] = match Arc::get_mut(&mut fst.polys) {
        let mut polys: Vec<Polynomial<_>> = fst.polys.iter().cloned().collect();

        for vs in iter {
            shares += vs.shares;
            polys
                .iter_mut()
                .zip(vs.polys.iter())
                .for_each(|(mut acc, p)| acc += p);
        }
        let polys: Arc<[_]> = polys.into();
        VecVerifiableShare { shares, polys }
    }
}

impl<F: Field, G> VecVerifiableShare<F, G>
where
    G: Group + std::ops::Mul<F, Output = G>,
{
    pub fn verify(&self) -> bool {
        let VecVerifiableShare { shares, polys } = self;
        let x = shares.x;
        for (&y, poly) in shares.ys.iter().zip(polys.iter()) {
            let mut check = G::identity();
            for (i, &a) in poly.0.iter().enumerate() {
                check += a * x.pow([i as u64]);
            }
            if check != G::generator() * y {
                return false;
            }
        }
        true
    }
}

pub fn share_many<F: Field, G: Group>(
    vals: &[F],
    ids: &[F],
    threshold: u64,
    rng: &mut impl RngCore,
) -> Vec<VecVerifiableShare<F, G>>
// FIX: This `where` clause is a bit much, it does however work.
where
    F: ops::Mul<G, Output = G>,
    Box<[G]>: FromIterator<<F as ops::Mul<G>>::Output>,
    for<'a> &'a crate::poly::Polynomial<F>: std::ops::Mul<G, Output = Polynomial<G>>,
{
    let n = ids.len();
    assert!(
        n >= threshold as usize,
        "Threshold should be less-than-equal to the number of shares: t={threshold}, n={n}"
    );
    assert!(
        ids.iter().all(|x| !x.is_zero_vartime()),
        "ID with zero-element provided. Zero-based x coordinates are insecure as they disclose the secret."
    );
    let polys: Vec<_> = vals
        .iter()
        .map(|v| {
            let mut p = Polynomial::<F>::random(threshold as usize, rng);
            p.0[0] = *v;
            p
        })
        .collect();
    let macs: Arc<[Polynomial<G>]> = polys.iter().map(|p| p * G::generator()).collect();

    let mut vshares: Vec<_> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let mut vecshare = Vec::with_capacity(vals.len());
        for (i, _) in vals.iter().enumerate() {
            let y = polys[i].eval(x);
            vecshare.push(y);
        }
        let shares = shamir::VecShare {
            x,
            ys: vecshare.into_boxed_slice(),
        };
        let polys = macs.clone();
        vshares.push(VecVerifiableShare { shares, polys })
    }

    vshares
}

pub fn reconstruct_many<F, G, T>(vec_shares: &[T]) -> Option<Vec<F>>
where
    G: Group + std::ops::Mul<F, Output = G>,
    F: Field,
    G: Group,
    T: Borrow<VecVerifiableShare<F, G>>,
{
    for shares in vec_shares {
        assert!(shares.borrow().verify());
    }

    let shares: Vec<_> = vec_shares.iter().map(|x| &x.borrow().shares).collect();
    Some(shamir::reconstruct_many(&shares))
}

#[cfg(test)]
mod test {
    use curve25519_dalek::{RistrettoPoint, Scalar};
    use rand::{thread_rng, Rng};

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
    fn sharing_many() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);
        let a: Vec<u32> = (0..32).map(|_| rng.gen()).collect();
        let vs1 = {
            let v: Vec<_> = a.clone().into_iter().map(to_scalar).collect();
            let ids: Vec<_> = PARTIES.map(Scalar::from).collect();
            share_many::<_, RistrettoPoint>(&v, &ids, 4, &mut rng)
        };
        for share in &vs1 {
            assert!(share.verify());
        }
        let vsum = reconstruct_many(&vs1).unwrap();
        let v: Vec<u32> = vsum
            .into_iter()
            .map(|x| {
                let x = &x.as_bytes()[0..4];
                let x: [u8; 4] = x.try_into().unwrap();
                u32::from_le_bytes(x)
            })
            .collect();
        assert_eq!(v, a);
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
    fn addition_many() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);
        let a: Vec<u32> = (0..32).map(|_| rng.gen()).collect();
        let b: Vec<u32> = (0..32).map(|_| rng.gen()).collect();
        let vs1 = {
            let v: Vec<_> = a.clone().into_iter().map(to_scalar).collect();
            let ids: Vec<_> = PARTIES.map(Scalar::from).collect();
            share_many::<_, RistrettoPoint>(&v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v: Vec<_> = b.clone().into_iter().map(to_scalar).collect();
            let ids: Vec<_> = PARTIES.map(Scalar::from).collect();
            share_many::<_, RistrettoPoint>(&v, &ids, 4, &mut rng)
        };
        for share in &vs1 {
            assert!(share.verify());
        }
        for share in &vs2 {
            assert!(share.verify());
        }
        let shares: Vec<VecVerifiableShare<_, _>> =
            vs1.into_iter().zip(vs2).map(|(s1, s2)| &s1 + &s2).collect();

        for share in &shares {
            assert!(share.verify());
        }

        let vsum = reconstruct_many(&shares).unwrap();
        let v: Vec<u32> = vsum
            .into_iter()
            .map(|x| {
                let x = &x.as_bytes()[0..4];
                let x: [u8; 4] = x.try_into().unwrap();
                u32::from_le_bytes(x)
            })
            .collect();
        assert_eq!(v, a.iter().zip(b).map(|(a, b)| a + b).collect::<Vec<_>>());
    }

    fn to_scalar(num: u32) -> Scalar {
        let num = num.to_le_bytes();
        let mut arr = [0; 32];
        arr[0] = num[0];
        arr[1] = num[1];
        arr[2] = num[2];
        arr[3] = num[3];
        Scalar::from_bytes_mod_order(arr)
    }

    #[test]
    fn addition_fixpoint() {
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let a = 1.0;
        let b = 3.0;
        type Fix = fixed::FixedU32<16>;
        // Function to pad a u32 to a [u8; 32]

        let a = Fix::from_num(a);
        let b = Fix::from_num(b);
        let v1 = to_scalar(a.to_bits());
        let v2 = to_scalar(b.to_bits());
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
