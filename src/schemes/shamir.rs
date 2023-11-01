//! This is vanilla Shamir Secret Sharing using an arbitrary field F.
//! See https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing
use std::borrow::Borrow;

use ff::{derive::rand_core::RngCore, Field};
use itertools::multiunzip;

// TODO: Important! Switch RngCore to CryptoRngCore

use crate::{
    agency::{Broadcast, Unicast},
    poly::Polynomial,
};

/// A Shamir Secret Share
/// This is a point evaluated at `x` given a secret polynomial.
/// Reconstruction can be done by obtaining `t` shares.
/// Shares with the same `x` can be added together.
/// Likewise can shares also be multiplied by a constant.
///
/// * `x`: The id of the share
/// * `y`: The 'share' part of the share
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Share<F: Field> {
    // NOTE: Consider
    //removing 'x' as it should be implied by the user handling it
    pub(crate) x: F,
    pub(crate) y: F,
}

// TODO: We could use a construct representing a group of shares,
// this could probably allow for the removal of the `x` in the Share.
// This should allow for an easier 'sharing' phase, where each party
// gets their correct version.

impl<F: Field> std::ops::Add for Share<F> {
    type Output = Self;

    /// Add two shares together.
    /// Note: These must share the same `x` value.
    ///
    /// * `rhs`: the other share
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.x, rhs.x);
        Self {
            x: self.x,
            y: self.y + rhs.y,
        }
    }
}

impl<F: Field> std::ops::AddAssign for Share<F> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.x, rhs.x);
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<F: Field> std::ops::Sub for Share<F> {
    type Output = Self;

    /// Add two shares together.
    /// Note: These must share the same `x` value.
    ///
    /// * `rhs`: the other share
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.x, rhs.x);
        Self {
            x: self.x,
            y: self.y + rhs.y,
        }
    }
}

impl<F: Field> std::ops::Add<F> for Share<F> {
    type Output = Self;

    /// Add a field to a share.
    /// This 'acts' as the field element is the in the same `x` point.
    ///
    /// * `rhs`: the field element to add
    /// NOTE: This will be redundant if we remove `x` as a share will be a field element
    fn add(self, rhs: F) -> Self::Output {
        Self {
            x: self.x,
            y: self.y + rhs,
        }
    }
}

impl<F: Field> std::iter::Sum for Share<F> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(Share { x, y }) = iter.next() else {
            return Share {
                x: F::ZERO,
                y: F::ZERO,
            };
        };
        let mut acc = y;
        for share in iter {
            assert_eq!(share.x, x);
            acc += share.y;
        }
        Share { x, y: acc }
    }
}

impl<F: Field> std::ops::Mul<F> for Share<F> {
    type Output = Self;

    /// Multiply a share with a field element
    ///
    /// * `rhs`: field element to multiply with
    fn mul(self, rhs: F) -> Self::Output {
        Self {
            y: self.y * rhs,
            ..self
        }
    }
}

// TODO: Move beaver stuff out, and generify over any kind of share.
// This might require a common 'Share' trait, or maybe just something that
// implements multiplication, who knows.
#[derive(Clone)]
pub struct BeaverTriple<F: Field>(Share<F>, Share<F>, Share<F>);

impl<F: Field> BeaverTriple<F> {
    /// Fake a set of beaver triples.
    ///
    /// This produces `n` shares corresponding to a shared beaver triple,
    /// however locally with the values known.
    ///
    /// * `ids`: ids to produce for
    /// * `threshold`: threshold to reconstruct
    /// * `rng`: rng to sample from
    pub fn fake(ids: &[F], threshold: u64, mut rng: &mut impl RngCore) -> Vec<Self> {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = a * b;
        // Share (preproccess)
        let a = share(a, ids, threshold, &mut rng);
        let b = share(b, ids, threshold, &mut rng);
        let c = share(c, ids, threshold, &mut rng);
        itertools::izip!(a, b, c)
            .map(|(a, b, c)| Self(a, b, c))
            .collect()
    }
}

/// Perform multiplication using beaver triples
///
/// * `x`: first share to multiply
/// * `y`: second share to multiply
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn beaver_multiply<F: Field + serde::Serialize + serde::de::DeserializeOwned, E>(
    x: Share<F>,
    y: Share<F>,
    triple: BeaverTriple<F>,
    network: &mut impl Broadcast<E>,
) -> Result<Share<F>, E> {
    let BeaverTriple(a, b, c) = triple;
    let ax = a + x;
    let by = b + y;

    // Sending both at once it more efficient.
    let resp = network.symmetric_broadcast::<_>((ax, by)).await?;
    let (ax, by): (Vec<_>, Vec<_>) = multiunzip(resp);

    let ax = reconstruct(&ax);
    let by = reconstruct(&by);

    Ok(y * ax + a * (-by) + c)
}

// TODO: Maybe cut the regular multiplication protocol out and allow multiplying shares directly,
// at the conseqeunce of increasing their degree? (Sidenote: maybe introduce degree tracking?)
// Instead provide a protocol for the degree reduction.
pub async fn regular_multiply<
    // FIX: Doesn't work
    F: Field + serde::Serialize + serde::de::DeserializeOwned, E
>(
    x: Share<F>,
    y: Share<F>,
    network: &mut impl Unicast<E>,
    threshold: u64,
    ids: &[F],
    rng: &mut impl RngCore,
) -> Result<Share<F>, E> {
    let i = x.x;
    // We need 2t < n, otherwise we cannot reconstruct,
    // however 't' is hidden from before, so we jyst have to assume it is.
    // x, y: degree t
    let z = x.y * y.y; // z: degree 2t
                       // Now we need to reduce the polynomial back to t
    let z = share(z, ids, threshold, rng); // share -> subshares
    let z = network.symmetric_unicast::<_>(z).await?;
                                                // Something about a recombination vector and randomization.
    let z = reconstruct(&z); // reconstruct the subshare
    Ok(Share { x: i, y: z })
}

/// Share/shard a secret value `v` into `n` shares
/// where `n` is the number of the `ids`
///
/// * `v`: secret value to share
/// * `ids`: ids to share to
/// * `threshold`: threshold to reconstruct it
/// * `rng`: rng to generate shares from
pub fn share<F: Field>(v: F, ids: &[F], threshold: u64, rng: &mut impl RngCore) -> Vec<Share<F>> {
    let n = ids.len();
    assert!(
        n >= threshold as usize,
        "Threshold should be less-than-equal to the number of shares: t={threshold}, n={n}"
    );
    assert!(
        ids.iter().all(|x| !x.is_zero_vartime()),
        "ID with zero-element provided. Zero-based x coordinates are insecure as they disclose the secret."
    );

    // Sample random t-degree polynomial
    let mut polynomial = Polynomial::random(threshold as usize, rng);
    polynomial.0[0] = v;
    let polynomial = polynomial;

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<Share<F>> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let y = polynomial.eval(x);
        shares.push(Share::<F> { x, y });
    }

    shares
}

/// Reconstruct or open shares
///
/// * `shares`: shares to be combined into an open value
pub fn reconstruct<F: Field>(shares: &[Share<F>]) -> F {
    // Lagrange interpolation:
    // L(x) = sum( y_i * l_i(x) )
    // where l_i(x) = prod( (x - x_k)/(x_i - x_k) | k != i)
    // here we always evaluate with x = 0
    let mut sum = F::ZERO;
    for share in shares.iter() {
        let xi = share.x;
        let yi = share.y;

        let mut prod = F::ONE;
        for Share { x: xk, y: _ } in shares.iter() {
            let xk = *xk;
            if xk != xi {
                prod *= -xk * (xi - xk).invert().unwrap_or(F::ZERO)
            }
        }
        sum += yi * prod;
    }
    sum
}

/// A secret shared vector
///
/// * `x`: the id
/// * `ys`: share values
#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct VecShare<F: Field> {
    pub(crate) x: F,
    pub(crate) ys: Box<[F]>,
}

// impl<F: Field> AsRef<Self> for VecShare<F> {
//     fn as_ref(&self) -> &Self {
//         self
//     }
// }

impl<F: Field> std::ops::Add for &VecShare<F> {
    type Output = VecShare<F>;

    fn add(self, rhs: Self) -> Self::Output {
        let x = self.x;
        let ys = if cfg!(feature = "rayon") {
            self.ys
                .par_iter()
                .zip(rhs.ys.par_iter())
                .map(|(&a, &b)| a + b)
                .collect()
        } else {
            self.ys
                .iter()
                .zip(rhs.ys.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        };
        VecShare { x, ys }
    }
}

impl<F: Field> std::ops::AddAssign for VecShare<F> {
    fn add_assign(&mut self, rhs: Self) {
        if cfg!(feature = "rayon") {
            self.ys
                .par_iter_mut()
                .zip(rhs.ys.par_iter())
                .for_each(|(a, b)| *a += b)
        } else {
            self.ys
                .iter_mut()
                .zip(rhs.ys.iter())
                .for_each(|(a, b)| *a += b)
        }
    }
}

impl<F: Field> From<Vec<Share<F>>> for VecShare<F> {
    fn from(value: Vec<Share<F>>) -> Self {
        let x = value[0].x;
        let ys = value.into_iter().map(|Share { x: _, y }| y).collect();
        VecShare { x, ys }
    }
}

impl<F: Field> From<VecShare<F>> for Vec<Share<F>> {
    fn from(value: VecShare<F>) -> Self {
        let VecShare { x, ys } = value;
        ys.iter().map(|&y| Share { x, y }).collect()
    }
}

impl<F: Field> std::iter::Sum for VecShare<F> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut first = iter.next().expect("Please don't sum zero items together");
        for elem in iter {
            first += elem;
        }
        first
    }
}

pub fn share_many<F: Field>(
    vs: &[F],
    ids: &[F],
    threshold: u64,
    rng: &mut impl RngCore,
) -> Vec<VecShare<F>> {
    // FIX: Code duplication with 'share'
    let n = ids.len();
    assert!(
        n >= threshold as usize,
        "Threshold should be less-than-or-equal to the number of shares: t={threshold}, n={n}"
    );

    // Sample random t-degree polynomial
    let polynomials: Vec<_> = vs
        .iter()
        .map(|v| {
            let mut p = Polynomial::random(threshold as usize, rng);
            p.0[0] = *v;
            p
        })
        .collect();

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<VecShare<F>> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let vecshare: Box<[_]> = if cfg!(feature = "rayon") {
            vs.par_iter()
                .enumerate()
                .map(|(i, _)| polynomials[i].eval(x))
                .collect()
        } else {
            vs.iter()
                .enumerate()
                .map(|(i, _)| polynomials[i].eval(x))
                .collect()
        };
        shares.push(VecShare { x, ys: vecshare })
    }

    shares
}

use rayon::prelude::*;

pub fn reconstruct_many<F: Field>(shares: &[impl Borrow<VecShare<F>>]) -> Vec<F> {
    // FIX: Code duplication with 'reconstruction'
    //
    // Lagrange interpolation:
    // L(x) = sum( y_i * l_i(x) )
    // where l_i(x) = prod( (x - x_k)/(x_i - x_k) | k != i)
    // here we always evaluate with x = 0
    let m = shares[0].borrow().ys.len();
    let mut sum = vec![F::ZERO; m];
    for share in shares.iter() {
        let xi = share.borrow().x;
        let yi = &share.borrow().ys;

        let mut prod = F::ONE;
        for VecShare { x: xk, ys: _ } in shares.iter().map(|s| s.borrow()) {
            let xk = *xk;
            if xk != xi {
                prod *= -xk * (xi - xk).invert().unwrap_or(F::ZERO)
            }
        }
        // If we are using rayon (running in parallel)
        if cfg!(feature = "rayon") {
            sum.par_iter_mut()
                .zip(yi.par_iter())
                .for_each(|(sum, &yi)| *sum += yi * prod);
        } else {
            sum.iter_mut()
                .zip(yi.iter())
                .for_each(|(sum, &yi)| *sum += yi * prod);
        }
    }
    sum
}

#[cfg(test)]
mod test {
    use crate::{network::InMemoryNetwork, element::Element32};

    use super::*;

    #[test]
    fn simple() {
        // We test that we can secret-share a number and reconstruct it.
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        let v = Element32::from(42u32);
        let ids: Vec<_> = (1..=5u32).map(Element32::from).collect();
        let shares = share(v, &ids, 4, &mut rng);
        let v: u32 = reconstruct(&shares).into();
        assert_eq!(v, 42u32);
    }

    #[test]
    fn addition() {
        // We test that we can secret-share a two numbers and add them.
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let a = 3;
        let b = 7;
        let vs1 = {
            let v = Element32::from(a);
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b);
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };

        // MPC
        let shares: Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a, &b)| a + b).collect();
        let v = reconstruct(&shares);
        let v: u32 = v.into();
        assert_eq!(v, a + b);
    }

    #[test]
    fn addition_many() {
        // We test that we can secret-share a two numbers and add them.
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let a: Vec<u32> = (0..256).map(|_| rng.gen_range(1..100)).collect();
        let b: Vec<u32> = (0..256).map(|_| rng.gen_range(1..100)).collect();
        let vs1 = {
            let v: Vec<_> = a.clone().into_iter().map(Element32::from).collect();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share_many(&v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v: Vec<_> = b.clone().into_iter().map(Element32::from).collect();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share_many(&v, &ids, 4, &mut rng)
        };

        // MPC
        let shares: Vec<_> = vs1.iter().zip(vs2.iter()).map(|(a, b)| a + b).collect();
        let v = reconstruct_many(&shares);
        let v: Vec<u32> = v.into_iter().map(|x| x.into()).collect();
        assert_eq!(v, a.iter().zip(b).map(|(a, b)| a + b).collect::<Vec<_>>());
    }

    use fixed::FixedU32;
    use rand::{thread_rng, Rng};

    #[test]
    fn addition_fixpoint() {
        // We test that we can secret-share a two *fixed point* numbers and add them.
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        type Fix = FixedU32<16>;
        let a = 1.0;
        let b = 3.0;
        let a = Fix::from_num(a);
        let b = Fix::from_num(b);

        let vs1 = {
            let v = Element32::from(a.to_bits() as u64);
            dbg!(&v);
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b.to_bits() as u64);
            dbg!(&v);
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };

        // MPC
        let shares: Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a, &b)| a + b).collect();
        let v = reconstruct(&shares);
        dbg!(v);

        // back to fixed
        let v: u32 = v.into();
        let v = Fix::from_bits(v);
        assert_eq!(v, a + b);
    }

    #[tokio::test]
    async fn beaver_mult() {
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        let parties: Vec<Element32> = (0..2u32).map(|i| i + 1).map(Element32::from).collect();

        let treshold: u64 = 2;

        // preproccessing
        let triples = BeaverTriple::fake(&parties, treshold, &mut rng);

        let mut taskset = tokio::task::JoinSet::new();
        // MPC
        let cluster = InMemoryNetwork::in_memory(parties.len());
        for network in cluster {
            let parties = parties.clone();
            let triple = triples[network.index].clone();
            taskset.spawn({
                async move {
                    let mut rng = rand::rngs::mock::StepRng::new(1, 7);
                    let mut network = network;
                    let v = Element32::from(5u32);
                    let shares = share(v, &parties, treshold, &mut rng);
                    let shares = network.symmetric_unicast(shares).await.unwrap();
                    let res = beaver_multiply(shares[0], shares[1], triple, &mut network).await.unwrap();
                    let res = network.symmetric_broadcast(res).await.unwrap();
                    let res = reconstruct(&res);
                    let res: u32 = res.into();
                    assert_eq!(res, 25);
                }
            });
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }

    #[tokio::test]
    async fn regular_mult() {
        let mut rng = thread_rng();
        let parties: Vec<Element32> = (0..10u32).map(|i| i + 1).map(Element32::from).collect();

        let treshold: u64 = 3;

        // preproccessing
        let _triples = BeaverTriple::fake(&parties, treshold, &mut rng);

        let mut taskset = tokio::task::JoinSet::new();
        // MPC
        let cluster = InMemoryNetwork::in_memory(parties.len());
        for network in cluster {
            let parties = parties.clone();
            taskset.spawn({
                async move {
                    let mut rng = rand::rngs::mock::StepRng::new(1, 7);
                    let mut network = network;
                    let v = Element32::from(5u32);
                    let shares = share(v, &parties, treshold, &mut rng);
                    let shares = network.symmetric_unicast(shares).await.unwrap();
                    // let res = regular_multiply(
                    //     shares[0],
                    //     shares[1], // we ignore the rest of the inputs
                    //     &mut network,
                    //     treshold,
                    //     &parties,
                    //     &mut rng,
                    // ).await;
                    let res = Share {
                        x: shares[0].x,
                        y: shares[0].y * shares[1].y,
                    };
                    let res = network.symmetric_broadcast(res).await.unwrap();
                    let res = reconstruct(&res);
                    let res: u32 = res.into();
                    assert_eq!(res, 25);
                }
            });
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
}