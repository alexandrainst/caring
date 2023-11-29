//! This is vanilla Shamir Secret Sharing using an arbitrary field F.
//! See <https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing>
use std::borrow::Borrow;

use ff::{derive::rand_core::RngCore, Field};

// TODO: Important! Switch RngCore to CryptoRngCore

use crate::{algebra::math::{Vector, lagrange_coefficients}, net::agency::Unicast, poly::Polynomial};

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
            y: self.y - rhs.y,
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

/// A share with a degree larger than the one.
///
/// This share have been constructed by multiplying two shares togehter,
/// each corresponding to a polynomial `p1`, `p2`, producing a share corresponding
/// to a polynomial `p3` with the degree `|p3| = |p1| + |p2|`.
///
/// You can recover the internal share and act if nothing happened.
/// However, this will encur the penalty of having a larger degree.
///
/// ```ignore
/// # use caring::schemes::shamir::Share;
/// # use caring::element::Element32;
/// # use caring::schemes::shamir::ExplodedShare;
/// # let a = Share{x: Element32::from(2u32), y: Element32::from(2u32)};
/// # let b = a.clone();
/// let a : Share<_> = a; // degree t
/// let b : Share<_> = b; // degree t
/// let c : ExplodedShare<_> = a * b; // degree 2t
/// let c : Share<_> = c.recover(); // degree 2t
/// ```
///
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct ExplodedShare<F: Field>(Share<F>);

impl<F: Field> ExplodedShare<F> {
    /// Recover the internal share from an exploded share.
    ///
    /// This accepts the higher degree WITHOUT a reduction,
    /// thus the new share have a higher degree than the initial two.
    pub fn giveup(self) -> Share<F> {
        self.0
    }

}

impl<F: Field> std::ops::Mul for Share<F> {
    type Output = ExplodedShare<F>;

    /// Multiply a share with itself, producing an "exploded" share.
    ///
    /// * `rhs`: share to multiply with
    ///
    /// The `ExplodedShare` corresponds to a polynomial with double the degree
    /// (if the two inital shares have the same degree)
    ///
    /// This "works" if the amount of shares is greater than the new degree,
    /// however this creates a limit on the amount of multiplications possible.
    ///
    /// To mitigate this you can use the `reduction` protocol.
    /// TODO: Construct that
    ///
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.x, rhs.x);
        ExplodedShare(Self {
            x: self.x,
            y: self.y * self.y,
        })
    }
}

/// Reduction of the associated polynomial on share.
///
/// * `z`: ExplodedShare to recover
/// * `unicast`: Unicasting functionality
/// * `threshold`: the original threshold
/// * `ids`: Party IDs to share to
/// * `rng`: Random number generator
/// ```ignore
/// # use caring::schemes::shamir::Share;
/// # use caring::element::Element32;
/// # use caring::schemes::shamir::ExplodedShare;
/// # let a = Share{x: Element32::from(2u32), y: Element32::from(2u32)};
/// # let b = a.clone();
/// let a : Share<_> = a; // degree t
/// let b : Share<_> = b; // degree t
/// let c : ExplodedShare<_> = a * b; // degree 2t
/// let c : Share<_> = reduction(c, network, threshold, ids, rng);
/// //  ^-- degree t
/// ```
pub async fn reducto<F: Field + serde::Serialize + serde::de::DeserializeOwned, U: Unicast>(
    z: ExplodedShare<F>,
    unicast: &mut U,
    threshold: u64,
    ids: &[F],
    rng: &mut impl RngCore,
) -> Result<Share<F>, U::Error> {
    // FIX: Doesn't work
    // TODO: Maybe use the `Shared` functionality?
    let z = z.0;
    let i = z.x;

    let n  = ids.len();
    assert!(n >= 2*threshold as usize);
    // We need 2t < n, otherwise we cannot reconstruct,
    // however 't' is hidden from before, so we just have to assume it is.
    // Now we need to reduce the polynomial back to t

    // issued subshares
    let subshares = share(z.y, ids, threshold, rng); // share -> subshares
                                                     //
    // Something about a recombination vector and randomization.

    // // randominization
    // let am_i_special = false;
    // let my_id = 0;

    // let poly_share = if am_i_special {
    //     // Should one or all parties do this?
    //     let mut zero_poly : Polynomial<F> = Polynomial::random(2*threshold as usize - 1, rng);
    //     zero_poly.0[0] = F::ZERO;
    //     let mut shares = share_many(&zero_poly.0, ids, threshold, rng);
    //     let mine = shares.remove(my_id);
    //     unicast.unicast(&shares);
    //     mine
    // } else {
    //     todo!()
    // };
    // let poly = poly_share.ys.into_iter().collect::<Polynomial<F>>();
    // // held subshares
    // for subshare in subshares.iter_mut() {
    //     // x can be subbed if we know our id
    //     subshare.y += poly.eval(&subshare.x);
    // }

    let subshares = unicast.symmetric_unicast::<_>(subshares).await?;

    // reduction (cache?)
    let coeffs = lagrange_coefficients(ids, F::ZERO);

    // inner product
    let z : F = subshares.into_iter().zip(coeffs).map(|(a,b)| a.y*b).sum();

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

    // Sample random t-degree polynomial, where t = threshold - 1, since we need
    // t+1 shares to construct a t-degree polynomial.
    let mut polynomial = Polynomial::random(threshold as usize - 1, rng);
    polynomial.0[0] = v;
    let polynomial = polynomial;

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<Share<F>> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let y = polynomial.eval(&x);
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

use derive_more::{Add, AddAssign};
/// A secret shared vector
///
/// * `x`: the id
/// * `ys`: share values
#[derive(Clone, serde::Deserialize, serde::Serialize, AddAssign)]
pub struct VecShare<F: Field> {
    pub(crate) x: F,
    pub(crate) ys: Vector<F>,
}

impl<F: Field> std::ops::Add for VecShare<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x,
            ys: self.ys + rhs.ys,
        }
    }
}

impl<F: Field> std::ops::Add for &VecShare<F> {
    type Output = VecShare<F>;

    fn add(self, rhs: Self) -> Self::Output {
        let a = &self.ys;
        let b = &rhs.ys;
        let ys: Vector<_> = a + b;
        VecShare { x: self.x, ys }
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
        ys.into_iter().map(|y| Share { x, y }).collect()
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

/// Share/shard `m` secret values `vs` into `n * m` shares
/// where `n` is the number of the `ids`
///
/// The result is a vector of vectorized shares `VecShare`,
/// with each `VecShare` being `m` shares corresponding to a party.
///
/// * `vs`: secret values to share
/// * `ids`: ids to share to
/// * `threshold`: threshold to reconstruct it
/// * `rng`: rng to generate shares from
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

    // Sample `m` random `t`-degree polynomials
    // where t = threshold - 1
    let polynomials: Vec<_> = vs
        .iter()
        .map(|v| {
            let mut p = Polynomial::random(threshold as usize - 1, rng);
            p.0[0] = *v;
            p
        })
        .collect();

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<VecShare<F>> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let vecshare = if cfg!(feature = "rayon") {
            vs.par_iter()
                .enumerate()
                .map(|(i, _)| polynomials[i].eval(&x))
                .collect()
        } else {
            vs.iter()
                .enumerate()
                .map(|(i, _)| polynomials[i].eval(&x))
                .collect()
        };
        shares.push(VecShare { x, ys: vecshare })
    }

    shares
}

use rayon::prelude::*;

/// Reconstruct or open shares
///
/// * `shares`: shares to be combined into open values
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
        let yi: &Vector<_> = &share.borrow().ys;

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
    use crate::algebra::element::Element32;

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
        // We test that we can secret-share two numbers and add them.
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
        // We test that we can secret-share two vectors of numbers and add them pairwise.
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
    use rand::Rng;

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
    async fn multiplication() {
        let cluster = crate::testing::Cluster::new(5);
        cluster
            .run(|mut network| async move {
                // setup
                let input: u32 = 5;
                let mut rng = rand::rngs::mock::StepRng::new(0, 7);
                let ids: Vec<_> = network
                    .participants()
                    .map(|i| i + 1)
                    .map(Element32::from)
                    .collect();
                let threshold = 2;

                // secret-sharing
                let shares = share(input.into(), &ids, threshold, &mut rng);
                let shares = network.symmetric_unicast(shares).await.unwrap();
                let [a, b, ..] = shares[..] else { panic!("Can't multiply with only one share") };

                dbg!(&a);
                dbg!(&b);

                // mpc
                let c = a * b;
                dbg!(&c);
                // HACK: It doesn't work yet.
                //
                let c = reducto(c, &mut network, threshold, &ids, &mut rng)
                    .await
                    .expect("reducto failed");
                //let c = c.giveup();
                dbg!(&c);

                // opening
                let shares = network.symmetric_broadcast(c).await.unwrap();
                let c: u32 = reconstruct(&shares).into();

                assert_eq!(c, 25);
            })
            .await
            .unwrap();
    }
}
