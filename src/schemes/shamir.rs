//! This is vanilla Shamir Secret Sharing using an arbitrary field F.
//! See <https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing>
use std::{borrow::Borrow, error::Error};

use ff::{derive::rand_core::RngCore, Field};

// TODO: Important! Switch RngCore to CryptoRngCore

use crate::{
    algebra::math::{lagrange_coefficients, Vector},
    net::agency::Unicast,
    schemes::InteractiveMult,
};

use crate::algebra::poly::Polynomial;

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
    pub(crate) y: F,
}

#[derive(Clone)]
pub struct ShamirParams<F> {
    pub threshold: u64,
    pub ids: Vec<F>,
}

// TODO: Collapse Field with Ser-De since we always require that combo?
impl<F: Field + serde::Serialize + serde::de::DeserializeOwned> super::Shared<F> for Share<F> {
    type Context = ShamirParams<F>;

    fn share(ctx: &Self::Context, secret: F, rng: &mut impl RngCore) -> Vec<Self> {
        share(secret, &ctx.ids, ctx.threshold, rng)
    }

    fn recombine(ctx: &Self::Context, shares: &[Self]) -> Option<F> {
        Some(reconstruct(ctx, shares))
    }
}

impl<F: Field + serde::Serialize + serde::de::DeserializeOwned> InteractiveMult<F> for Share<F> {
    async fn interactive_mult<U: Unicast>(
        ctx: &Self::Context,
        net: &mut U,
        a: Self,
        b: Self,
    ) -> Result<Self, Box<dyn Error>> {
        let c = a * b;
        let mut rng = rand::thread_rng();
        let c = deflate(ctx, c, net, &mut rng).await?;
        Ok(c)
    }
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
        shares.push(Share::<F> { y });
    }

    shares
}

/// Reconstruct or open shares
///
/// * `shares`: shares to be combined into an open value
pub fn reconstruct<F: Field>(ctx: &ShamirParams<F>, shares: &[Share<F>]) -> F {
    // Lagrange interpolation:
    // L(x) = sum( y_i * l_i(x) )
    // where l_i(x) = prod( (x - x_k)/(x_i - x_k) | k != i)
    // here we always evaluate with x = 0
    let mut sum = F::ZERO;
    for (&xi, yi) in ctx.ids.iter().zip(shares) {
        // let xi = share.x;

        let mut prod = F::ONE;
        for &xk in ctx.ids.iter() {
            if xk != xi {
                prod *= -xk * (xi - xk).invert().unwrap_or(F::ZERO)
            }
        }
        sum += yi.y * prod;
    }
    sum
}

impl<F: Field> std::ops::Add for Share<F> {
    type Output = Self;

    /// Add two shares together.
    /// Note: These must share the same `x` value.
    ///
    /// * `rhs`: the other share
    fn add(self, rhs: Self) -> Self::Output {
        // assert_eq!(self.x, rhs.x);
        Self {
            // x: self.x,
            y: self.y + rhs.y,
        }
    }
}

impl<F: Field> std::ops::AddAssign for Share<F> {
    fn add_assign(&mut self, rhs: Self) {
        // assert_eq!(self.x, rhs.x);
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
        // assert_eq!(self.x, rhs.x);
        Self {
            // x: self.x,
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
            // x: self.x,
            y: self.y + rhs,
        }
    }
}

impl<F: Field> std::iter::Sum for Share<F> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let Some(Share { y }) = iter.next() else {
            return Share {
                // x: F::ZERO,
                y: F::ZERO,
            };
        };
        let mut acc = y;
        for share in iter {
            // assert_eq!(share.x, x);
            acc += share.y;
        }
        Share { y: acc }
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

/// A share with an inflated degree.
///
/// This share have been constructed by multiplying two shares togehter,
/// each corresponding to a polynomial `p1`, `p2`, producing a share corresponding
/// to a polynomial `p3` with the degree `|p3| = |p1| + |p2|`.
/// Most regularly a share with degree double the previous one.
///
/// You can recover the internal share and act if nothing happened.
/// However, this will encur the penalty of having a larger degree.
///
/// ```ignore
/// # use caring::schemes::shamir::Share;
/// # use caring::element::Element32;
/// # use caring::schemes::shamir::inflatedShare;
/// # let a = Share{x: Element32::from(2u32), y: Element32::from(2u32)};
/// # let b = a.clone();
/// let a : Share<_> = a; // degree t
/// let b : Share<_> = b; // degree t
/// let c : InflatedShare<_> = a * b; // degree 2t
/// let c : Share<_> = c.give_up(); // degree 2t
/// ```
///
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct InflatedShare<F: Field>(Share<F>);

impl<F: Field> InflatedShare<F> {
    /// Recover the internal share from an inflated share.
    ///
    /// This accepts the higher degree WITHOUT a reduction,
    /// thus the new share have a higher degree than the initial two.
    pub fn giveup(self) -> Share<F> {
        self.0
    }
}

impl<F: Field> std::ops::Mul for Share<F> {
    type Output = InflatedShare<F>;

    /// Multiply a share with itself, producing an "inflated" share.
    ///
    /// * `rhs`: share to multiply with
    ///
    /// The `inflatedShare` corresponds to a polynomial with double the degree
    /// (if the two inital shares have the same degree)
    ///
    /// This "works" if the amount of shares is greater than the new degree,
    /// however this creates a limit on the amount of multiplications possible.
    ///
    /// To mitigate this you can use the `reduction` protocol.
    /// TODO: Construct that
    ///
    fn mul(self, rhs: Self) -> Self::Output {
        // assert_eq!(self.x, rhs.x);
        InflatedShare(Self {
            // x: self.x,
            y: self.y * rhs.y,
        })
    }
}

/// Reduction of the associated polynomial on share.
/// * `ctx` Shamir parameters
/// * `z`: inflatedShare to recover
/// * `net`: network functionality
/// * `rng`: Random number generator
///
/// ```ignore
/// # use caring::schemes::shamir::Share;
/// # use caring::element::Element32;
/// # use caring::schemes::shamir::inflatedShare;
/// # let a = Share{x: Element32::from(2u32), y: Element32::from(2u32)};
/// # let b = a.clone();
/// let a : Share<_> = a; // degree t
/// let b : Share<_> = b; // degree t
/// let c : InflatedShare<_> = a * b; // degree 2t
/// let c : Share<_> = deflate(ctx, c, net, rng);
/// //  ^-- degree t
/// ```
///
#[tracing::instrument(skip_all)]
pub async fn deflate<F: Field + serde::Serialize + serde::de::DeserializeOwned, U: Unicast>(
    ctx: &ShamirParams<F>,
    z: InflatedShare<F>,
    net: &mut U,
    rng: &mut impl RngCore,
) -> Result<Share<F>, <U as Unicast>::Error> {
    let z = z.0;
    // let x = z.x;
    let n = ctx.ids.len();
    tracing::info!(threshold = ctx.threshold, party_size = n,);
    // Consider if this should be an error instead.
    assert!(
        n >= 2 * ctx.threshold as usize,
        "Threshold larger than the player count!"
    );
    // We need 2t < n, otherwise we cannot reconstruct,
    // however 't' is hidden from before, so we just have to assume it is.
    // Now we need to reduce the polynomial back to t

    let random_poly = share(F::ZERO, &ctx.ids, ctx.threshold * 2, rng);
    let randomness = net.symmetric_unicast(random_poly).await?;
    let mut y = z.y;
    for r in randomness {
        y += r.y;
    }

    // issued subshares
    let subshares = share(y, &ctx.ids, ctx.threshold, rng); // share -> subshares
    let subshares = net.symmetric_unicast::<_>(subshares).await?;

    // reduction (cache?)
    // This should be equivalent to during a recombination
    let coeffs = lagrange_coefficients(&ctx.ids, F::ZERO);

    // inner product
    let y: F = subshares
        .into_iter()
        .zip(coeffs)
        .map(|(a, b)| a.y * b)
        .sum();

    Ok(Share { y })
}

/// A secret shared vector
///
/// * `x`: the id
/// * `ys`: share values
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct VecShare<F: Field> {
    //pub(crate) x: F,
    pub(crate) ys: Vector<F>,
}

impl<F: Field> std::ops::Add for VecShare<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // debug_assert_eq!(self.x, rhs.x);
        Self {
            // x: self.x,
            ys: self.ys + rhs.ys,
        }
    }
}

impl<F: Field> std::ops::Add<&Self> for VecShare<F> {
    type Output = VecShare<F>;

    fn add(self, rhs: &Self) -> Self::Output {
        // debug_assert_eq!(self.x, rhs.x);
        let a = self.ys;
        let b = &rhs.ys;
        let ys: Vector<_> = a + b;
        VecShare { ys }
    }
}

impl<F: Field> From<Vec<Share<F>>> for VecShare<F> {
    fn from(value: Vec<Share<F>>) -> Self {
        // let x = value[0].x;
        let ys = value.into_iter().map(|Share { y }| y).collect();
        VecShare { ys }
    }
}

impl<F: Field> From<VecShare<F>> for Vec<Share<F>> {
    fn from(value: VecShare<F>) -> Self {
        let VecShare { ys } = value;
        ys.into_iter().map(|y| Share { y }).collect()
    }
}

impl<F: Field> std::ops::AddAssign for VecShare<F> {
    fn add_assign(&mut self, rhs: VecShare<F>) {
        self.ys += &rhs.ys;
    }
}

impl<F: Field> std::ops::AddAssign<&VecShare<F>> for VecShare<F> {
    fn add_assign(&mut self, rhs: &VecShare<F>) {
        self.ys += &rhs.ys;
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
        shares.push(VecShare { ys: vecshare })
    }

    shares
}

use rayon::prelude::*;

/// Reconstruct or open shares
///
/// * `shares`: shares to be combined into open values
pub fn reconstruct_many<F: Field>(
    ctx: &ShamirParams<F>,
    shares: &[impl Borrow<VecShare<F>>],
) -> Vec<F> {
    // FIX: Code duplication with 'reconstruction'
    //
    // Lagrange interpolation:
    // L(x) = sum( y_i * l_i(x) )
    // where l_i(x) = prod( (x - x_k)/(x_i - x_k) | k != i)
    // here we always evaluate with x = 0
    let m = shares[0].borrow().ys.len();
    let mut sum = vec![F::ZERO; m];
    for (&xi, share) in ctx.ids.iter().zip(shares) {
        let yi: &Vector<_> = &share.borrow().ys;

        let mut prod = F::ONE;
        for &xk in &ctx.ids {
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
        let ids: Vec<_> = (1..=5u32).map(Element32::from).collect();
        let ctx = ShamirParams { threshold: 4, ids };
        // We test that we can secret-share a number and reconstruct it.
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        let v = Element32::from(42u32);
        let shares = share(v, &ctx.ids, 4, &mut rng);
        let v: u32 = reconstruct(&ctx, &shares).into();
        assert_eq!(v, 42u32);
    }

    #[test]
    fn addition() {
        // We test that we can secret-share two numbers and add them.
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let a = 3;
        let b = 7;
        let ids: Vec<_> = PARTIES.map(Element32::from).collect();
        let vs1 = {
            let v = Element32::from(a);
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b);
            share(v, &ids, 4, &mut rng)
        };

        let ctx = ShamirParams { threshold: 2, ids };

        // MPC
        let shares: Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a, &b)| a + b).collect();
        let v = reconstruct(&ctx, &shares);
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
        let ids: Vec<_> = PARTIES.map(Element32::from).collect();
        let vs1 = {
            let v: Vec<_> = a.clone().into_iter().map(Element32::from).collect();
            share_many(&v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v: Vec<_> = b.clone().into_iter().map(Element32::from).collect();
            share_many(&v, &ids, 4, &mut rng)
        };

        let ctx = ShamirParams { threshold: 2, ids };

        // MPC
        let shares: Vec<_> = vs1
            .iter()
            .zip(vs2.iter())
            .map(|(a, b)| a.clone() + b)
            .collect();
        let v = reconstruct_many(&ctx, &shares);
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
        let ids: Vec<_> = PARTIES.map(Element32::from).collect();
        let ctx = ShamirParams {
            threshold: 2,
            ids: ids.clone(),
        };

        let vs1 = {
            let v = Element32::from(a.to_bits() as u64);
            dbg!(&v);
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b.to_bits() as u64);
            dbg!(&v);
            share(v, &ids, 4, &mut rng)
        };

        // MPC
        let shares: Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a, &b)| a + b).collect();
        let v = reconstruct(&ctx, &shares);
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
                let [a, b, ..] = shares[..] else {
                    panic!("Can't multiply with only one share")
                };

                dbg!(&a);
                dbg!(&b);

                // mpc
                let c = a * b;
                dbg!(&c);
                // HACK: It doesn't work yet.
                //
                let ctx = ShamirParams { threshold, ids };
                let c = deflate(&ctx, c, &mut network, &mut rng)
                    .await
                    .expect("reducto failed");
                //let c = c.giveup();
                dbg!(&c);

                // opening
                let shares = network.symmetric_broadcast(c).await.unwrap();
                let c: u32 = reconstruct(&ctx, &shares).into();

                assert_eq!(c, 25);
            })
            .await
            .unwrap();
    }
}
