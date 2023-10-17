//! This is vanilla Shamir Secret Sharing using an arbitrary field F.
//! See https://en.wikipedia.org/wiki/Shamir%27s_secret_sharing
use ff::{derive::rand_core::RngCore, Field};
use itertools::multiunzip;

use crate::{poly::Polynomial, agency::Broadcast};

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

#[derive(Clone)]
pub struct BeaverTriple<F: Field> (Share<F>, Share<F>, Share<F>);

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
        itertools::izip!(a,b,c).map(|(a,b,c)| Self(a,b,c))
            .collect()
    }
}

/// Perform multiplication using beaver triples
///
/// * `x`: first share to multiply
/// * `y`: second share to multiply
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn multiply<
    F: Field + serde::Serialize + serde::de::DeserializeOwned,
> (
    x: Share<F>,
    y: Share<F>,
    triple: BeaverTriple<F>,
    network: &mut impl Broadcast
) -> Share<F> {
    let BeaverTriple(a,b,c) = triple;
    let ax = a+x;
    let by = b+y;

    // Sending both at once it more efficient.
    let resp = network.symmetric_broadcast((ax, by)).await;
    let (ax, by) : (Vec<_>, Vec<_>) = multiunzip(resp);

    let ax = reconstruct(&ax);
    let by = reconstruct(&by);

    y*ax + a*(-by) + c
}

// TODO: Multiplication in some form

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

#[cfg(test)]
mod test {
    use crate::{element::Element32, connection::{Network, InMemoryNetwork}};

    use super::*;

    #[test]
    fn simple() {
        // We test that we can secret-share a number and reconstruct it.
        let mut rng = rand::thread_rng();
        let v = Element32::from(42u32);
        let ids: Vec<_> = (1..=5u32).map(Element32::from).collect();
        let shares = share(v, &ids, 4, &mut rng);
        let v = reconstruct(&shares);
        assert_eq!(v, Element32::from(42u32));
    }

    #[test]
    fn addition() {
        // We test that we can secret-share a two numbers and add them.
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        let a = 3;
        let b = 7;
        let vs1 = {
            let v = Element32::from(a);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };

        // MPC
        let shares: Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a, &b)| a + b).collect();
        let v = reconstruct(&shares);
        let v: u32 = v.into();
        assert_eq!(v, a + b);
    }

    use fixed::FixedU32;
    use rand::thread_rng;

    #[test]
    fn addition_fixpoint() {
        // We test that we can secret-share a two *fixed point* numbers and add them.
        const PARTIES: std::ops::Range<u32> = 1..5u32;
        type Fix = FixedU32<16>;
        let a = 1.0;
        let b = 3.0;
        let a = Fix::from_num(a);
        let b = Fix::from_num(b);

        let vs1 = {
            let v = Element32::from(a.to_bits() as u64);
            dbg!(&v);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b.to_bits() as u64);
            dbg!(&v);
            let mut rng = rand::thread_rng();
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
        let mut rng = thread_rng();
        let parties: Vec<Element32> = (0..2u32).map(|i| i+1)
            .map(Element32::from).collect();

        let treshold: u64 = 2;

        // preproccessing
        let triples = BeaverTriple::fake(&parties, treshold, &mut rng);

        let mut taskset = tokio::task::JoinSet::new();
        // MPC
        let cluster = InMemoryNetwork::in_memory(parties.len());
        for network in cluster {
            let parties = parties.clone();
            let triple = triples[network.index].clone();
            taskset.spawn({async move {
                let mut network = network;
                let v = Element32::from(5u32);
                let shares = share(v, &parties, treshold, &mut thread_rng());
                let shares = network.symmetric_unicast(shares).await;
                let res = multiply(shares[0], shares[1], triple, &mut network).await;
                let res = network.symmetric_broadcast(res).await;
                let res = reconstruct(&res);
                let res : u32 = res.into();
                assert_eq!(res, 25);
            }});
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
}
