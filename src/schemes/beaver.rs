use std::{
    marker::PhantomData,
    ops::{self, Add, Mul}, error::Error, convert::Infallible,
};

use ff::Field;
use group::Group;
use rand::RngCore;

use crate::{
    agency::Broadcast,
    schemes::{feldman, shamir},
};

pub trait Shared<F>:
    Sized + Add<Output = Self> + serde::Serialize + serde::de::DeserializeOwned
    // TODO: Add multiply-by-constant
{
    type Context;

    fn share(ctx: &mut Self::Context, secret: F) -> Vec<Self>;
    fn recombine(ctx: &mut Self::Context, shares: &[Self]) -> Option<F>;
    // TODO: Should be Result<F, impl Error>
}

pub struct ShamirParams<F> {
    threshold: u64,
    ids: Vec<F>,
    rng: Box<dyn RngCore>,
}

impl<F: Field + serde::Serialize + serde::de::DeserializeOwned> Shared<F> for shamir::Share<F> 
{
    type Context = ShamirParams<F>;

    fn share(ctx: &mut Self::Context, secret: F) -> Vec<Self> {
        shamir::share(secret, &ctx.ids, ctx.threshold, &mut ctx.rng)
    }

    fn recombine(_ctx: &mut Self::Context, shares: &[Self]) -> Option<F> {
        Some(shamir::reconstruct(shares))
    }
}

impl<
        F: ff::Field + serde::Serialize + serde::de::DeserializeOwned,
        G: Group + serde::Serialize + serde::de::DeserializeOwned + std::ops::Mul<F, Output = G>
    > Shared<F> for feldman::VerifiableShare<F, G>
{
    type Context = ShamirParams<F>;

    fn share(ctx: &mut Self::Context, secret: F) -> Vec<Self> {
        feldman::share::<F,G>(secret, &ctx.ids, ctx.threshold, &mut ctx.rng)
    }

    fn recombine(_ctx: &mut Self::Context, shares: &[Self]) -> Option<F> {
        feldman::reconstruct::<F,G>(shares)
    }

}

#[derive(Clone)]
pub struct BeaverTriple<F, S: Shared<F>> {
    phantom: PhantomData<F>,
    shares: (S, S, S),
}

impl<F: Field, C, S: Shared<F, Context = C>> BeaverTriple<F, S> {
    /// Fake a set of beaver triples.
    ///
    /// This produces `n` shares corresponding to a shared beaver triple,
    /// however locally with the values known.
    ///
    /// * `ids`: ids to produce for
    /// * `threshold`: threshold to reconstruct
    /// * `rng`: rng to sample from
    pub fn fake(ctx: &mut C, mut rng: &mut impl RngCore) -> Vec<Self> {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c: F = a * b;
        // Share (preproccess)
        let a = S::share(ctx, a);
        let b = S::share(ctx, b);
        let c = S::share(ctx, c);
        itertools::izip!(a, b, c)
            .map(|(a, b, c)| Self {
                shares: (a, b, c),
                phantom: PhantomData,
            })
            .collect()
    }
}

/// Perform multiplication using beaver triples
///
/// * `x`: first share to multiply
/// * `y`: second share to multiply
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn beaver_multiply<
    C,
    S: Shared<F, Context=C> + Copy + std::ops::Mul<F, Output=S>,
    F: Field + serde::Serialize + serde::de::DeserializeOwned,
    E,
>(
    ctx: &mut C,
    x: S,
    y: S,
    triple: BeaverTriple<F, S>,
    network: &mut impl Broadcast,
) -> Option<S> {
    let BeaverTriple {
        shares: (a, b, c),
        phantom: _,
    } = triple;
    let ax: S = a + x;
    let by: S = b + y;

    // Sending both at once it more efficient.
    let resp = network.symmetric_broadcast::<_>((ax, by)).await.unwrap();
    let (ax, by): (Vec<_>, Vec<_>) = itertools::multiunzip(resp);

    let ax = S::recombine(ctx, &ax)?;
    let by = S::recombine(ctx, &by)?;

    Some(y * ax + a * (-by) + c)
}
