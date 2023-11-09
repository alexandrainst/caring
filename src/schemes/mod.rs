pub mod beaver;
pub mod feldman;
pub mod shamir;
pub mod spdz;
pub mod spdz2k;

use std::ops::Add;

use ff::Field;
use group::Group;
use rand::RngCore;



/// For a value of type `F` the value is secret-shared
///
/// The secret-shared value needs to support addition and serialization
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

