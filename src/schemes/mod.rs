//! # Secret Sharing Schemes
//! This module contains different secret sharing schemes along with some common abstract functionality.
//! - Shamir (Passive Security, Addition)
//! - Feldman (Active Security, Addition)
//! - SPDZ (TODO)
//! - SPDZ2k (TODO)
//!
//! Most of these schemes are generic over a finite field or ring `F`, which they work in.
//! As such it is possible to use any kind of finite field for them.
//!
//! As most of the schemes have different properties and settings, we can't abstract over them
//! as easily. However an initial stepping stone is the `Shared<F>` trait, which represent a
//! secret 'Share' for `F`. As such by providing a context/parameters to the scheme,
//! one can perform arbitrary sharing/reconstruction.
//!
//! Other relevant trait suggestions for the future:
//! - `ThresholdShared`
//! - `Mult`
//! - `Add`
//! - `Reveal`
//! - `BeaverMult`
//!
//! Note, things hould be kept as I/O agnostic as possible,
//! and therefore should at most use the Broadcast/Unicast traits for
//! modelling interaction. However when possible, not having interaction
//! is preferable, but hard.

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
/// The secret-shared value needs to support addition and serialization.
/// This is used to implement generic MPC based schemes and protocols,
/// such as beaver triple multiplication.
pub trait Shared<F>:
    Sized + Add<Output = Self> + serde::Serialize + serde::de::DeserializeOwned
    // TODO: Add multiply-by-constant
    // NOTE: Maybe remove addition since we could have secret-sharing schemes that don't support
    // it, but just support sharing and reconstruction.
{
    /// The context needed to use the scheme.
    /// This can be a struct containing the threshold, ids and other things.
    type Context : Send + Clone;


    /// Perform secret sharing splitting `secret` into a number of shares.
    ///
    /// * `ctx`: scheme and instance specific context required to perform secret sharing
    /// * `secret`: secret value to share
    /// * `rng`: cryptographic secure random number generator
    ///
    fn share(ctx: &Self::Context, secret: F, rng: &mut impl RngCore) -> Vec<Self>;


    /// Recombine the shares back into a secret,
    /// returning an value if successfull.
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `shares`: (secret-shared) shares to combine back
    fn recombine(ctx: &Self::Context, shares: &[Self]) -> Option<F>;
    // TODO: Should be Result<F, impl Error> with some generic Secret-sharing error
}

#[derive(Clone)]
pub struct ShamirParams<F> {
    threshold: u64,
    ids: Vec<F>,
}

impl<F: Field + serde::Serialize + serde::de::DeserializeOwned> Shared<F> for shamir::Share<F> 
{
    type Context = ShamirParams<F>;

    fn share(ctx: &Self::Context, secret: F, rng: &mut impl RngCore) -> Vec<Self> {
        shamir::share(secret, &ctx.ids, ctx.threshold, rng)
    }

    fn recombine(_ctx: &Self::Context, shares: &[Self]) -> Option<F> {
        Some(shamir::reconstruct(shares))
    }
}

impl<
        F: ff::Field + serde::Serialize + serde::de::DeserializeOwned,
        G: Group + serde::Serialize + serde::de::DeserializeOwned + std::ops::Mul<F, Output = G>
    > Shared<F> for feldman::VerifiableShare<F, G>
{
    type Context = ShamirParams<F>;

    fn share(ctx: &Self::Context, secret: F, rng: &mut impl RngCore) -> Vec<Self> {
        feldman::share::<F,G>(secret, &ctx.ids, ctx.threshold, rng)
    }

    fn recombine(_ctx: &Self::Context, shares: &[Self]) -> Option<F> {
        feldman::reconstruct::<F,G>(shares)
    }

}

