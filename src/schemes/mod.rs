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

pub mod feldman;
pub mod pedersen;
pub mod rep3;
pub mod shamir;
pub mod spdz;
pub mod spdz2k;

use std::{
    error::Error,
    ops::{Add, Sub},
};

use futures::Future;

use rand::RngCore;

use crate::net::{
    agency::{Broadcast, Unicast},
    Tuneable,
};

/// Currently unused trait, but might be a better way to represent that a share
/// can be multiplied by a const, however, it could also just be baked into 'Shared' directly.
trait MulByConst<A>:
    Shared<Value = A> + std::ops::Mul<A, Output = Self> + std::ops::MulAssign<A>
{
}

/// For a value of type `F` the value is secret-shared
///
/// The secret-shared value needs to support addition and serialization.
/// This is used to implement generic MPC based schemes and protocols,
/// such as beaver triple multiplication.
///
/// (Maybe rename to SecretShared?)
pub trait Shared:
    Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Clone
{
    /// The context needed to use the scheme.
    /// This can be a struct containing the threshold, ids and other things.
    type Context: Send + Clone;
    type Value: Clone;

    /// Perform secret sharing splitting `secret` into a number of shares.
    ///
    /// * `ctx`: scheme and instance specific context required to perform secret sharing
    /// * `secret`: secret value to share
    /// * `rng`: cryptographic secure random number generator
    ///
    fn share(ctx: &Self::Context, secret: Self::Value, rng: &mut impl RngCore) -> Vec<Self>;

    /// Recombine the shares back into a secret,
    /// returning an value if successfull.
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `shares`: (secret-shared) shares to combine back
    fn recombine(ctx: &Self::Context, shares: &[Self]) -> Option<Self::Value>;
    // TODO: Should be Result<F, impl Error> with some generic Secret-sharing error

    // These vecs of vecs are pretty annoying
    fn share_many(
        ctx: &Self::Context,
        secrets: &[Self::Value],
        rng: &mut impl RngCore,
    ) -> Vec<Vec<Self>> {
        let shares: Vec<_> = secrets
            .iter()
            .map(|secret| Self::share(ctx, secret.clone(), rng))
            .collect();
        crate::help::transpose(shares)
    }

    /// Recombine several (different) shares back into multiple secrets,
    /// Return an option for each successfull recombination
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `many_shares`: shares by each party to be recombined.
    fn recombine_many(ctx: &Self::Context, many_shares: &[impl AsRef<[Self]>]) -> Vec<Option<F>> {
        // This is ugly and a bit inefficient.
        let n = many_shares[0].as_ref().len();
        let m = many_shares.len();
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let mut buf = Vec::with_capacity(m);
            for party in many_shares {
                buf.push(party.as_ref()[i].clone());
            }
            let res = Self::recombine(ctx, &buf);
            output.push(res);
        }
        output
    }
}

pub trait SharedVec:
    Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Clone
{
    type Value;
    type Context: Send + Clone;

    fn share(ctx: &Self::Context, secrets: &[Self::Value], rng: &mut impl RngCore) -> Self;
    fn recombine(ctx: &Self::Context, shares: &[Self], rng: &mut impl RngCore) -> Vec<Self::Value>;
}

/// Support for multiplication of two shares for producing a share.
///
/// Note, that this is different to beaver multiplication as it does not require
/// triplets, however it does require a native multiplication protocol.
///
pub trait InteractiveMult: Shared {
    /// Perform interactive multiplication
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `net`: Unicasting network
    /// * `a`: first share to multiply
    /// * `b`: second share to multiply
    ///
    /// Returns a result which contains the shared value corresponding
    /// to the multiplication of `a` and `b`.
    fn interactive_mult<U: Unicast + Tuneable + Broadcast>(
        ctx: &Self::Context,
        net: &mut U,
        a: Self,
        b: Self,
    ) -> impl Future<Output = Result<Self, Box<dyn Error>>>;
}
