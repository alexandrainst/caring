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
pub mod shamir;
pub mod spdz;
pub mod spdz2k;

use std::{error::Error, ops::{Add, Sub}};

use ff::Field;
use futures::Future;
use group::Group;
use rand::{thread_rng, RngCore};

use crate::net::agency::Unicast;

trait MulByConst<A>: Shared<A> + std::ops::Mul<A, Output = Self> + std::ops::MulAssign<A> {}

/// For a value of type `F` the value is secret-shared
///
/// The secret-shared value needs to support addition and serialization.
/// This is used to implement generic MPC based schemes and protocols,
/// such as beaver triple multiplication.
pub trait Shared<F>:
    Sized + Add<Output = Self> + Sub<Output = Self> + serde::Serialize + serde::de::DeserializeOwned
{
    /// The context needed to use the scheme.
    /// This can be a struct containing the threshold, ids and other things.
    type Context: Send + Clone;

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

pub trait InteractiveMult<F> : Shared<F> {
    /// Perform interactive multiplication
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `net`: Unicasting network
    /// * `a`: first share to multiply
    /// * `b`: second share to multiply
    ///
    /// Returns a result which contains the shared value corresponding
    /// to the multiplication of `a` and `b`.
    fn interactive_mult<U: Unicast>(ctx: &Self::Context, net: &mut U, a: Self, b: Self) -> impl Future<Output = Result<Self, Box<dyn Error>>>;
}



// Move to shamir.rs


