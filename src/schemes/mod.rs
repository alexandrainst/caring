//! # Secret Sharing Schemes
//! This module contains different secret sharing schemes along with some common abstract functionality.
//! - Shamir (Passive Security, Addition, BeaverMult, InteractiveMult)
//! - Feldman (Active Security, Addition, BeaverMult)
//! - Pedersen (Active Security, Addition, BeaverMult)
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
//! Note, things hould be kept as I/O agnostic as possible,
//! and therefore should at most use the Broadcast/Unicast traits for
//! modelling interaction. However when possible, not having interaction
//! is preferable, but hard.

pub mod feldman;
pub mod pedersen;
pub mod rep3;
pub mod shamir;
pub mod spdz;

use std::{
    error::Error,
    future::Future,
    ops::{Add, Mul, Sub},
};

use rand::RngCore;

use crate::{
    algebra::{
        math::{RowMult, Vector},
        Length,
    },
    net::Communicate,
};

/// Currently unused trait, but might be a better way to represent that a share
/// can be multiplied by a const, however, it could also just be baked into 'Shared' directly.
pub trait MulByConst<A>:
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
    + Mul<Self::Value, Output = Self>
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Clone
    + Send
    + Sync
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
    fn share(ctx: &Self::Context, secret: Self::Value, rng: impl RngCore) -> Vec<Self>;

    /// Recombine the shares back into a secret,
    /// returning an value if successfull.
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `shares`: (secret-shared) shares to combine back
    fn recombine(ctx: &Self::Context, shares: &[Self]) -> Option<Self::Value>;
    // TODO: Should be Result<F, impl Error> with some generic Secret-sharing error

    // These vecs of vecs are pretty annoying
    fn share_many_naive(
        ctx: &Self::Context,
        secrets: &[Self::Value],
        mut rng: impl RngCore,
    ) -> Vec<Vec<Self>> {
        let shares: Vec<_> = secrets
            .iter()
            .map(|secret| Self::share(ctx, secret.clone(), &mut rng))
            .collect();
        crate::help::transpose(shares)
    }

    /// Naively recombine several (different) shares back into multiple secrets,
    /// Return an option for each successfull recombination
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `many_shares`: shares by each party to be recombined.
    fn recombine_many_naive(
        ctx: &Self::Context,
        many_shares: &[impl AsRef<[Self]>],
    ) -> Vec<Option<Self::Value>> {
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

pub trait SharedMany: Shared {
    type Vectorized: Sized
        + FromIterator<Self>
        + for<'a> Add<&'a Self::Vectorized, Output = Self::Vectorized>
        + for<'a> Sub<&'a Self::Vectorized, Output = Self::Vectorized>
        + RowMult<Self::Value>
        + Length
        + serde::Serialize
        + serde::de::DeserializeOwned
        + Clone
        + Send
        + Sync;

    /// Perform secret sharing splitting many `secret`s into a number of vectorized shares.
    ///
    /// * `ctx`: scheme and instance specific context required to perform secret sharing
    /// * `secrets`: secret values to share
    /// * `rng`: cryptographic secure random number generator
    ///
    fn share_many(
        ctx: &Self::Context,
        secrets: &[Self::Value],
        rng: impl RngCore,
    ) -> Vec<Self::Vectorized>;

    /// Recombine several (different) shares back into multiple secrets,
    /// Return an option for each successfull recombination
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `many_shares`: shares by each party to be recombined.
    fn recombine_many(
        ctx: &Self::Context,
        many_shares: &[Self::Vectorized],
    ) -> Option<Vector<Self::Value>>;
}

pub mod interactive {
    use std::ops::Mul;

    use thiserror::Error;
    use tracing::instrument;

    use crate::{
        algebra::math::Vector,
        net::{Communicate, Id, Tuneable},
    };

    #[derive(Debug, Error)]
    #[error("Communication failure: {0}")]
    pub struct CommunicationError(Box<dyn Error + Send>);
    impl CommunicationError {
        fn new(e: impl Error + Send + 'static) -> Self {
            Self(Box::new(e))
        }
    }

    #[derive(Debug, Error)]
    pub enum SharingError {
        #[error(transparent)]
        Communication(#[from] CommunicationError),
        #[error("Failure reconstruction")]
        Reconstruction(),
    }

    use super::*;
    impl<S, V, Ctx> InteractiveShared for S
    where
        S: Shared<Value = V, Context = Ctx> + Send,
        V: Send + Clone,
        Ctx: Send + Sync + Clone,
    {
        type Context = S::Context;
        type Value = V;
        type Error = SharingError;

        async fn share(
            ctx: &mut Self::Context,
            secret: Self::Value,
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Self, Self::Error> {
            let mut shares = S::share(ctx, secret, rng);
            let my_share = shares.remove(coms.id().0);
            coms.unicast(&shares)
                .await
                .map_err(CommunicationError::new)?;
            Ok(my_share)
        }

        async fn recombine(
            ctx: &mut Self::Context,
            secret: Self,
            mut coms: impl Communicate,
        ) -> Result<V, Self::Error> {
            let shares = coms
                .symmetric_broadcast(secret)
                .await
                .map_err(CommunicationError::new)?;
            Shared::recombine(&*ctx, &shares).ok_or(SharingError::Reconstruction())
        }

        async fn symmetric_share(
            ctx: &mut Self::Context,
            secret: Self::Value,
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Vec<Self>, Self::Error> {
            let shares = S::share(ctx, secret, rng);
            let shared = coms
                .symmetric_unicast(shares)
                .await
                .map_err(CommunicationError::new)?;
            Ok(shared)
        }

        async fn receive_share(
            _ctx: &mut Self::Context,
            mut coms: impl Communicate,
            from: Id,
        ) -> Result<Self, Self::Error> {
            let s = Tuneable::recv_from(&mut coms, from).await;
            let s = s.map_err(CommunicationError::new)?;
            Ok(s)
        }
    }

    pub trait InteractiveShared:
        Sized
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Self::Value, Output = Self>
        + serde::Serialize
        + serde::de::DeserializeOwned
        + Clone
        + Sync
    {
        type Context: Sync + Send;
        type Value: Clone + Send;
        type Error: Send + Sized + Error + 'static;

        fn share(
            ctx: &mut Self::Context,
            secret: Self::Value,
            rng: impl RngCore + Send,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Self, Self::Error>> + Send;

        fn symmetric_share(
            ctx: &mut Self::Context,
            secret: Self::Value,
            rng: impl RngCore + Send,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Vec<Self>, Self::Error>> + Send;

        fn receive_share(
            ctx: &mut Self::Context,
            coms: impl Communicate,
            from: Id,
        ) -> impl std::future::Future<Output = Result<Self, Self::Error>> + Send;

        fn recombine(
            ctx: &mut Self::Context,
            secrets: Self,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Self::Value, Self::Error>> + Send;
    }

    pub trait InteractiveSharedMany: InteractiveShared {
        type VectorShare: Sized
            + FromIterator<Self>
            + for<'a> Add<&'a Self::VectorShare, Output = Self::VectorShare>
            + for<'a> Sub<&'a Self::VectorShare, Output = Self::VectorShare>
            + RowMult<Self::Value>
            + Length
            + serde::Serialize
            + serde::de::DeserializeOwned
            + Clone
            + Send
            + Sync;

        fn share_many(
            ctx: &mut Self::Context,
            secrets: &[Self::Value],
            rng: impl RngCore + Send,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Self::VectorShare, Self::Error>> + Send;

        fn symmetric_share_many(
            ctx: &mut Self::Context,
            secrets: &[Self::Value],
            rng: impl RngCore + Send,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Vec<Self::VectorShare>, Self::Error>> + Send;

        fn receive_share_many(
            ctx: &mut Self::Context,
            coms: impl Communicate,
            from: Id,
        ) -> impl std::future::Future<Output = Result<Self::VectorShare, Self::Error>> + Send;

        fn recombine_many(
            ctx: &mut Self::Context,
            secrets: Self::VectorShare,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Vector<Self::Value>, Self::Error>> + Send;
    }

    impl<S, V, Ctx> InteractiveSharedMany for S
    where
        S: InteractiveShared<Error = SharingError, Value = V, Context = Ctx>
            + SharedMany<Value = V, Context = Ctx>
            + Send,
        V: Send + Sync + Clone,
        Ctx: Send + Sync,
    {
        type VectorShare = S::Vectorized;

        #[instrument(skip_all)]
        async fn share_many(
            ctx: &mut Self::Context,
            secrets: &[Self::Value],
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Self::VectorShare, Self::Error> {
            let shares = S::share_many(&*ctx, secrets, rng);
            let my_share = shares[coms.id().0].clone();
            coms.unicast(&shares)
                .await
                .map_err(CommunicationError::new)?;
            Ok(my_share)
        }

        #[instrument(skip_all)]
        async fn symmetric_share_many(
            ctx: &mut Self::Context,
            secrets: &[Self::Value],
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Vec<Self::VectorShare>, Self::Error> {
            let shares: Vec<S::Vectorized> = S::share_many(&*ctx, secrets, rng);
            let shared = coms
                .symmetric_unicast(shares)
                .await
                .map_err(CommunicationError::new)?;
            Ok(shared)
        }

        #[instrument(skip_all)]
        async fn receive_share_many(
            _ctx: &mut Self::Context,
            mut coms: impl Communicate,
            from: Id,
        ) -> Result<Self::VectorShare, Self::Error> {
            let s = Tuneable::recv_from(&mut coms, from).await;
            let s = s.map_err(CommunicationError::new)?;
            Ok(s)
        }

        #[instrument(skip_all)]
        async fn recombine_many(
            ctx: &mut Self::Context,
            secrets: Self::VectorShare,
            mut coms: impl Communicate,
        ) -> Result<Vector<Self::Value>, Self::Error> {
            let shares: Vec<S::Vectorized> = coms
                .symmetric_broadcast(secrets)
                .await
                .map_err(CommunicationError::new)?;
            S::recombine_many(&*ctx, &shares).ok_or(SharingError::Reconstruction())
        }
    }

    // NOTE: Not used currently.
    //
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
        fn interactive_mult<U: Communicate>(
            ctx: &Self::Context,
            net: &mut U,
            a: Self,
            b: Self,
        ) -> impl Future<Output = Result<Self, Box<dyn Error>>>;
    }
}

/// Reserve part of a context for performing
/// concurrent operations in the new subcontext.
pub trait Reserve {
    /// Reserve `amount` operations in resources.
    fn reserve(&mut self, amount: usize) -> Self;

    /// Put back unused resources.
    fn put_back(&mut self, other: Self);
}

// NOTE: Unused.
pub trait Verify: Sized {
    type Args: Send;

    fn verify(&self, coms: impl Communicate, args: Self::Args)
        -> impl Future<Output = bool> + Send;

    fn verify_many(
        batch: &[Self],
        coms: impl Communicate,
        args: Self::Args,
    ) -> impl Future<Output = Vec<bool>> + Send;
}
