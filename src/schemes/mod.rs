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
    ops::{Add, Sub},
};

use rand::RngCore;

use crate::net::Communicate;

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
    + serde::Serialize
    + serde::de::DeserializeOwned
    + Clone
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
    fn share_many(
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

    /// Recombine several (different) shares back into multiple secrets,
    /// Return an option for each successfull recombination
    ///
    /// * `ctx`: scheme and instance specific context
    /// * `many_shares`: shares by each party to be recombined.
    fn recombine_many(
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
    ) -> impl Future<Output = Result<Self, Box<dyn Error>>> + Send;
}

pub mod interactive {
    use itertools::Itertools;
    use thiserror::Error;

    use crate::{
        algebra::math::Vector,
        net::{Communicate, Tuneable},
    };

    #[derive(Debug, Error)]
    #[error("Communication failure: {0}")]
    pub struct CommunicationError(Box<dyn Error + Send>);
    impl CommunicationError {
        fn new(e: impl Error + Send + 'static) -> Self {
            Self(Box::new(e))
        }
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
        type Error = CommunicationError;

        async fn share(
            ctx: &mut Self::Context,
            secret: Self::Value,
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Self, Self::Error> {
            let shares = S::share(ctx, secret, rng);
            let my_share = shares[coms.id()].clone();
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
            Ok(Shared::recombine(&*ctx, &shares).unwrap())
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
            from: usize,
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
        + serde::Serialize
        + serde::de::DeserializeOwned
        + Clone
        + Sync
    {
        type Context: Sync + Send;
        type Value: Clone + Send;
        type Error: Send + Sized + Error + 'static;

        // Note: Not sure if ctx should be passed as a move or as a mutable reference.
        // Some schemes require bookkeeping (spdz) so we need the mutability.
        // Another method could be to swap Context and Share<_>,
        // however that would probably just result in the same issues.
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
            from: usize,
        ) -> impl std::future::Future<Output = Result<Self, Self::Error>> + Send;

        fn recombine(
            ctx: &mut Self::Context,
            secrets: Self,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Self::Value, Self::Error>> + Send;
    }

    pub trait InteractiveSharedMany: InteractiveShared {
        type VectorShare;

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
            from: usize,
        ) -> impl std::future::Future<Output = Result<Self::VectorShare, Self::Error>> + Send;

        fn recombine_many(
            ctx: &mut Self::Context,
            secrets: Self::VectorShare,
            coms: impl Communicate,
        ) -> impl std::future::Future<Output = Result<Vector<Self::Value>, Self::Error>> + Send;
    }

    // TODO: Consider using specialized SharedMany instead.
    impl<S, V, Ctx> InteractiveSharedMany for S
    where
        S: InteractiveShared<Error = CommunicationError, Value = V, Context = Ctx>
            + Shared<Value = V, Context = Ctx>
            + Send,
        V: Send + Clone,
    {
        type VectorShare = Vector<S>;

        async fn share_many(
            ctx: &mut Self::Context,
            secrets: &[Self::Value],
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Self::VectorShare, Self::Error> {
            let shares = S::share_many(&*ctx, secrets, rng);
            let my_share = shares[coms.id()].clone();
            coms.unicast(&shares)
                .await
                .map_err(CommunicationError::new)?;
            Ok(my_share.into())
        }

        async fn symmetric_share_many(
            ctx: &mut Self::Context,
            secrets: &[Self::Value],
            rng: impl RngCore + Send,
            mut coms: impl Communicate,
        ) -> Result<Vec<Self::VectorShare>, Self::Error> {
            let shares: Vec<Vector<Self>> = S::share_many(&*ctx, secrets, rng)
                .into_iter()
                .map(|v| v.into())
                .collect();
            let shared = coms
                .symmetric_unicast(shares)
                .await
                .map_err(CommunicationError::new)?;
            Ok(shared)
        }

        async fn receive_share_many(
            _ctx: &mut Self::Context,
            mut coms: impl Communicate,
            from: usize,
        ) -> Result<Self::VectorShare, Self::Error> {
            let s = Tuneable::recv_from(&mut coms, from).await;
            let s = s.map_err(CommunicationError::new)?;
            Ok(s)
        }

        async fn recombine_many(
            ctx: &mut Self::Context,
            secrets: Self::VectorShare,
            mut coms: impl Communicate,
        ) -> Result<Vector<Self::Value>, Self::Error> {
            let shares = coms
                .symmetric_broadcast(secrets)
                .await
                .map_err(CommunicationError::new)?;
            let res: Vector<Self::Value> = Shared::recombine_many(&*ctx, &shares)
                .into_iter()
                .map(|x| x.unwrap())
                .collect(); // TODO: Proper errors
            Ok(res)
        }
    }
}

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
