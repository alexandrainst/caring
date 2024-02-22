//! This module describes traits for 'communication' functionalities.
//! These are the broadcast: send a message to many
//! and unicast: send a message to each
//!
//! The function of these is to provide an interface over the
//! concrete `Network` struct, allowing a looser API.
//!
//! ...Mostly due to the way `Network` have been constructed,
//! thus carrying over the type parameters for the reader/writer.
//!
//! Although that allows us a rich number of possibilities with concrete stuff.
//! Especially if muxing becomes relevant.
//!
//! Never mind, the point being that we want to add things here that
//! more less abstracts to an ideal functionality.
//!
//! Other canditates for these could be other protocols, such as coin flips,
//! oblivios transfer.
//!
//! A *protocol suite* in this manner might implement several of these,
//! but in the same way it allows to build and construct protocols and suites
//! from subprotocols in a very elegant manner IMO.
//!

use std::{error::Error, marker::PhantomData};

use futures::Future;
use itertools::Itertools;

// NOTE: We should probably find a way to include drop-outs in the broadcasts, since threshold
// schemes will continue to function if we lose connections underway. Maybe this is just handled by
// the network? But that would require the ability to resume a protocol after handling the drop-out.
// Another method is just ignore drop-outs, and as such the network will never error out.
// Otherwise we could do something totally different, which is let the network just have a
// threshold, ignoring drop-outs until then, then returning errors.
//
// In the same manner we can let the network have re-try strategies and the like.
// It is probably better handled in that layer anyway.
//
// One could still be for the broadcast/unicast operations to have a `size` function
// which gives the current network size. However I am not sure if this will be relevant?

pub trait Broadcast {
    type Error: Error + 'static;

    /// Broadcast a message to all other parties.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msg`: Message to send
    fn broadcast(&mut self, msg: &impl serde::Serialize);

    /// Broadcast a message to all parties and await their messages
    /// Messages are ordered by their index.
    ///
    /// * `msg`: message to send and receive
    fn symmetric_broadcast<T>(
        &mut self,
        msg: T,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned;

    /// Receive a message for each party.
    ///
    /// Asymmetric, waiting
    ///
    /// Returns: A list sorted by the connections (skipping yourself)
    fn receive_all<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>;
}

pub trait Unicast {
    type Error: Error + 'static;

    /// Unicast messages to each party
    ///
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`, skipping your own index.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msgs`: Messages to send
    fn unicast(&mut self, msgs: &[impl serde::Serialize]);

    /// Unicast a message to each party and await their messages
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`.
    ///
    /// * `msg`: message to send and receive
    fn symmetric_unicast<T>(
        &mut self,
        msgs: Vec<T>,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned;

    /// Receive a message for each party.
    ///
    /// Asymmetric, waiting
    ///
    /// Returns: A list sorted by the connections (skipping yourself)
    fn receive_all<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>;
}

use digest::Digest;
use tracing::{event, Level};

use crate::net::Tuneable;
// INFO: Reconsider if Broadcast should be a supertrait or just a type parameter
// There is also the question if we should overload the existing methods or provide
// new methods prefixed with 'verified' or something.

pub struct VerifiedBroadcast<B: Broadcast + Tuneable, D: Digest>(B, PhantomData<D>);

impl<B: Broadcast + Tuneable, D: Digest> VerifiedBroadcast<B, D> {
    pub fn inner(self) -> B {
        self.0
    }

    pub fn new(broadcast: B) -> Self {
        Self(broadcast, PhantomData)
    }
    // }

    // impl<B: Broadcast, D: Digest> Broadcast for VerifiedBroadcast<B, D> {
    //     type Error = BroadcastVerificationError<B::Error>;

    /// Ensure that a received broadcast is the same across all parties.
    #[tracing::instrument(skip_all)]
    pub async fn symmetric_broadcast<T: AsRef<[u8]>>(
        &mut self,
        msg: T,
    ) -> Result<Vec<T>, BroadcastVerificationError<<B as Broadcast>::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        // TODO: Testing
        let inner = &mut self.0;

        // 1. Send hash of the message
        event!(Level::INFO, "Sending commit by hashed message");
        let mut digest = D::new();
        digest.update(&msg);
        let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        let msg_hashes = inner
            .symmetric_broadcast(hash)
            .await
            .map_err(BroadcastVerificationError::Other)?;

        // 3. Hash the hashes together and broadcast that
        event!(Level::INFO, "Broadcast sum of all commits");
        let mut digest = D::new();
        for hash in msg_hashes.iter() {
            digest.update(hash);
        }
        let sum: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        let sum_all: Vec<Box<[u8]>> = inner
            .symmetric_broadcast(sum)
            .await
            .map_err(BroadcastVerificationError::Other)?;

        let check = sum_all.iter().all_equal();
        if !check {
            event!(Level::ERROR, "Failed verifying commit sum");
            // If some of the hashes are different, someone has gotten different results.
            return Err(BroadcastVerificationError::VerificationFailure);
        }

        // 2. Send the message and check that the hashes match
        event!(Level::INFO, "Sending original message");
        let messages = inner
            .symmetric_broadcast(msg)
            .await
            .map_err(BroadcastVerificationError::Other)?;

        event!(Level::INFO, "Verifiying commitments");
        for (msg, hash) in messages.iter().zip(msg_hashes) {
            // PERF: Maybe just reset the digest?
            let mut digest = D::new();
            digest.update(msg);
            let res = digest.finalize().to_vec().into_boxed_slice();
            if res != hash {
                return Err(BroadcastVerificationError::VerificationFailure);
            }
        }
        // Finally, return the packets
        Ok(messages)
    }

    #[tracing::instrument(skip_all)]
    pub async fn broadcast<T>(&mut self, msg: &T)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + AsRef<[u8]>,
    {
        let inner = &mut self.0;

        event!(Level::INFO, "Sending commit by hashed message");
        let mut digest = D::new();
        digest.update(msg);
        let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        inner.broadcast(&hash);

        // unneeded. (but everyone else except you to do it)
        inner.symmetric_broadcast(hash).await.unwrap();

        inner.broadcast(msg);

        // hope!
        let _ = inner.symmetric_broadcast(true).await.unwrap();
    }

    #[tracing::instrument(skip_all)]
    pub async fn recv_from<T: serde::de::DeserializeOwned + AsRef<[u8]>>(
        &mut self, party: usize,
    ) -> Result<T, BroadcastVerificationError<<B as Tuneable>::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let inner = &mut self.0;
        let hash : Box<[u8]> = inner.recv_from(party).await.expect("Proper error handling");

        // todo; exclude the broadcaster.
        let all = inner.symmetric_broadcast(hash).await.unwrap();
        if !all.iter().all_equal() {
            return Err(BroadcastVerificationError::VerificationFailure);
        }

        let msg : T = inner.recv_from(party).await.expect("Proper error handling");
        let mut digest = D::new();
        digest.update(&msg);
        let new_hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();

        if new_hash != all[0] { // We could also just use 'hash'
            let _ = inner.symmetric_broadcast(false).await;
            Err(BroadcastVerificationError::VerificationFailure)
        } else {
            let checks = inner.symmetric_broadcast(true).await.unwrap();
            if checks.iter().any(|c| !c) {
                return Err(BroadcastVerificationError::VerificationFailure)
            }
            Ok(msg)
        }

    }
}

#[derive(thiserror::Error, Debug)]
pub enum BroadcastVerificationError<E> {
    #[error("Could not verify broadcast")]
    VerificationFailure,
    #[error(transparent)]
    Other(E),
}

mod test {
    use crate::net::{agency::{BroadcastVerificationError, Unicast}, Tuneable};
    #[allow(unused_imports)]
    use crate::net::{agency::VerifiedBroadcast, network::InMemoryNetwork};
    use digest::crypto_common::KeyInit;
    #[allow(unused_imports)]
    use itertools::Itertools;
    use sha2::Digest;

    #[tokio::test]
    async fn verified_broadcast() {
        let (n1, n2, n3) = InMemoryNetwork::in_memory(3)
            .drain(..3)
            .tuples()
            .next()
            .unwrap();

        let t1 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n1);
            let resp = vb
                .symmetric_broadcast(String::from("Hi from Alice"))
                .await
                .unwrap();
            assert_eq!(resp[0], "Hi from Alice");
            assert_eq!(resp[1], "Hi from Bob");
            assert_eq!(resp[2], "Hi from Charlie");
        };
        let t2 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n2);
            let resp = vb
                .symmetric_broadcast(String::from("Hi from Bob"))
                .await
                .unwrap();
            assert_eq!(resp[0], "Hi from Alice");
            assert_eq!(resp[1], "Hi from Bob");
            assert_eq!(resp[2], "Hi from Charlie");
        };
        let t3 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n3);
            let resp = vb
                .symmetric_broadcast(String::from("Hi from Charlie"))
                .await
                .unwrap();
            assert_eq!(resp[0], "Hi from Alice");
            assert_eq!(resp[1], "Hi from Bob");
            assert_eq!(resp[2], "Hi from Charlie");
        };

        futures::join!(t1, t2, t3);
    }

    #[tokio::test]
    async fn verified_broadcast_assym() {
        let (n1, n2, n3) = InMemoryNetwork::in_memory(3)
            .drain(..3)
            .tuples()
            .next()
            .unwrap();

        let t1 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n1);
            vb.broadcast(&String::from("Hello everyone!")).await;
        };

        let t2 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n2);
            let resp : String = vb.recv_from(0).await.unwrap();
            assert_eq!(resp, "Hello everyone!");
        };
        let t3 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n3);
            let resp : String = vb.recv_from(0).await.unwrap();
            assert_eq!(resp, "Hello everyone!");
        };

        futures::join!(t1, t2, t3);
    }


    #[tokio::test]
    async fn verified_broadcast_cheating_assym() {
        let (n1, n2, n3) = InMemoryNetwork::in_memory(3)
            .drain(..3)
            .tuples()
            .next()
            .unwrap();

        let t1 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n1);
            vb.recv_from::<String>(2).await
        };

        let t2 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n2);
            vb.recv_from::<String>(2).await
        };
        let t3 = async {
            let vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n3);
            let mut net = vb.inner();
            let s0 = String::from("Hi from cheating Charlie");
            let s1 = String::from("Hi from charming Charlie");
            // protocol emulation.
            let mut digest = sha2::Sha256::new();
            digest.update(&s1);
            let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
            net.broadcast(&hash);
            net.symmetric_broadcast(hash).await.unwrap();
            // fake broadcast
            net.send_to(0, &s0);
            net.send_to(1, &s1);
            let _ = net.symmetric_broadcast(true).await.unwrap();

        };

        let (res1, res2, _) = futures::join!(t1, t2, t3);
        assert!(res1.as_ref().is_err_and(|e| {
            matches!(e, BroadcastVerificationError::VerificationFailure)
        }), "Should be a verification failure, was: {res1:?}");
        assert!(res2.as_ref().is_err_and(|e| {
            matches!(e, BroadcastVerificationError::VerificationFailure)
        }), "Should be a verification failure, was: {res2:?}");
    }
}
