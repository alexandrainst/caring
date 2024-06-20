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

pub trait Broadcast: Send {
    type BroadcastError: Error + Send + 'static;
    // type Error: Error + 'static;

    /// Broadcast a message to all other parties.
    ///
    /// * `msg`: Message to send
    /// Returns: an error if there were problems broadcasting the message.
    fn broadcast(
        &mut self,
        msg: &(impl serde::Serialize + Sync),
    ) -> impl std::future::Future<Output = Result<(), Self::BroadcastError>> + Send;

    /// Broadcast a message to all parties and await their messages
    /// Messages are ordered by their index.
    ///
    /// This function is symmetric, and as such it is expected that all other parties
    /// call this function concurrently.
    ///
    /// * `msg`: message to send and receive
    fn symmetric_broadcast<T>(
        &mut self,
        msg: T,
    ) -> impl Future<Output = Result<Vec<T>, Self::BroadcastError>> + Send
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + Sync;

    /// Receive a message from a party
    ///
    /// Returns: a message from the given party or an error
    fn recv_from<T: serde::de::DeserializeOwned + Send>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::BroadcastError>> + Send;

    /// Size of the broadcasting network including yourself,
    /// as such there is n-1 outgoing connections
    fn size(&self) -> usize;
}

impl<'a, B: Broadcast> Broadcast for &'a mut B {
    type BroadcastError = B::BroadcastError;

    fn broadcast(
        &mut self,
        msg: &(impl serde::Serialize + Sync),
    ) -> impl std::future::Future<Output = Result<(), Self::BroadcastError>> + Send {
        (**self).broadcast(msg)
    }

    fn symmetric_broadcast<T>(
        &mut self,
        msg: T,
    ) -> impl Future<Output = Result<Vec<T>, Self::BroadcastError>> + Send
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + Sync,
    {
        (**self).symmetric_broadcast(msg)
    }

    fn recv_from<T: serde::de::DeserializeOwned + Send>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::BroadcastError>> + Send {
        (**self).recv_from(idx)
    }

    fn size(&self) -> usize {
        (**self).size()
    }
}

// TODO: Possible rename this trait as it's name is confusing.
pub trait Unicast {
    type UnicastError: Error + Send + 'static;

    /// Unicast messages to each party
    ///
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`, skipping your own index.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msgs`: Messages to send
    fn unicast(
        &mut self,
        msgs: &[impl serde::Serialize + Send + Sync],
    ) -> impl std::future::Future<Output = Result<(), Self::UnicastError>> + Send;

    /// Unicast a message to each party and await their messages
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`.
    ///
    /// * `msg`: message to send and receive
    fn symmetric_unicast<T>(
        &mut self,
        msgs: Vec<T>,
    ) -> impl Future<Output = Result<Vec<T>, Self::UnicastError>> + Send
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + Sync;

    /// Receive a message for each party.
    ///
    /// Asymmetric, waiting
    ///
    /// Returns: A list sorted by the connections (skipping yourself)
    fn receive_all<T: serde::de::DeserializeOwned + Send>(
        &mut self,
    ) -> impl Future<Output = Result<Vec<T>, Self::UnicastError>> + Send;

    /// Size of the unicasting network including yourself,
    /// as such there is n-1 outgoing connections
    fn size(&self) -> usize;
}

impl<'a, U: Unicast> Unicast for &'a mut U {
    type UnicastError = U::UnicastError;

    fn size(&self) -> usize {
        (**self).size()
    }

    fn receive_all<T: serde::de::DeserializeOwned + Send>(
        &mut self,
    ) -> impl Future<Output = Result<Vec<T>, Self::UnicastError>> + Send {
        (**self).receive_all()
    }

    fn unicast(
        &mut self,
        msgs: &[impl serde::Serialize + Send + Sync],
    ) -> impl std::future::Future<Output = Result<(), Self::UnicastError>> + Send {
        (**self).unicast(msgs)
    }

    fn symmetric_unicast<T>(
        &mut self,
        msgs: Vec<T>,
    ) -> impl Future<Output = Result<Vec<T>, Self::UnicastError>> + Send
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Send + Sync,
    {
        (**self).symmetric_unicast(msgs)
    }
}

use digest::Digest;
use tracing::{event, Level};

pub struct VerifiedBroadcast<B: Broadcast, D: Digest>(B, PhantomData<D>);

impl<B: Broadcast, D: Digest> VerifiedBroadcast<B, D> {
    pub fn inner(self) -> B {
        self.0
    }

    pub fn new(broadcast: B) -> Self {
        Self(broadcast, PhantomData)
    }

    #[tracing::instrument(skip_all)]
    pub async fn symmetric_broadcast<T>(
        &mut self,
        msg: T,
    ) -> Result<Vec<T>, BroadcastVerificationError<B::BroadcastError>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        // TODO: Testing
        let inner = &mut self.0;

        let msg = bincode::serialize(&msg).expect("Serialization failed.");

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
        for hash in &msg_hashes {
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
                event!(Level::ERROR, "Received object is not equal to the hash");
                return Err(BroadcastVerificationError::VerificationFailure);
            }
        }

        let messages: Result<_, _> = messages
            .into_iter()
            .map(|m| bincode::deserialize(&m))
            .collect();

        let messages = messages.unwrap();

        // Finally, return the packets
        Ok(messages)
    }

    #[tracing::instrument(skip_all)]
    pub async fn broadcast<T>(
        &mut self,
        msg: &T,
    ) -> Result<(), BroadcastVerificationError<B::BroadcastError>>
    where
        T: serde::Serialize,
    {
        let inner = &mut self.0;
        let msg = bincode::serialize(msg).unwrap();

        event!(Level::INFO, "Sending commit by hashed message");
        let mut digest = D::new();
        digest.update(&msg);
        let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        inner.broadcast(&hash).await.unwrap();

        // unneeded. (but everyone else except you to do it)
        event!(Level::INFO, "Broadcasting received hash");
        inner.symmetric_broadcast(hash).await.unwrap();

        event!(Level::INFO, "Broadcasting payload");
        inner.broadcast(&msg).await.unwrap();

        // hope!
        event!(Level::INFO, "Broadcasting agreement to the message");
        let _ = inner.symmetric_broadcast(true).await.unwrap();
        Ok(())
    }

    #[tracing::instrument(skip_all)]
    pub async fn recv_from<T>(
        &mut self,
        party: usize,
    ) -> Result<T, BroadcastVerificationError<B::BroadcastError>>
    where
        T: serde::de::DeserializeOwned,
    {
        let inner = &mut self.0;
        let hash: Box<[u8]> = inner
            .recv_from(party)
            .await
            .map_err(BroadcastVerificationError::Other)?;

        // todo; exclude the broadcaster.
        event!(Level::INFO, "Broadcasting received hash");
        let all = inner.symmetric_broadcast(hash).await.unwrap();
        if !all.iter().all_equal() {
            return Err(BroadcastVerificationError::VerificationFailure);
        }

        let msg: Vec<u8> = inner
            .recv_from(party)
            .await
            .map_err(BroadcastVerificationError::Other)?;
        let mut digest = D::new();
        digest.update(&msg);
        let new_hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();

        if new_hash != all[0] {
            // We could also just use 'hash'
            event!(Level::ERROR, "Received object is not equal to the hash");
            let _ = inner.symmetric_broadcast(false).await;
            Err(BroadcastVerificationError::VerificationFailure)
        } else {
            event!(Level::INFO, "Received message did match hash");
            let checks = inner
                .symmetric_broadcast(true)
                .await
                .expect("TODO: Need to convert this.");
            if checks.iter().any(|c| !c) {
                event!(Level::ERROR, "Disagreement about broadcasted message.");
                return Err(BroadcastVerificationError::VerificationFailure);
            }

            let buf = std::io::Cursor::new(msg);
            let msg = bincode::deserialize_from(buf)
                .map_err(|_| BroadcastVerificationError::Malformed)?;

            Ok(msg)
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum BroadcastVerificationError<E> {
    #[error("Could not verify broadcast")]
    VerificationFailure,
    #[error("Could not deserialize object")]
    Malformed,
    #[error(transparent)]
    Other(E),
}

impl<B: Broadcast, D: Digest + Send> Broadcast for VerifiedBroadcast<B, D> {
    type BroadcastError = BroadcastVerificationError<<B as Broadcast>::BroadcastError>;

    async fn broadcast(&mut self, msg: &impl serde::Serialize) -> Result<(), Self::BroadcastError> {
        self.broadcast(msg).await
    }

    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, Self::BroadcastError>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        self.symmetric_broadcast(msg).await
    }

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::BroadcastError>> + Send {
        self.recv_from(idx)
    }

    fn size(&self) -> usize {
        self.0.size()
    }
}

mod test {

    #[allow(unused_imports)]
    use super::*;

    #[allow(unused_imports)]
    use crate::net::Tuneable;
    #[allow(unused_imports)]
    use crate::net::{agency::VerifiedBroadcast, network::InMemoryNetwork};

    #[allow(unused_imports)]
    use itertools::Itertools;

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
            vb.broadcast(&String::from("Hello everyone!"))
                .await
                .unwrap();
        };

        let t2 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n2);
            let resp: String = vb.recv_from(0).await.unwrap();
            assert_eq!(resp, "Hello everyone!");
        };
        let t3 = async {
            let mut vb = VerifiedBroadcast::<_, sha2::Sha256>::new(n3);
            let resp: String = vb.recv_from(0).await.unwrap();
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
            use digest::Digest;
            let mut digest = sha2::Sha256::new();
            digest.update(&s1);
            let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
            net.broadcast(&hash).await.unwrap();
            net.symmetric_broadcast(hash).await.unwrap();
            // fake broadcast
            net.send_to(0, &s0).await.unwrap();
            net.send_to(1, &s1).await.unwrap();
            let _ = net.symmetric_broadcast(true).await.unwrap();
        };

        let (res1, res2, _) = futures::join!(t1, t2, t3);
        assert!(
            res1.as_ref()
                .is_err_and(|e| { matches!(e, BroadcastVerificationError::VerificationFailure) }),
            "Should be a verification failure, was: {res1:?}"
        );
        assert!(
            res2.as_ref()
                .is_err_and(|e| { matches!(e, BroadcastVerificationError::VerificationFailure) }),
            "Should be a verification failure, was: {res2:?}"
        );
    }
}
