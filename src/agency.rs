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

use itertools::Itertools;
use thiserror::Error;

pub trait Broadcast {
    fn broadcast(&mut self, msg: &impl serde::Serialize);

    // TODO: Reconsider this
    #[allow(async_fn_in_trait)]
    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, impl std::error::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned;

    #[allow(async_fn_in_trait)]
    async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Result<Vec<T>, impl std::error::Error>;
}

pub trait Unicast {
    fn unicast(&mut self, msgs: &[impl serde::Serialize]);

    // TODO: Reconsider this
    #[allow(async_fn_in_trait)]
    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Result<Vec<T>, impl std::error::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned;

    #[allow(async_fn_in_trait)]
    async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Result<Vec<T>, impl std::error::Error>;
}

use digest::Digest;
// INFO: Reconsider if Broadcast should be a supertrait or just a type parameter
// There is also the question if we should overload the existing methods or provide
// new methods prefixed with 'verified' or something.
trait VerifiedBroadcast<D: Digest>: Broadcast {
    /// Ensure that a received broadcast is the same across all parties.
    async fn symmetric_broadcast<T: AsRef<[u8]>>(
        &mut self,
        msg: T,
    ) -> Result<Vec<T>, impl std::error::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        // TODO: Major clean-up here
        // TODO: Testing
        // PERF: We don't need to check ourselves.
        // We can also do some of the checks concurrently
        // TODO: Return a Result instead of panicking
        // TODO: Split this up into asymmetric functions
        // NOTE: Maybe send `msg` after the hash check?
        // I.e. do 3, then 1, 2.

        // 1. Send hash of the message
        let mut digest = D::new();
        digest.update(&msg);
        let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        let msg_hashes = Broadcast::symmetric_broadcast(self, hash).await?;

        // 3. Hash the hashes together and broadcast that
        let mut digest = D::new();
        for hash in msg_hashes.iter() {
            digest.update(hash);
        }
        let sum: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        let sum_all: Vec<Box<[u8]>> = Broadcast::symmetric_broadcast(self, sum).await?;
        let check = sum_all.iter().all_equal();
        if !check {
            // If some of the hashes are different, someone has gotten different results.
            return Err(BroadcastVerificationError::VerificationFailure);
        }

        // 2. Send the message and check that the hashes match
        let messages = Broadcast::symmetric_broadcast(self, msg).await?;
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

    fn broadcast(&mut self, _msg: &impl serde::Serialize) {
        todo!("Need to apply verification layer")
    }

    async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Vec<T> {
        todo!("Need to apply verification layer")
    }
}

#[derive(Error, Debug)]
pub enum BroadcastVerificationError {
    #[error("Could not verify broadcast")]
    VerificationFailure,
    #[error(transparent)]
    Other(Box<dyn std::error::Error>),
}
