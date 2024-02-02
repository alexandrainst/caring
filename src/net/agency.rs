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
    type Error: Error;

    fn broadcast(&mut self, msg: &impl serde::Serialize);

    // TODO: Reconsider this
    fn symmetric_broadcast<T>(
        &mut self,
        msg: T,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned;

    fn receive_all<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>;
}

pub trait Unicast {
    type Error;

    fn unicast(&mut self, msgs: &[impl serde::Serialize]);

    // TODO: Reconsider this
    fn symmetric_unicast<T>(
        &mut self,
        msgs: Vec<T>,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned;

    fn receive_all<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<Vec<T>, Self::Error>>;
}

use digest::Digest;
// INFO: Reconsider if Broadcast should be a supertrait or just a type parameter
// There is also the question if we should overload the existing methods or provide
// new methods prefixed with 'verified' or something.

pub struct VerifiedBroadcast<B: Broadcast, D: Digest>(B, PhantomData<D>);

impl<B: Broadcast, D: Digest> VerifiedBroadcast<B,D> {

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
    pub async fn symmetric_broadcast<T: AsRef<[u8]>>(
        &mut self,
        msg: T,
    ) -> Result<Vec<T>, BroadcastVerificationError<B::Error>>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        // TODO: Testing
        let inner = &mut self.0;

        // 1. Send hash of the message
        let mut digest = D::new();
        digest.update(&msg);
        let hash: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        let msg_hashes = inner.symmetric_broadcast(hash)
            .await
            .map_err(BroadcastVerificationError::Other)?;

        // 3. Hash the hashes together and broadcast that
        let mut digest = D::new();
        for hash in msg_hashes.iter() {
            digest.update(hash);
        }
        let sum: Box<[u8]> = digest.finalize().to_vec().into_boxed_slice();
        let sum_all: Vec<Box<[u8]>> = inner.symmetric_broadcast(sum)
            .await
            .map_err(BroadcastVerificationError::Other)?;

        let check = sum_all.iter().all_equal();
        if !check {
            // If some of the hashes are different, someone has gotten different results.
            return Err(BroadcastVerificationError::VerificationFailure);
        }

        // 2. Send the message and check that the hashes match
        let messages = inner.symmetric_broadcast(msg)
            .await
            .map_err(BroadcastVerificationError::Other)?;

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


    pub fn broadcast(&mut self, _msg: &impl serde::Serialize) {
        todo!("Need to copy/translate/move implementation from symmetric")
    }

    pub async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Vec<T> {
        todo!("Need to apply verification layer")
    }
}





#[derive(thiserror::Error, Debug)]
pub enum BroadcastVerificationError<E> {
    #[error("Could not verify broadcast")]
    VerificationFailure,
    #[error(transparent)]
    Other(E),
}
