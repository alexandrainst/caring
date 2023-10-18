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

use tokio::io::{AsyncWrite, AsyncRead};

use crate::connection::Network;

pub trait Broadcast {
    fn broadcast(&mut self, msg: &impl serde::Serialize);

    // TODO: Reconsider this
    #[allow(async_fn_in_trait)]
    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned;

}

pub trait Unicast {
    fn unicast(&mut self, msgs: &[impl serde::Serialize]);

    // TODO: Reconsider this
    #[allow(async_fn_in_trait)]
    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned;
}



impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Unicast for Network<R,W> {
    fn unicast(&mut self, msgs: &[impl serde::Serialize]) {
        self.unicast(msgs)
    }

    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned {
        self.symmetric_unicast(msgs).await
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Broadcast for Network<R,W> {
    fn broadcast(&mut self, msg: &impl serde::Serialize) {
        self.broadcast(msg)
    }

    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned {
            self.symmetric_broadcast(msg).await
    }
}
