use std::{
    error::Error,
    ops::{Index, IndexMut},
};

use futures::Future;

use crate::net::{
    connection::{Connection, ConnectionError},
    network::Network,
};

pub mod agency;
pub mod connection;
pub mod network;

// TODO: Serde trait bounds on `T`
// TODO: Properly use this trait for other things (Connection/Agency etc.)
pub trait Channel {
    type Error: Error;

    /// Send a message over the channel
    ///
    /// * `msg`: message to serialize and send
    fn send<T: serde::Serialize>(&self, msg: &T) -> impl Future<Output = Result<(), Self::Error>>;

    fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, Self::Error>>;
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin,
        W: tokio::io::AsyncWrite + std::marker::Unpin,
    > Channel for Connection<R, W>
{
    type Error = ConnectionError;

    async fn send<T: serde::Serialize>(&self, _msg: &T) -> Result<(), Self::Error> {
        todo!()
    }

    fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, Self::Error>> {
        Connection::recv(self)
    }
}

/// Tune to a specific channel
pub trait Tuneable {
    type Error: Error;
    type SubChannel: Channel;

    fn id(&self) -> usize;

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::Error>>;

    fn send_to<T: serde::Serialize>(&self, idx: usize, msg: &T);
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin,
        W: tokio::io::AsyncWrite + std::marker::Unpin,
    > Tuneable for Network<R, W>
{
    type Error = ConnectionError;
    type SubChannel = Connection<R, W>;

    fn id(&self) -> usize {
        self.index
    }

    async fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> Result<T, Self::Error> {
        self[idx].recv().await
    }

    fn send_to<T: serde::Serialize>(&self, idx: usize, msg: &T) {
        self[idx].send(msg)
    }
}
