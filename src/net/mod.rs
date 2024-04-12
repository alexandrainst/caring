use std::error::Error;

use futures::Future;

use crate::net::{
    connection::{Connection, ConnectionError},
    network::Network,
};

pub mod agency;
pub mod connection;
pub mod mux;
pub mod network;

// TODO: Serde trait bounds on `T`
// TODO: Properly use this trait for other things (Connection/Agency etc.)
// TODO: Somehow redo the Tuneable, Unicast and Broadcast traits.
pub trait Channel {
    type Error: Error + 'static;

    /// Send a message over the channel
    ///
    /// * `msg`: message to serialize and send
    fn send<T: serde::Serialize + Sync>(
        &mut self,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, Self::Error>> + Send;
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin + Send,
        W: tokio::io::AsyncWrite + std::marker::Unpin + Send,
    > Channel for Connection<R, W>
{
    type Error = ConnectionError;

    async fn send<T: serde::Serialize + Sync>(&mut self, msg: &T) -> Result<(), Self::Error> {
        Connection::send(self, &msg).await
    }

    fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, Self::Error>> {
        Connection::recv(self)
    }
}

/// Tune to a specific channel
pub trait Tuneable {
    type Error: Error + 'static;

    fn id(&self) -> usize;

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::Error>>;

    fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        idx: usize,
        msg: &T,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>>;
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin,
        W: tokio::io::AsyncWrite + std::marker::Unpin,
    > Tuneable for Network<R, W>
{
    type Error = ConnectionError;

    fn id(&self) -> usize {
        self.index
    }

    async fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> Result<T, Self::Error> {
        self[idx].recv().await
    }

    async fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        idx: usize,
        msg: &T,
    ) -> Result<(), Self::Error> {
        self[idx].send(msg).await
    }
}
