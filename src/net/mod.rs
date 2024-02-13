use std::{error::Error, ops::{Index, IndexMut}};

use futures::Future;

use crate::net::{
    agency::{Broadcast, Unicast},
    connection::{Connection, ConnectionError},
    network::Network,
};

pub mod agency;
pub mod connection;
pub mod network;

// TODO: Serde trait bounds on `T`
// TODO: Properly use this trait for other things (Connection/Agency etc.)
pub trait Channel {
    type Error : Error;

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
    type Error;
    type SubChannel: Channel;
    type Idx;

    fn id(&self) -> Self::Idx;

    fn tune_mut(&mut self, idx: Self::Idx) -> &mut Self::SubChannel;

    fn tune(&self, idx: Self::Idx) -> &Self::SubChannel;

}


impl<
        R: tokio::io::AsyncRead + std::marker::Unpin,
        W: tokio::io::AsyncWrite + std::marker::Unpin,
    > Tuneable for Network<R, W>
{
    type Error = ConnectionError;
    type SubChannel = Connection<R, W>;
    type Idx = usize;

    fn id(&self) -> Self::Idx {
        self.index
    }

    fn tune_mut(&mut self, idx: Self::Idx) -> &mut Self::SubChannel {
        self.index_mut(idx)
    }

    fn tune(&self, idx: Self::Idx) -> &Self::SubChannel {
        self.index(idx)
    }
}
