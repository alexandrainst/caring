use std::error::Error;

use futures::Future;

use crate::net::connection::{Connection, ConnectionError, RecvBytes, SendBytes};

pub mod agency;
pub mod connection;
pub mod mux;
pub mod network;

/// A communication medium between you and another party.
///
/// Allows you to send and receive arbitrary messages.
pub trait Channel {
    type Error: Error + Send + Sync + 'static;

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

/// A [Channel] which can be split into a sender and receiver.
pub trait SplitChannel: Channel {
    type Sender: SendBytes<SendError = Self::Error> + Send;
    type Receiver: RecvBytes<RecvError = Self::Error> + Send;
    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver);
}

impl<'a, C: Channel> Channel for &'a mut C {
    type Error = C::Error;

    fn send<T: serde::Serialize + Sync>(
        &mut self,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        (**self).send(msg)
    }

    fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, Self::Error>> + Send {
        (**self).recv()
    }
}

/// Tune to a specific channel
pub trait Tuneable {
    type Error: Error + Send + 'static;

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

impl<'a, R: Tuneable + ?Sized> Tuneable for &'a mut R {
    type Error = R::Error;

    fn id(&self) -> usize {
        (**self).id()
    }

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::Error>> {
        (**self).recv_from(idx)
    }

    fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        idx: usize,
        msg: &T,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> {
        (**self).send_to(idx, msg)
    }
}

/// General communication with support for most network functionality.
pub trait Communicate: agency::Broadcast + agency::Unicast + Tuneable + Send {}
impl<'a, C: Communicate> Communicate for &'a mut C {}
