use std::{error::Error, future::Future};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use tokio_util::bytes::{Bytes, BytesMut};

pub mod agency;
pub mod connection;
pub mod mux;
pub mod network;

pub trait SendBytes: Send {
    type SendError: Error + Send + Sync + 'static;

    fn send_bytes(
        &mut self,
        bytes: Bytes,
    ) -> impl std::future::Future<Output = Result<(), Self::SendError>> + Send;

    fn send<T: Serialize + Sync>(
        &mut self,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::SendError>> + Send {
        async {
            let msg = bincode::serialize(msg).unwrap();
            self.send_bytes(msg.into()).await
        }
    }
}

impl<S: SendBytes> SendBytes for &mut S {
    type SendError = S::SendError;

    fn send_bytes(
        &mut self,
        bytes: Bytes,
    ) -> impl std::future::Future<Output = Result<(), Self::SendError>> + Send {
        (**self).send_bytes(bytes)
    }
}

#[derive(Debug, Error)]
pub enum ReceiverError<E> {
    #[error("Bad Serialization: {0}")]
    BadSerialization(bincode::Error),
    #[error("IO Error: {0}")]
    IO(#[from] E),
}

pub trait RecvBytes: Send {
    type RecvError: Error + Send + Sync + 'static;
    fn recv_bytes(
        &mut self,
    ) -> impl std::future::Future<Output = Result<BytesMut, Self::RecvError>> + Send;

    fn recv<T: DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, ReceiverError<Self::RecvError>>> + Send {
        async {
            let msg = self.recv_bytes().await?;
            bincode::deserialize(&msg).map_err(ReceiverError::BadSerialization)
        }
    }
}

impl<R: RecvBytes> RecvBytes for &mut R {
    type RecvError = R::RecvError;

    fn recv_bytes(
        &mut self,
    ) -> impl std::future::Future<Output = Result<BytesMut, Self::RecvError>> + Send {
        (**self).recv_bytes()
    }
}

/// A communication medium between you and another party.
///
/// Allows you to send and receive arbitrary messages.
pub trait Channel: SendBytes<SendError = Self::Error> + RecvBytes<RecvError = Self::Error> {
    type Error: Error + Send + Sync + 'static;
}
impl<C: Channel> Channel for &mut C {
    type Error = C::Error;
}

/// A [Channel] which can be split into a sender and receiver.
pub trait SplitChannel: Channel + Send {
    type Sender: SendBytes<SendError = Self::SendError> + Send;
    type Receiver: RecvBytes<RecvError = Self::RecvError> + Send;
    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver);
}

impl<'a, C: SplitChannel> SplitChannel for &'a mut C {
    type Sender = C::Sender;
    type Receiver = C::Receiver;

    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver) {
        (**self).split()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct Id(pub usize);

/// Tune to a specific channel
pub trait Tuneable {
    type TuningError: Error + Send + 'static;
    type Channel: SplitChannel + Send;

    fn id(&self) -> Id;

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        id: Id,
    ) -> impl Future<Output = Result<T, Self::TuningError>> + Send;

    fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        id: Id,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::TuningError>> + Send;

    fn channels(&mut self) -> &mut [Self::Channel];
}

impl<'a, N: Tuneable + ?Sized> Tuneable for &'a mut N {
    type TuningError = N::TuningError;
    type Channel = N::Channel;

    fn id(&self) -> Id {
        (**self).id()
    }

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        id: Id,
    ) -> impl Future<Output = Result<T, Self::TuningError>> + Send {
        (**self).recv_from(id)
    }

    fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        id: Id,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::TuningError>> + Send {
        (**self).send_to(id, msg)
    }

    fn channels(&mut self) -> &mut [Self::Channel] {
        (**self).channels()
    }
}

/// General communication with support for most network functionality.
pub trait Communicate: agency::Broadcast + agency::Unicast + Tuneable + Send {}
impl<'a, C: Communicate> Communicate for &'a mut C {}
