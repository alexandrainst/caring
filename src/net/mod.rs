use std::{error::Error, future::Future};

use serde::{de::DeserializeOwned, Serialize};
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

pub trait RecvBytes: Send {
    type RecvError: Error + Send + Sync + 'static;
    fn recv_bytes(
        &mut self,
    ) -> impl std::future::Future<Output = Result<BytesMut, Self::RecvError>> + Send;

    fn recv<T: DeserializeOwned>(
        &mut self,
    ) -> impl Future<Output = Result<T, Self::RecvError>> + Send {
        async {
            let msg = self.recv_bytes().await?;
            Ok(bincode::deserialize(&msg).unwrap())
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
pub trait Channel: SendBytes + RecvBytes {
    type Error: Error + Send;
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

/// Tune to a specific channel
pub trait Tuneable {
    type TuningError: Error + Send + 'static;

    fn id(&self) -> usize;

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::TuningError>>;

    fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        idx: usize,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::TuningError>>;
}

impl<'a, R: Tuneable + ?Sized> Tuneable for &'a mut R {
    type TuningError = R::TuningError;

    fn id(&self) -> usize {
        (**self).id()
    }

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::TuningError>> {
        (**self).recv_from(idx)
    }

    fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        idx: usize,
        msg: &T,
    ) -> impl Future<Output = Result<(), Self::TuningError>> {
        (**self).send_to(idx, msg)
    }
}

/// General communication with support for most network functionality.
pub trait Communicate: agency::Broadcast + agency::Unicast + Tuneable + Send {}
impl<'a, C: Communicate> Communicate for &'a mut C {}
