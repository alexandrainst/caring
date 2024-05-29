//! Module for doing arbitrary communication in 'some' medium.
//! This 'medium' can be anything that implements `AsyncRead`/`AsyncWrite`.
//! There is built-in support for TCP and in-memory duplex-based connections.
//!
//! One thing to consider is multiplexing, in the case where we want to
//! perform multiple protocols in parallel. Thus to ensure we receive the right packets
//! back and forth, we need to open a connection for each 'protocol'.
//! One method for this is to use something like:
//! <https://github.com/black-binary/async-smux>
//!
//! In relation to the above, we might want to restrict 'send' with mut.
//! although, maybe 'recv' is enough. We just need to prevent threads or other
//! concurrent things from sending/receiving out of order.
//!
//! All in all, this is only relevant if we want to perform some form extra concurrent protocol.
//! This could be background verification and 'anti-cheat' detection, error-reporting,
//! background beaver share generation, or other preproccessing actions.

use std::error::Error;

use futures::{SinkExt, StreamExt};
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncWrite, DuplexStream, ReadHalf, WriteHalf},
    net::{
        tcp::{OwnedReadHalf, OwnedWriteHalf},
        TcpStream,
    },
    time::error::Elapsed,
};

use tokio_util::{
    bytes::{Bytes, BytesMut},
    codec::{FramedRead, FramedWrite, LengthDelimitedCodec},
};

use crate::net::{Channel, RecvBytes, SendBytes, SplitChannel};

pub struct Connection<R: AsyncRead, W: AsyncWrite> {
    sender: Sending<W>,
    receiver: Receiving<R>,
}

#[derive(Error, Debug)]
pub enum ConnectionError {
    #[error("Deserialization failed")]
    MalformedMessage(#[from] bincode::Error),
    #[error("Connection timed out after {0}")]
    TimeOut(Elapsed),
    #[error("No message to receive")]
    Closed,
    #[error("Unknown error")]
    Unknown(#[from] Box<dyn Error + Send + Sync + 'static>),
}

impl<R: AsyncRead, W: AsyncWrite> Connection<R, W> {
    /// Construct a new connection from a reader and writer
    /// Messages are serialized with bincode and length delimated.
    ///
    /// * `reader`: Reader to receive messages from
    /// * `writer`: Writer to send messages to
    pub fn new(reader: R, writer: W) -> Self {
        let codec = LengthDelimitedCodec::new();
        let reader = FramedRead::new(reader, codec.clone());
        let writer = FramedWrite::new(writer, codec);

        let sender = Sending(writer);
        let receiver = Receiving(reader);
        Connection { sender, receiver }
    }

    /// Destroy the connection, returning the internal reader and writer.
    pub async fn destroy(self) -> Result<(R, W), ConnectionError> {
        let Self { sender, receiver } = self;
        // Should not wait much here since we drop input
        // it is really only unsent packages holding us back
        Ok((receiver.0.into_inner(), sender.0.into_inner()))
    }
}

pub struct Sending<W: AsyncWrite>(FramedWrite<W, LengthDelimitedCodec>);
pub struct Receiving<R: AsyncRead>(FramedRead<R, LengthDelimitedCodec>);

impl<W: AsyncWrite + Unpin + Send> SendBytes for Sending<W> {
    type SendError = ConnectionError;

    async fn send_bytes(&mut self, bytes: Bytes) -> Result<(), Self::SendError> {
        SinkExt::<_>::send(&mut self.0, bytes)
            .await
            .map_err(|_| ConnectionError::Closed)
    }
}

impl<R: AsyncRead + Unpin + Send> RecvBytes for Receiving<R> {
    type RecvError = ConnectionError;

    async fn recv_bytes(&mut self) -> Result<BytesMut, Self::RecvError> {
        self.0
            .next()
            .await
            .ok_or(ConnectionError::Closed)?
            .map_err(|e| ConnectionError::Unknown(Box::new(e)))
    }
}

impl<R: AsyncRead + Unpin + Send, W: AsyncWrite + Unpin + Send> Connection<R, W> {
    /// Send a message, waiting until receival
    ///
    /// * `msg`: Message to send
    pub async fn send(
        &mut self,
        msg: &(impl serde::Serialize + Sync),
    ) -> Result<(), ConnectionError> {
        self.sender.send(msg).await
    }

    /// Receive a message waiting for arrival
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, ConnectionError> {
        self.receiver.recv().await
    }
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin + Send,
        W: tokio::io::AsyncWrite + std::marker::Unpin + Send,
    > SendBytes for Connection<R, W>
{
    type SendError = ConnectionError;

    fn send_bytes(
        &mut self,
        bytes: Bytes,
    ) -> impl std::future::Future<Output = Result<(), Self::SendError>> + Send {
        self.sender.send_bytes(bytes)
    }
}
impl<
        R: tokio::io::AsyncRead + std::marker::Unpin + Send,
        W: tokio::io::AsyncWrite + std::marker::Unpin + Send,
    > RecvBytes for Connection<R, W>
{
    type RecvError = ConnectionError;

    fn recv_bytes(
        &mut self,
    ) -> impl std::future::Future<Output = Result<BytesMut, Self::RecvError>> + Send {
        self.receiver.recv_bytes()
    }
}

impl<R: AsyncRead + Unpin + Send, W: AsyncWrite + Unpin + Send> Channel for Connection<R, W> {
    type Error = ConnectionError;
}

impl<R: AsyncRead + Unpin + Send, W: AsyncWrite + Unpin + Send> SplitChannel for Connection<R, W> {
    type Sender = Sending<W>;
    type Receiver = Receiving<R>;

    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver) {
        (&mut self.sender, &mut self.receiver)
    }
}

pub type TcpConnection = Connection<OwnedReadHalf, OwnedWriteHalf>;
impl TcpConnection {
    /// New TCP-based connection from a stream
    ///
    /// * `stream`: TCP stream to use
    pub fn from_tcp(stream: TcpStream) -> Self {
        let (reader, writer) = stream.into_split();
        Self::new(reader, writer)
    }

    pub async fn to_tcp(self) -> Result<TcpStream, ConnectionError> {
        let (r, w) = self.destroy().await?;
        // UNWRAP: Should never fail, as we build the connection from two
        // streams before. However! One could construct TcpConnection manually
        // suing `Connection::new`, thus it 'can' fail.
        // But just don't do that.
        Ok(r.reunite(w).expect("TCP Streams didn't match"))
    }
}

/// Connection to a in-memory data stream.
/// This always have a corresponding other connection in the same process.
pub type DuplexConnection = Connection<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>>;
impl DuplexConnection {
    /// Construct a duplex/in-memory connection pair
    pub fn in_memory() -> (Self, Self) {
        let (s1, s2) = tokio::io::duplex(64);

        let (r1, w1) = tokio::io::split(s1);
        let (r2, w2) = tokio::io::split(s2);

        (Self::new(r1, w1), Self::new(r2, w2))
    }
}

#[cfg(test)]
mod test {

    use std::net::SocketAddrV4;

    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn in_memory() {
        let (conn1, conn2) = DuplexConnection::in_memory();
        let h1 = async move {
            let mut conn = conn1;
            conn.send(&"Hello").await.unwrap();
            println!("[1] Message sent");
            conn.send(&"Buddy").await.unwrap();
            println!("[1] Message sent");
            let msg: Box<str> = conn.recv().await.unwrap();
            println!("[1] Message received");
            assert_eq!(msg, "Greetings friend".into());
        };
        let h2 = async move {
            let mut conn = conn2;
            let msg: Box<str> = conn.recv().await.unwrap();
            println!("[2] Message received");
            assert_eq!(msg, "Hello".into());
            let msg: Box<str> = conn.recv().await.unwrap();
            println!("[2] Message received");
            assert_eq!(msg, "Buddy".into());
            conn.send(&"Greetings friend").await.unwrap();
            println!("[2] Message sent");
        };

        futures::join!(h1, h2);
    }

    #[tokio::test]
    async fn tcp() {
        let addr = "127.0.0.1:4321".parse::<SocketAddrV4>().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let h1 = async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut conn = Connection::from_tcp(stream);
            conn.send(&"Hello").await.unwrap();
            println!("[1] Message sent");
            conn.send(&"Buddy").await.unwrap();
            println!("[1] Message sent");
            let msg: Box<str> = conn.recv().await.unwrap();
            println!("[1] Message received");
            assert_eq!(msg, "Greetings friend".into());
        };
        let h2 = async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut conn = Connection::from_tcp(stream);
            let msg: Box<str> = conn.recv().await.unwrap();
            println!("[2] Message received");
            assert_eq!(msg, "Hello".into());
            let msg: Box<str> = conn.recv().await.unwrap();
            println!("[2] Message received");
            assert_eq!(msg, "Buddy".into());
            conn.send(&"Greetings friend").await.unwrap();
            println!("[2] Message sent");
        };

        futures::join!(h1, h2);
    }
}
