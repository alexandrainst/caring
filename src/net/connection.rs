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
    sync::mpsc::Sender,
    time::error::Elapsed,
};

use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

pub struct Connection<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> {
    input: tokio::sync::mpsc::Sender<Box<[u8]>>,
    reader: FramedRead<R, LengthDelimitedCodec>,
    sending: tokio::task::JoinHandle<FramedWrite<W, LengthDelimitedCodec>>,
    flush: tokio::sync::mpsc::Sender<()>,
    did_flush: tokio::sync::mpsc::Receiver<()>,
}

impl<R: AsyncRead + Unpin + Send + 'static, W: AsyncWrite + Unpin + Send + 'static>
    Connection<R, W>
{
    /// Construct a new connection from a reader and writer
    /// Messages are serialized with bincode and length delimated.
    ///
    /// * `reader`: Reader to receive messages from
    /// * `writer`: Writer to send messages to
    pub fn new(reader: R, writer: W) -> Self {
        let codec = LengthDelimitedCodec::new();
        let reader = FramedRead::new(reader, codec.clone());
        let mut writer = FramedWrite::new(writer, codec);

        let (flush, mut should_flush) = tokio::sync::mpsc::channel(1);
        let (flushed, did_flush) = tokio::sync::mpsc::channel(1);

        let (input, mut outgoing): (Sender<Box<[u8]>>, _) = tokio::sync::mpsc::channel(8);
        let sending = tokio::spawn(async move {
            // This self-drops after the sender is gone.
            loop {
                tokio::select! {
                    Some(()) = should_flush.recv() => {
                        // HACK: This really should be guaranteed from the writer's `send` itself.
                        writer.flush().await.unwrap();
                        flushed.send(()).await.unwrap();
                    },
                    Some(msg) = outgoing.recv() => {
                        writer
                            .send(msg.into())
                            .await
                            .expect("Something went wrong sending a message");
                    },
                    else => {
                        break writer
                    }
                };
            }
        });

        Connection {
            input,
            reader,
            sending,
            flush,
            did_flush,
        }
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Connection<R, W> {
    /// Destroy the connection, returning the internal reader and writer.
    pub async fn destroy(self) -> Result<(R, W), ConnectionError> {
        let Self {
            input,
            reader,
            sending,
            ..
        } = self;
        drop(input);
        // Should not wait much here since we drop input
        // it is really only unsent packages holding us back
        let writer = sending
            .await
            .map_err(|e| ConnectionError::Unknown(Box::new(e)))?
            .into_inner();
        let reader = reader.into_inner();
        Ok((reader, writer))
    }

    pub async fn flush(&mut self) -> Result<(), ConnectionError> {
        self.flush
            .send(())
            .await
            .map_err(|_| ConnectionError::Closed)?;
        self.did_flush.recv().await;
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum ConnectionError {
    #[error("Deserialization failed")]
    BadSerialization(#[from] bincode::Error),
    #[error("Connection timed out after {0}")]
    TimeOut(Elapsed),
    #[error("No message to receive")]
    Closed,
    #[error("Unknown error")]
    Unknown(#[from] Box<dyn Error + Send>),
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Connection<R, W> {
    /// Send a message without waiting
    ///
    /// The message is queued, if the queue is too full it panics.
    ///
    /// * `msg`: Message to send
    pub fn send(&self, msg: &impl serde::Serialize) {
        let msg = bincode::serialize(msg).unwrap();
        self.input.try_send(msg.into()).unwrap();
    }

    /// Receive a message waiting for arrival
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, ConnectionError> {
        // TODO: Handle timeouts?
        let buf = self
            .reader
            .next()
            .await
            .ok_or(ConnectionError::Closed)?
            .map_err(|e| ConnectionError::Unknown(Box::new(e)))?;
        let buf = std::io::Cursor::new(buf);
        bincode::deserialize_from(buf).map_err(ConnectionError::BadSerialization)
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

pub type TlsConnection = Connection<
    ReadHalf<tokio_rustls::TlsStream<TcpStream>>,
    WriteHalf<tokio_rustls::TlsStream<TcpStream>>,
>;
impl TlsConnection {
    /// New TLS-based connection from a stream
    ///
    /// * `stream`: TCP stream to use
    pub fn from_tls(stream: tokio_rustls::TlsStream<TcpStream>) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        Self::new(reader, writer)
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

    use tokio::net::{TcpListener, TcpStream};

    use super::*;

    #[tokio::test]
    async fn in_memory() {
        let (conn1, conn2) = DuplexConnection::in_memory();
        let h1 = async move {
            let mut conn = conn1;
            conn.send(&"Hello");
            println!("[1] Message sent");
            conn.send(&"Buddy");
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
            conn.send(&"Greetings friend");
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
            conn.send(&"Hello");
            println!("[1] Message sent");
            conn.send(&"Buddy");
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
            conn.send(&"Greetings friend");
            println!("[2] Message sent");
        };

        futures::join!(h1, h2);
    }
}
