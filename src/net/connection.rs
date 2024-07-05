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

use std::{error::Error, time::Duration};

use futures::{SinkExt, StreamExt};
use futures_concurrency::future::Join;
use thiserror::Error;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf},
    time::error::Elapsed,
};

#[cfg(not(feature = "turmoil"))]
use tokio::net::{
    tcp::{OwnedReadHalf, OwnedWriteHalf},
    TcpStream,
};

#[cfg(feature = "turmoil")]
use turmoil::net::{
    tcp::{OwnedReadHalf, OwnedWriteHalf},
    TcpStream,
};

use tokio_util::{
    bytes::{Bytes, BytesMut},
    codec::{FramedRead, FramedWrite, LengthDelimitedCodec},
};

use crate::net::{connection::latency::Delayed, Channel, RecvBytes, SendBytes, SplitChannel};

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

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Connection<R, W> {
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
    pub fn destroy(self) -> (R, W) {
        let Self { sender, receiver } = self;
        // Should not wait much here since we drop input
        // it is really only unsent packages holding us back
        (receiver.0.into_inner(), sender.0.into_inner())
    }

    pub fn delayed_read(self, delay: Duration) -> Connection<Delayed<R>, W> {
        let r = self.receiver.0.into_inner();
        let r = Delayed::new(r, delay);
        let w = self.sender.0.into_inner();
        Connection::new(r, w)
    }

    pub fn delayed_write(self, delay: Duration) -> Connection<R, Delayed<W>> {
        let w = self.sender.0.into_inner();
        let w = Delayed::new(w, delay);
        let r = self.receiver.0.into_inner();
        Connection::new(r, w)
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
    pub fn from_tcp_stream(stream: TcpStream) -> Self {
        #[cfg(not(feature = "turmoil"))]
        let _ = stream.set_nodelay(true);

        let (reader, writer) = stream.into_split();
        Self::new(reader, writer)
    }

    pub fn to_tcp_stream(self) -> TcpStream {
        let (r, w) = self.destroy();
        // UNWRAP: Should never fail, as we build the connection from two
        // streams before. However! One could construct TcpConnection manually
        // suing `Connection::new`, thus it 'can' fail.
        // But just don't do that.
        r.reunite(w).expect("TCP Streams didn't match")
    }

    pub async fn shutdown(self) -> Result<(), std::io::Error> {
        self.to_tcp_stream().shutdown().await
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

    /// Gracefully shutdown the connection.
    ///
    /// # Errors
    ///
    /// This function will return an error if the connection could not be shutdown cleanly.
    pub async fn shutdown(self) -> Result<(), std::io::Error> {
        let (mut r, mut w) = self.destroy();
        // HACK: A single read/write so we don't exit too early.
        // We ignore the errors, since we don't care if we can't send or receive,
        // it is supposed to be closed.
        let (_, _) = (r.read_u8(), w.write_u8(0)).join().await;
        let mut stream = r.unsplit(w);
        stream.shutdown().await
    }
}

pub mod latency {
    //! Primitive latency for arbitrary `AsyncWrite`/`AsyncRead` types
    //! for similating network latency.
    //!
    //! When reading or flushing we run a timer in which we wait until it is done,
    //! after which the
    //!
    //! Writing is probably the most realistic, given that it runs a timer after flushing,
    //! which could be slow.

    use std::{pin::Pin, time::Duration};

    use futures::{pin_mut, FutureExt};
    use tokio::{
        io::{AsyncRead, AsyncWrite},
        time::Sleep,
    };

    pub struct Delayed<T> {
        inner: T,
        delay: Duration,
        timer: Pin<Box<Sleep>>,
    }

    impl<T> Delayed<T> {
        pub fn new(t: T, delay: Duration) -> Self {
            Self {
                inner: t,
                delay,
                timer: Box::pin(tokio::time::sleep(delay)),
            }
        }

        pub fn destroy(self) -> T {
            self.inner
        }
    }

    impl<T> AsyncWrite for Delayed<T>
    where
        T: AsyncWrite + Unpin,
    {
        fn poll_write(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &[u8],
        ) -> std::task::Poll<Result<usize, std::io::Error>> {
            let writer = &mut self.inner;
            pin_mut!(writer);
            writer.poll_write(cx, buf)
        }

        fn poll_flush(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), std::io::Error>> {
            let Delayed {
                timer,
                inner,
                delay,
            } = &mut *self;
            match timer.poll_unpin(cx) {
                std::task::Poll::Ready(()) => {
                    pin_mut!(inner);
                    timer.set(tokio::time::sleep(*delay));
                    inner.poll_flush(cx)
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }

        fn poll_shutdown(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Result<(), std::io::Error>> {
            let writer = &mut self.inner;
            pin_mut!(writer);
            writer.poll_shutdown(cx)
        }
    }

    impl<T> AsyncRead for Delayed<T>
    where
        T: AsyncRead + Unpin,
    {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            let Delayed {
                timer,
                inner,
                delay,
            } = &mut *self;
            match timer.poll_unpin(cx) {
                std::task::Poll::Ready(()) => {
                    pin_mut!(inner);
                    timer.set(tokio::time::sleep(*delay));
                    inner.poll_read(cx, buf)
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod test {

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

    #[cfg(not(feature = "turmoil"))]
    #[tokio::test]
    async fn tcp() {
        use std::net::SocketAddrV4;
        use tokio::net::TcpListener;
        let addr = "127.0.0.1:4321".parse::<SocketAddrV4>().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let h1 = async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut conn = Connection::from_tcp_stream(stream);
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
            let mut conn = Connection::from_tcp_stream(stream);
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
    async fn in_memory_delayed_read() {
        use std::time::Instant;
        let (conn1, conn2) = DuplexConnection::in_memory();
        let h1 = async move {
            let mut conn = conn1.delayed_read(Duration::from_millis(50));
            let t0 = Instant::now();
            conn.send(&"Hello").await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[1] Message sent in {delta_t:#?}");

            let t0 = Instant::now();
            conn.send(&"Buddy").await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[1] Message sent in {delta_t:#?}");

            let t0 = Instant::now();
            let msg: Box<str> = conn.recv().await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[1] Message received in {delta_t:#?}");
            assert_eq!(msg, "Greetings friend".into());
        };
        let h2 = async move {
            let mut conn = conn2.delayed_read(Duration::from_millis(50));
            let t0 = Instant::now();
            let msg: Box<str> = conn.recv().await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[2] Message received in {delta_t:#?}");
            assert_eq!(msg, "Hello".into());

            let t0 = Instant::now();
            let msg: Box<str> = conn.recv().await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[2] Message received in {delta_t:#?}");
            assert_eq!(msg, "Buddy".into());

            let t0 = Instant::now();
            conn.send(&"Greetings friend").await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[2] Message sent in {delta_t:#?}");
        };

        futures::join!(h1, h2);
    }

    #[tokio::test]
    async fn in_memory_delayed_write() {
        use std::time::Instant;
        let (conn1, conn2) = DuplexConnection::in_memory();
        let h1 = async move {
            let mut conn = conn1.delayed_write(Duration::from_millis(50));
            let t0 = Instant::now();
            conn.send(&"Hello").await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[1] Message sent in {delta_t:#?}");

            let t0 = Instant::now();
            conn.send(&"Buddy").await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[1] Message sent in {delta_t:#?}");

            let t0 = Instant::now();
            let msg: Box<str> = conn.recv().await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[1] Message received in {delta_t:#?}");
            assert_eq!(msg, "Greetings friend".into());
        };
        let h2 = async move {
            let mut conn = conn2.delayed_write(Duration::from_millis(50));
            let t0 = Instant::now();
            let msg: Box<str> = conn.recv().await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[2] Message received in {delta_t:#?}");
            assert_eq!(msg, "Hello".into());

            let t0 = Instant::now();
            let msg: Box<str> = conn.recv().await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[2] Message received in {delta_t:#?}");
            assert_eq!(msg, "Buddy".into());

            let t0 = Instant::now();
            conn.send(&"Greetings friend").await.unwrap();
            let delta_t = Instant::now() - t0;
            println!("[2] Message sent in {delta_t:#?}");
        };

        futures::join!(h1, h2);
    }
}
