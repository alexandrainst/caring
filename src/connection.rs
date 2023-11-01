//! Module for doing arbitrary communication in 'some' medium.
//! This 'medium' can be anything that implements `AsyncRead`/`AsyncWrite`.
//! There is built-in support for TCP and in-memory duplex-based connections.
//!
//! One thing to consider is multiplexing, in the case where we want to
//! perform multiple protocols in parallel. Thus to ensure we receive the right packets
//! back and forth, we need to open a connection for each 'protocol'.
//! One method for this is to use something like:
//! https://github.com/black-binary/async-smux
//!
//! In relation to the above, we might want to restrict 'send' with mut.
//! although, maybe 'recv' is enough. We just need to prevent threads or other
//! concurrent things from sending/receiving out of order.
//!
//! All in all, this is only relevant if we want to perform some form extra concurrent protocol.
//! This could be background verification and 'anti-cheat' detection, error-reporting,
//! background beaver share generation, or other preproccessing actions.


use futures::{SinkExt, StreamExt};
use tokio::{
    io::{AsyncRead, AsyncWrite, DuplexStream, ReadHalf, WriteHalf},
    net::{
        tcp::{OwnedReadHalf, OwnedWriteHalf},
        TcpStream,
    },
    sync::mpsc::Sender,
};

use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

pub struct Connection<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> {
    input: tokio::sync::mpsc::Sender<Box<[u8]>>,
    reader: FramedRead<R, LengthDelimitedCodec>,
    task: tokio::task::JoinHandle<FramedWrite<W, LengthDelimitedCodec>>,
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin + Send + 'static> Connection<R, W> {
    /// Construct a new connection from a reader and writer
    /// Messages are serialized with bincode and length delimated.
    ///
    /// * `reader`: Reader to receive messages from
    /// * `writer`: Writer to send messages to
    pub fn new(reader: R, writer: W) -> Self {
        let (input, mut outgoing): (Sender<Box<[u8]>>, _) = tokio::sync::mpsc::channel(8);
        let codec = LengthDelimitedCodec::new();
        let reader = FramedRead::new(reader, codec.clone());
        let mut writer = FramedWrite::new(writer, codec);

        let task = tokio::spawn(async move {
            // This self-drops after the sender is gone.
            while let Some(msg) = outgoing.recv().await {
                writer.send(msg.into()).await.unwrap();
            }
            // return the writer
            writer
        });
        Connection {
            task,
            input,
            reader,
        }
    }

    /// Destroy the connection, returning the internal reader and writer.
    pub async fn destroy(self) -> (R, W) {
        let Self {
            input,
            reader,
            task,
        } = self;
        let reader = reader.into_inner();
        drop(input);
        // Should not wait much here since we drop input
        // it is really only unsent packages holding us back
        let writer = task.await.unwrap().into_inner();
        (reader, writer)
    }
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
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> T {
        // TODO: Handle unstable connections
        // TODO: Handle timeouts?
        let buf = self.reader.next().await.unwrap().unwrap();
        let buf = std::io::Cursor::new(buf);
        // TODO: Handle bad deserialization (assume malicious?)
        // We should probably use a Result<>
        bincode::deserialize_from(buf).unwrap()
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

    pub async fn to_tcp(self) -> TcpStream {
        let (r, w) = self.destroy().await;
        // UNWRAP: Should never fail, as we build the connection from two
        // streams before. However! One could construct TcpConnection manually
        // suing `Connection::new`, thus it 'can' fail.
        // But just don't do that.
        r.reunite(w).expect("TCP Streams didn't match")
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
        let h1 = tokio::spawn(async move {
            let mut conn = conn1;
            conn.send(&"Hello");
            println!("[1] Message sent");
            conn.send(&"Buddy");
            println!("[1] Message sent");
            let msg: Box<str> = conn.recv().await;
            println!("[1] Message received");
            assert_eq!(msg, "Greetings friend".into());
        });
        let h2 = tokio::spawn(async move {
            let mut conn = conn2;
            let msg: Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Hello".into());
            let msg: Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Buddy".into());
            conn.send(&"Greetings friend");
            println!("[2] Message sent");
        });

        h2.await.unwrap();
        h1.await.unwrap();
    }

    #[tokio::test]
    async fn tcp() {
        let addr = "127.0.0.1:4321".parse::<SocketAddrV4>().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let h1 = tokio::spawn(async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut conn = Connection::from_tcp(stream);
            conn.send(&"Hello");
            println!("[1] Message sent");
            conn.send(&"Buddy");
            println!("[1] Message sent");
            let msg: Box<str> = conn.recv().await;
            println!("[1] Message received");
            assert_eq!(msg, "Greetings friend".into());
        });
        let h2 = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut conn = Connection::from_tcp(stream);
            let msg: Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Hello".into());
            let msg: Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Buddy".into());
            conn.send(&"Greetings friend");
            println!("[2] Message sent");
        });

        h2.await.unwrap();
        h1.await.unwrap();
    }
}
