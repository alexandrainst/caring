use std::io::Write;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream};
use tokio::io::{ReadHalf, WriteHalf};

use crate::net::{
    connection::{Connection, ConnectionError, Receiving, Sending},
    network::{InMemoryNetwork, Network},
    Channel, ReceiverError, RecvBytes, SendBytes, SplitChannel,
};

pub struct TappedConnection<R: AsyncRead, W: AsyncWrite>(TapSending<W>, TapReceiving<R>);

impl<R: AsyncRead, W: AsyncWrite> Connection<R, W> {
    pub fn tap(self, name: String) -> TappedConnection<R, W> {
        let r = TapReceiving {
            name: name.clone(),
            inner: self.receiver,
            buf: Vec::new(),
        };
        let s = TapSending {
            name,
            inner: self.sender,
            buf: Vec::new(),
        };
        TappedConnection(s, r)
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> SendBytes
    for TappedConnection<R, W>
{
    type SendError = ConnectionError;

    async fn send_bytes(&mut self, bytes: tokio_util::bytes::Bytes) -> Result<(), Self::SendError> {
        self.0.send_bytes(bytes).await
    }

    async fn send<T: serde::Serialize + Sync>(&mut self, msg: &T) -> Result<(), Self::SendError> {
        self.0.send(msg).await
    }
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> RecvBytes
    for TappedConnection<R, W>
{
    type RecvError = ConnectionError;

    async fn recv_bytes(&mut self) -> Result<tokio_util::bytes::BytesMut, Self::RecvError> {
        self.1.recv_bytes().await
    }

    async fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> Result<T, ReceiverError<Self::RecvError>> {
        self.1.recv().await
    }
}

pub struct TapSending<W: AsyncWrite> {
    inner: Sending<W>,
    name: String,
    buf: Vec<u8>,
}

impl<W: AsyncWrite + Send + Unpin> SendBytes for TapSending<W> {
    type SendError = ConnectionError;

    async fn send_bytes(&mut self, bytes: tokio_util::bytes::Bytes) -> Result<(), Self::SendError> {
        self.inner.send_bytes(bytes).await
    }

    async fn send<T: serde::Serialize + Sync>(&mut self, msg: &T) -> Result<(), Self::SendError> {
        let pretty = serde_json::to_string(msg).unwrap();
        let msg = bincode::serialize(msg).unwrap();
        let name = &self.name;
        let _ = writeln!(self.buf, "[{name}] Sending: {pretty}");
        let mut stdout = tokio::io::stdout();
        let _ = stdout.write_all(&self.buf).await;
        self.buf.clear();
        self.send_bytes(msg.into()).await
    }
}

impl<R: AsyncRead + Send + Unpin> RecvBytes for TapReceiving<R> {
    type RecvError = ConnectionError;

    async fn recv_bytes(&mut self) -> Result<tokio_util::bytes::BytesMut, Self::RecvError> {
        self.inner.recv_bytes().await
    }
}

pub struct TapReceiving<R: AsyncRead> {
    inner: Receiving<R>,
    name: String,
    buf: Vec<u8>,
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> Channel for TappedConnection<R, W> {
    type Error = ConnectionError;
}

impl<R: AsyncRead + Send + Unpin, W: AsyncWrite + Send + Unpin> SplitChannel
    for TappedConnection<R, W>
{
    type Sender = TapSending<W>;
    type Receiver = TapReceiving<R>;

    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver) {
        (&mut self.0, &mut self.1)
    }
}

pub type TappedDuplexConnection = TappedConnection<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>>;
pub type TappedInMemoryNetwork = Network<TappedDuplexConnection>;

impl InMemoryNetwork {
    pub fn tap(self, name: impl AsRef<str>) -> TappedInMemoryNetwork {
        let name = name.as_ref();
        let index = self.index;
        let connections: Vec<_> = self
            .connections
            .into_iter()
            .enumerate()
            .map(|(id, c)| {
                let id = if id >= index { id + 1 } else { id };
                let name = format!("{name} | Channel {id}");
                c.tap(name)
            })
            .collect();
        TappedInMemoryNetwork { index, connections }
    }
}
