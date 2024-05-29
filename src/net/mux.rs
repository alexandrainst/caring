//! Multiplexing Connections and Networks
//!
//! This moudle provides tools to multiplex channels/connections and networks
//! in order to run multiple protocols concurrently.

// TODO: Add Multiplex trait with impls for Network and SplitChannel?

use std::{error::Error, vec::Drain};
use std::{ops::RangeBounds, sync::Arc};

use futures::{
    channel::{mpsc, oneshot},
    future::join_all,
    Future, FutureExt, SinkExt, StreamExt,
};
use itertools::{multiunzip, Itertools};
use thiserror::Error;
use tokio::join;

use crate::{
    help,
    net::{network::Network, Channel, RecvBytes, SendBytes, SplitChannel},
};

#[derive(Debug, Error)]
pub enum MuxError {
    #[error("Dead Gateway, perhaps it was dropped too early?")]
    DeadGateway,
    #[error(transparent)]
    Connection(Arc<dyn Error + Send + Sync + 'static>),
}

pub struct MuxedSender {
    id: usize,
    gateway: mpsc::UnboundedSender<MultiplexedMessage>,
    error: oneshot::Receiver<MuxError>,
}

pub struct MuxedReceiver {
    id: usize,
    mailbox: mpsc::UnboundedReceiver<BytesMut>,
    error: oneshot::Receiver<MuxError>,
}

struct MultiplexedMessage(Bytes, usize);

impl MultiplexedMessage {
    fn from_bytes(mut bytes: BytesMut) -> Self {
        let id = bytes.get_u32() as usize;
        MultiplexedMessage(bytes.freeze(), id)
    }

    fn make_bytes(self) -> Bytes {
        let bytes = self.0;
        let id = self.1;
        let mut msg = BytesMut::new();
        msg.put_u32(id as u32);
        msg.put(bytes);
        msg.freeze()
    }
}

impl SendBytes for MuxedSender {
    type SendError = MuxError;

    async fn send_bytes(&mut self, bytes: tokio_util::bytes::Bytes) -> Result<(), Self::SendError> {
        futures::select! {
            default => {
                self.gateway
                    .send(MultiplexedMessage(bytes, self.id))
                    .await
                    .map_err(|_| MuxError::DeadGateway)
            },
            res = &mut self.error => {
                let err = res.map_err(|_| MuxError::DeadGateway)?;
                Err(err)
            },
        }
    }
}

use tokio_util::bytes::{Buf, BufMut, Bytes, BytesMut};
impl RecvBytes for MuxedReceiver {
    type RecvError = MuxError;

    async fn recv_bytes(&mut self) -> Result<tokio_util::bytes::BytesMut, Self::RecvError> {
        futures::select! {
            msg = self.mailbox.next() => {
                let msg = msg.ok_or(MuxError::DeadGateway)?;
                Ok(msg)
            },
            res = &mut self.error => {
                // Change to error
                let err = res.map_err(|_| MuxError::DeadGateway)?;
                Err(err)
            }
        }
    }
}

// NOTE: This can still be axed in favor of revamping Connection with
// generic parameters using Send/RecvBytes instead of AsyncWrite/AsyncRead.
// (FramedRead is made compatible with the Send/RecvBytes traits)
//
/// Multiplexed Connection
///
/// Aqquirred by constructing a [Gateway] using [Gateway::multiplex]
pub struct MuxConn(MuxedSender, MuxedReceiver);

impl Channel for MuxConn {
    type Error = MuxError;
}

impl SendBytes for MuxConn {
    type SendError = MuxError;

    fn send_bytes(
        &mut self,
        bytes: Bytes,
    ) -> impl std::future::Future<Output = Result<(), Self::SendError>> + Send {
        self.0.send_bytes(bytes)
    }
}
impl RecvBytes for MuxConn {
    type RecvError = MuxError;

    fn recv_bytes(
        &mut self,
    ) -> impl std::future::Future<Output = Result<BytesMut, Self::RecvError>> + Send {
        self.1.recv_bytes()
    }
}

impl SplitChannel for MuxConn {
    type Sender = MuxedSender;
    type Receiver = MuxedReceiver;

    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver) {
        (&mut self.0, &mut self.1)
    }
}

struct GatewayInner<'a, C>
where
    C: SplitChannel,
{
    channel: &'a mut C,
    mailboxes: Vec<mpsc::UnboundedSender<BytesMut>>,
    inbox: mpsc::UnboundedReceiver<MultiplexedMessage>,
    errors: Vec<[oneshot::Sender<MuxError>; 2]>,
}

/// Gateway channel for multiplexed connections/channels ([MuxConn]),
/// interally holding a [SplitChannel].
///
/// Constructed by [Gateway::multiplex]
pub struct Gateway<'a, C: SplitChannel> {
    inner: GatewayInner<'a, C>,
    muxes: Vec<MuxConn>,
}

impl<'a, C: SplitChannel + Send + 'static> Gateway<'a, C> {
    /// Multiplex a channel to share it into `n` new connections.
    ///
    /// * `net`: Connection to use as a gateway for multiplexing
    /// * `n`: Number of new connections to multiplex into
    ///
    /// Returns a gateway which the MuxConn communicate through, along with the MuxConn
    ///
    /// # Example
    /// ```
    /// # use crate::caring::net::SendBytes;
    /// # use caring::net::connection::Connection;
    /// # use caring::net::mux::Gateway;
    /// # tokio_test::block_on(async {
    /// # let (c1, c2) = Connection::in_memory();
    /// # let first = async move {
    /// # let mut con = c1;
    /// use crate::caring::net::Channel;
    /// use itertools::Itertools;
    ///
    /// let mut gateway = Gateway::multiplex(&mut con, 2);
    /// let (mut m1, mut m2) = gateway.drain(..).collect_tuple().unwrap();
    /// let t1 = async move {
    ///     m1.send(&String::from("Hello")).await.unwrap();
    /// };
    /// let t2 = async move {
    ///     m2.send(&String::from("Friend")).await.unwrap();
    /// };
    /// futures::join!(t1, t2, gateway.drive()); // Gateway needs to be run aswell.
    /// # };
    /// #
    /// # use crate::caring::net::RecvBytes;
    /// # use itertools::Itertools;
    /// # use crate::caring::net::Channel;
    /// # let second = async move {
    /// # let mut con = c2;
    /// # let mut gateway = Gateway::multiplex(&mut con, 2);
    /// # let (mut m1, mut m2) = gateway.drain(..).collect_tuple().unwrap();
    /// # let t1 = async move {
    /// #     let _ : String = m1.recv().await.unwrap();
    /// # };
    /// # let t2 = async move {
    /// #     let _ : String = m2.recv().await.unwrap();
    /// # };
    /// # futures::join!(t1, t2, gateway.drive());
    /// # };
    /// # futures::join!(first, second)
    /// # });
    ///
    /// ```
    ///
    pub fn multiplex(con: &'a mut C, n: usize) -> Self {
        let (gateway, inbox) = mpsc::unbounded();

        let (sends, channels, errors) = multiunzip((0..n).map(|id| {
            let (send, recv) = mpsc::unbounded();
            let gateway = gateway.clone();

            let mailbox = recv;
            let (error_coms1, error) = oneshot::channel();
            let receiver = MuxedReceiver { id, mailbox, error };
            let (error_coms2, error) = oneshot::channel();
            let sender = MuxedSender { id, gateway, error };
            let chan = MuxConn(sender, receiver);

            //
            (send, chan, [error_coms1, error_coms2])
        }));

        let inner = GatewayInner {
            mailboxes: sends,
            inbox,
            channel: con,
            errors,
        };

        Self {
            inner,
            muxes: channels,
        }
    }

    pub async fn map<T, F: Future<Output = T>>(self, func: impl FnMut(MuxConn) -> F) -> Vec<T> {
        let res = join_all(self.muxes.into_iter().map(func));
        let (res, _) = join!(res, self.inner.run());
        res
    }

    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> Drain<MuxConn> {
        self.muxes.drain(range)
    }

    pub async fn drive(self) {
        self.inner.run().await;
    }
}

impl<'a, C: SplitChannel + Send> GatewayInner<'a, C> {
    async fn run(self) -> Self {
        let mut gateway = self;
        {
            let (sending, recving) = gateway.channel.split();

            let send_out = async {
                while let Some(msg) = gateway.inbox.next().await {
                    // TODO: Error propagation.
                    sending.send_bytes(msg.make_bytes()).await.unwrap();
                }
            }
            .fuse();

            let recv_in = async {
                loop {
                    match recving.recv_bytes().await {
                        Ok(mut msg) => {
                            let id = msg.get_u32() as usize;
                            let bytes = msg;
                            gateway.mailboxes[id].send(bytes).await.unwrap();
                        }
                        Err(e) => break e,
                    }
                }
            }
            .fuse();

            futures::pin_mut!(send_out, recv_in);
            futures::select! { // Drive both futures to completion.
                () = send_out => {},
                err = recv_in => {
                    let err = Arc::new(err);
                    for [c1, c2] in gateway.errors.drain(..) {
                        // ignore dropped connections,
                        // they can't handle errors when they don't exist.
                        let _ = c1.send(MuxError::Connection(err.clone()));
                        let _ = c2.send(MuxError::Connection(err.clone()));
                    }
                },
            };
        }

        gateway
    }
}

pub struct NetworkGateway<'a, C: SplitChannel> {
    gateways: Vec<Gateway<'a, C>>,
    index: usize,
}

type MuxNet = Network<MuxConn>;

impl<'a, C> NetworkGateway<'a, C>
where
    C: SplitChannel + Send + 'static,
{
    pub fn multiplex(net: &'a mut Network<C>, n: usize) -> (Self, Vec<MuxNet>) {
        let mut gateways = Vec::new();
        let mut matrix = Vec::new();
        let index = net.index;
        for conn in net.connections.iter_mut() {
            let mut gateway = Gateway::multiplex(conn, n);
            let muxes = gateway.drain(..).collect_vec();
            matrix.push(muxes);
            gateways.push(gateway);
        }
        let gateway = NetworkGateway { gateways, index };

        let matrix = help::transpose(matrix);
        let muxnets: Vec<_> = matrix
            .into_iter()
            .map(|connections| MuxNet { connections, index })
            .collect();

        (gateway, muxnets)
    }

    pub async fn drive(self) {
        let _ = join_all(self.gateways.into_iter().map(|c| c.drive())).await;
    }
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use futures_concurrency::future::Join;
    use itertools::Itertools;
    use tokio::join;

    use crate::net::{
        connection::Connection,
        mux::{Gateway, NetworkGateway},
        RecvBytes, SendBytes, SplitChannel,
    };

    async fn chat(c: &mut impl SplitChannel, text: &'static str) -> String {
        let text = String::from(text);
        let (s, r) = c.split();
        let (res, msg) = join!(s.send(&text), r.recv());
        res.unwrap();
        let msg = msg.unwrap();
        assert_eq!(text, msg);
        msg
    }

    // TODO: Better names for tests.

    #[tokio::test]
    async fn sunshine() {
        let (mut c1, mut c2) = Connection::in_memory();
        let p1 = async {
            let mut gateway = Gateway::multiplex(&mut c1, 3);
            let (mut m1, mut m2, mut m3) = gateway.drain(..).collect_tuple().unwrap();

            let s = async move {
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };

            let (s, _) = join!(s, gateway.inner.run());
            s
        };

        let p2 = async {
            let mut gateway = Gateway::multiplex(&mut c2, 3);
            let (mut m1, mut m2, mut m3) = gateway.drain(..).collect_tuple().unwrap();
            let s = async move {
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };
            let (s, _) = (s, gateway.inner.run()).join().await;
            s
        };

        let (s1, s2) = futures::join!(p1, p2);
        assert_eq!(s1, "Hello, how are you? Great!");
        assert_eq!(s2, "Hello, how are you? Great!");
    }

    #[tokio::test]
    async fn moonshine() {
        let (mut c1, c2) = Connection::in_memory();
        let p1 = async {
            let mut gateway = Gateway::multiplex(&mut c1, 3);
            let (mut m1, mut m2, mut m3) = gateway.drain(..).collect_tuple().unwrap();
            let h = async {
                // Wait a little such the errors get time to propagate
                tokio::time::sleep(Duration::from_millis(5)).await;
                let (s1, s2, s3) = futures::join!(
                    async { m1.send(&String::from("Are you there?")).await },
                    async { m2.recv::<()>().await },
                    async { m3.send(&String::from("Hello?")).await },
                );
                s1.expect_err("Should be closed");
                s2.expect_err("Should be closed");
                s3.expect_err("Should be closed");
            };
            join!(h, gateway.inner.run())
        };

        let p2 = async {
            drop(c2);
        };

        let (_, _) = futures::join!(p1, p2);
    }

    #[tokio::test]
    async fn network() {
        crate::testing::Cluster::new(3)
            .run(|mut net| async move {
                let (gateway, mut muxed) = NetworkGateway::multiplex(&mut net, 2);
                let (m1, m2) = muxed.drain(..).collect_tuple().unwrap();
                let h1 = tokio::spawn(async move {
                    let mut m = m1;
                    let res = m.symmetric_broadcast(String::from("Hello")).await.unwrap();
                    assert_eq!(res, vec!["Hello"; 3]);
                });
                let h2 = tokio::spawn(async move {
                    let mut m = m2;
                    let res = m.symmetric_broadcast(String::from("World")).await.unwrap();
                    assert_eq!(res, vec!["World"; 3]);
                });
                let (r1, r2, _) = futures::join!(h1, h2, gateway.drive());
                r1.unwrap();
                r2.unwrap();
            })
            .await
            .unwrap();
    }
}
