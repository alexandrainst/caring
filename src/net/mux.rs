//! Multiplexing Connections and Networks
//!
//! This moudle provides tools to multiplex channels/connections and networks
//! in order to run multiple protocols concurrently.

// TODO: Add Multiplex trait with impls for Network and SplitChannel?

use std::error::Error;
use std::sync::Arc;


use futures::future::join_all;
use thiserror::Error;
use tokio::sync::{mpsc::{self, unbounded_channel, UnboundedSender}, oneshot};
use tokio_util::bytes::{Buf, BufMut, Bytes, BytesMut};

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
        if let Ok(err) = self.error.try_recv() {
            return Err(err)
        };

        self.gateway
            .send(MultiplexedMessage(bytes, self.id))
            .map_err(|_| MuxError::DeadGateway)
    }
}

impl RecvBytes for MuxedReceiver {
    type RecvError = MuxError;

    async fn recv_bytes(&mut self) -> Result<tokio_util::bytes::BytesMut, Self::RecvError> {
        tokio::select! {
            msg = self.mailbox.recv() => {
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

/// # Multiplexed Gateway Channel
///
/// Enables splitting a channel into multiple multiplexed channels.
/// The multiplexed channels must be *driven* by the gateway
/// (see [Gateway::drive]) otherwise the multiplexed channels won't
/// be able to communicate.
/// 
/// ## Example:
/// ```
/// use caring::net::{connection::Connection, mux::Gateway, RecvBytes, SendBytes};
/// # tokio_test::block_on(async {
/// let (c1,c2) = Connection::in_memory();
///
/// tokio::spawn(async {// party 1
///     let con = c1;
///     let (mut gateway, mut m1) = Gateway::single(con);
///     let mut m2 = gateway.muxify();
///     tokio::spawn(async move {
///         m1.send(&"Hello MUX1!".to_owned()).await.unwrap();
///     });
///     tokio::spawn(async move {
///         m2.send(&"Hello MUX2!".to_owned()).await.unwrap();
///     });
///     gateway.drive().await;
/// });
///
/// tokio::spawn( async {// party 2
///     let con = c2;
///     let (mut gateway, mut m1) = Gateway::single(con);
///     let mut m2 = gateway.muxify();
///     tokio::spawn(async move {
///         let msg : String = m1.recv().await.unwrap();
///         assert_eq!(msg, "Hello MUX1!");
///     });
///     tokio::spawn(async move {
///         let msg : String = m2.recv().await.unwrap();
///         assert_eq!(msg, "Hello MUX2!");
///     });
///     gateway.drive().await;
/// });
/// })
/// ```
pub struct Gateway<C>
where
    C: SplitChannel,
{
    channel: C,
    mailboxes: Vec<mpsc::UnboundedSender<BytesMut>>,
    inbox: mpsc::UnboundedReceiver<MultiplexedMessage>,
    errors: Vec<[oneshot::Sender<MuxError>; 2]>,
    outbox: mpsc::WeakUnboundedSender<MultiplexedMessage>
}

impl<C: SplitChannel + Send> Gateway<C> {
    pub async fn drive(self) -> Self {
        let mut gateway = self;
        {
            let (sending, recving) = gateway.channel.split();

            let send_out = async {
                while let Some(msg) = gateway.inbox.recv().await {
                    // TODO: Error propagation.
                    sending.send_bytes(msg.make_bytes()).await.unwrap();
                }
            };

            let recv_in = async {
                loop {
                    match recving.recv_bytes().await {
                        Ok(mut msg) => {
                            let id = msg.get_u32() as usize;
                            let bytes = msg;
                            gateway.mailboxes[id].send(bytes).unwrap();
                        }
                        Err(e) => break e,
                    }
                }
            };

            tokio::select! { // Drive both futures to completion.
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

    pub fn single(channel: C) -> (Self, MuxConn) {
        let (outbox, inbox) = unbounded_channel();
        let gateway = outbox.clone();
        let outbox= outbox.downgrade();
        let mut new = Self {
            channel,
            mailboxes: vec![],
            errors: vec![],
            inbox,
            outbox,
        };
        let con = new.add_mux(gateway);
        (new, con)

    }

    pub fn destroy(self) -> C {
        self.channel
    }

    /// Multiplex a channel to share it into `n` new connections.
    ///
    /// * `net`: Connection to use as a gateway for multiplexing
    /// * `n`: Number of new connections to multiplex into
    ///
    /// Returns a gateway which the MuxConn communicate through, along with the MuxConn
     pub fn multiplex(con: C, n: usize) -> (Self, Vec<MuxConn>) {
        let (mut gateway, con) =  Self::single(con);
        let mut muxes = vec![con];
        for _ in 1..n {
            muxes.push(gateway.muxify());
        }
        (gateway, muxes)
    }


    fn add_mux(&mut self, gateway: UnboundedSender<MultiplexedMessage>) -> MuxConn {
        let id = self.mailboxes.len();
        let (errors_coms1, error) =  oneshot::channel();
        let mx_sender = MuxedSender {
            id,
            gateway,
            error,
        };
        let (outbox, mailbox) = tokio::sync::mpsc::unbounded_channel();
        let (errors_coms2, error) =  oneshot::channel();
        let mx_receiver = MuxedReceiver {
            id,
            mailbox,
            error,
        };
        self.errors.push([errors_coms1, errors_coms2]);
        self.mailboxes.push(outbox);
        MuxConn(mx_sender, mx_receiver)
    }

    pub fn muxify(&mut self) -> MuxConn {
        let gateway = self.outbox.clone().upgrade().expect("We are holding the receiver");
        self.add_mux(gateway)
    }
}

pub struct NetworkGateway<C: SplitChannel> {
    gateways: Vec<Gateway<C>>,
    index: usize,
}

type MuxNet = Network<MuxConn>;

impl<C> NetworkGateway<C>
where
    C: SplitChannel + Send,
{
    pub fn multiplex(net: Network<C>, n: usize) -> (NetworkGateway<C>, Vec<MuxNet>) {
        let mut gateways = Vec::new();
        let mut matrix = Vec::new();
        let index = net.index;
        for conn in net.connections {
            let (gateway, muxes) = Gateway::multiplex(conn, n);
            matrix.push(muxes);
            gateways.push(gateway);
        }
        let gateway = NetworkGateway { gateways, index, };

        let matrix = help::transpose(matrix);
        let muxnets: Vec<_> = matrix
            .into_iter()
            .map(|connections| MuxNet { connections, index })
            .collect();

        (gateway, muxnets)
    }

    pub fn multiplex_borrow(net: &mut Network<C>, n: usize)
        -> (NetworkGateway<&mut C>, Vec<MuxNet>)
    {
        let net = net.as_mut();
        NetworkGateway::<&mut C>::multiplex(net, n)
    }

    pub async fn drive(self) -> Self {
        let gateways = join_all(self.gateways.into_iter().map(|c| c.drive())).await;
        Self { gateways, index: self.index }
    }

    pub fn destroy(mut self) -> Network<C> {
        let index= self.index;
        let connections : Vec<_> = self.gateways.drain(..).map(|g| g.destroy()).collect();
        Network { connections, index }
    }

    pub fn new_mux(&mut self) -> MuxNet {
        let connections = self.gateways.iter_mut().map(|g| g.muxify() ).collect();
        MuxNet { connections, index: self.index }
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

    #[tokio::test]
    async fn sunshine() {
        let (c1, c2) = Connection::in_memory();
        let p1 = async {
            let (gateway, mut muxes) = Gateway::multiplex(c1, 3);
            let (mut m1, mut m2, mut m3) = muxes.drain(..).collect_tuple().unwrap();

            let s = async move {
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };

            let (s, mut gateway) = join!(s, gateway.drive());
            gateway.channel.send(&"bye".to_owned()).await.unwrap();
            gateway.channel.shutdown().await.unwrap();
            s
        };

        let p2 = async {
            let (gateway, mut muxes) = Gateway::multiplex(c2, 3);
            let (mut m1, mut m2, mut m3) = muxes.drain(..).collect_tuple().unwrap();
            let s = async move {
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };
            let (s, mut gateway) = (s, gateway.drive()).join().await;
            let _ : String = gateway.channel.recv().await.unwrap();
            gateway.channel.shutdown().await.unwrap();
            s
        };

        let (s1, s2) = futures::join!(p1, p2);
        assert_eq!(s1, "Hello, how are you? Great!");
        assert_eq!(s2, "Hello, how are you? Great!");
    }

    #[tokio::test]
    async fn moonshine() {
        let (c1, c2) = Connection::in_memory();
        let p1 = async {
            let (gateway, mut muxes) = Gateway::multiplex(c1, 3);
            let (mut m1, mut m2, mut m3) = muxes.drain(..).collect_tuple().unwrap();
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
            join!(h, gateway.drive())
        };

        let p2 = async {
            drop(c2);
        };

        let (_, _) = futures::join!(p1, p2);
    }

    #[tokio::test]
    async fn network() {
        crate::testing::Cluster::new(3)
            .run(|net| async move {
                let (gateway, mut muxed) = NetworkGateway::multiplex(net, 2);
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
                let (r1, r2, gateway) = futures::join!(h1, h2, gateway.drive());
                gateway.destroy().shutdown().await.unwrap();
                r1.unwrap();
                r2.unwrap();
            })
            .await
            .unwrap();
    }


    #[tokio::test]
    async fn network_borrowed() {
        crate::testing::Cluster::new(3)
            .run(|mut net| async move {
                let net_ref = net.as_mut();
                let (gateway, mut muxed) = NetworkGateway::multiplex(net_ref, 2);
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
                net.shutdown().await.unwrap();
                r1.unwrap();
                r2.unwrap();
            })
            .await
            .unwrap();
    }
}
