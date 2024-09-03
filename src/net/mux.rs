//! Multiplexing Connections and Networks
//!
//! This moudle provides tools to multiplex channels/connections and networks
//! in order to run multiple protocols concurrently.

// TODO: Add Multiplex trait with impls for Network and SplitChannel?

use std::error::Error;
use std::sync::Arc;

use futures_concurrency::prelude::*;
use tracing::Instrument;

use futures::future::try_join_all;
use num_traits::ToPrimitive;
use thiserror::Error;
use tokio::sync::{
    mpsc::{self, unbounded_channel, UnboundedSender},
    oneshot,
};
use tokio_util::bytes::{Buf, BufMut, Bytes, BytesMut};

use crate::{
    help,
    net::{network::Network, Channel, RecvBytes, SendBytes, SplitChannel, Tuneable},
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
        msg.put_u32(id.to_u32().expect("Too many multiplexed connections!"));
        msg.put(bytes);
        msg.freeze()
    }
}

impl SendBytes for MuxedSender {
    type SendError = MuxError;

    async fn send_bytes(&mut self, bytes: tokio_util::bytes::Bytes) -> Result<(), Self::SendError> {
        if let Ok(err) = self.error.try_recv() {
            return Err(err);
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

/// Multiplexed Connection
///
/// Aqquirred by constructing a [Gateway] using [`Gateway::single`], [`Gateway::multiplex`],
/// [`Gateway::multiplex_array`] or [`Gateway::muxify`] on an existing gateway.
///
/// Errors are propogated from the underlying connection inside the gateway.
///
/// The gateway needs to be driven for the muxed connection to function using [`Gateway::drive`].
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

impl MuxConn {
    pub async fn shutdown(mut self) {
        let _ = (self.0.send(&()), self.1.recv::<()>()).join().await;
    }
}

/// # Multiplexed Gateway Channel
///
/// Enables splitting a channel into multiple multiplexed channels.
/// The multiplexed channels must be *driven* by the gateway
/// (see [``Gateway::drive``]) otherwise the multiplexed channels won't
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
///     gateway.drive().await.unwrap();
/// });
///
/// tokio::spawn(async {// party 2
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
///     gateway.drive().await.unwrap();
/// });
/// # })
/// ```
pub struct Gateway<C>
where
    C: SplitChannel,
{
    channel: C,
    mailboxes: Vec<mpsc::UnboundedSender<BytesMut>>,
    inbox: mpsc::UnboundedReceiver<MultiplexedMessage>,
    errors: Vec<[oneshot::Sender<MuxError>; 2]>,
    outbox: mpsc::WeakUnboundedSender<MultiplexedMessage>,
}

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("Multiplexed connection {0} not found: {1}")]
    MailboxNotFound(usize, &'static str),
    #[error("Underlying connection died: {0}")]
    DeadConnection(#[from] Arc<dyn Error + Send + Sync + 'static>),
}

impl<C: SplitChannel + Send> Gateway<C> {
    /// Drive a gateway until all multiplexed connections are complete
    ///
    /// # Errors
    ///
    /// - [`GatewayError::MailboxNotFound`] if a given multiplexed connections has been
    ///   dropped and is receiving messages.
    /// - [`GatewayError::DeadConnection`] if the underlying connection have failed.
    ///
    pub async fn drive(mut self) -> Result<Self, GatewayError> {
        // TODO: maybe have this be nonconsuming so it can be resumed after new muxes are added?
        // This however would compromise the possible destruction when errors occur,
        // thus leaving the error handling in a bad state.
        let (sending, recving) = self.channel.split();
        let send_out = async {
            loop {
                if let Some(msg) = self.inbox.recv().await {
                    match sending.send_bytes(msg.make_bytes()).await {
                        Ok(()) => continue,
                        Err(e) => break Err(e),
                    }
                }
                break Ok(());
            }
        };

        let recv_in = async {
            loop {
                match recving.recv_bytes().await {
                    Ok(mut msg) => {
                        let id = msg.get_u32() as usize;
                        let bytes = msg;
                        let Some(mailbox) = self.mailboxes.get_mut(id) else {
                            break Err(GatewayError::MailboxNotFound(
                                id,
                                "Mux never existed, bad format?",
                            ));
                        };
                        let Ok(()) = mailbox.send(bytes) else {
                            break Err(GatewayError::MailboxNotFound(
                                id,
                                "Receiving mux have died",
                            ));
                        };
                    }
                    Err(e) => break Ok(e),
                }
            }
        };

        tokio::select! { // Drive both futures to completion.
            res = send_out => {
                match res {
                    Ok(()) => {
                        Ok(self)
                    },
                    Err(err) => {
                        tracing::error!("Failed to send message: {err}");
                        Err(self.propogate_error(err))
                    }
                }
            },
            err = recv_in => {
                let err : C::Error = err?; // return early on missing mailbox.
                tracing::error!("Failed to receive message: {err}");
                Err(self.propogate_error(err))
            },
        }
    }

    fn propogate_error<E: Error + Send + Sync + 'static>(mut self, err: E) -> GatewayError {
        let err = Arc::new(err);
        for [c1, c2] in self.errors.drain(..) {
            // ignore dropped connections,
            // they can't handle errors when they don't exist.
            let _ = c1.send(MuxError::Connection(err.clone()));
            let _ = c2.send(MuxError::Connection(err.clone()));
        }
        GatewayError::DeadConnection(err)
    }

    fn new(channel: C) -> (Self, UnboundedSender<MultiplexedMessage>) {
        let (outbox, inbox) = unbounded_channel();
        let link = outbox.clone(); // needs to kept alive
        let outbox = outbox.downgrade();
        let gateway = Self {
            channel,
            mailboxes: vec![],
            errors: vec![],
            inbox,
            outbox,
        };
        (gateway, link)
    }

    /// Multiplex a channel to a single new muxed connection.
    ///
    /// New muxed connections can be constructed using [`Gateway::muxify`].
    pub fn single(channel: C) -> (Self, MuxConn) {
        let (mut gateway, link) = Self::new(channel);
        let con = gateway.add_mux(link);
        (gateway, con)
    }

    pub fn destroy(self) -> C {
        self.channel
    }

    /// Multiplex a channel to share it into `n` new connections.
    ///
    /// * `con`: Connection to use as a gateway for multiplexing
    /// * `n`: Number of new connections to multiplex into
    ///
    /// Returns a gateway which the [`MuxConn`] communicate through, along with the [`MuxConn`]'s
    #[must_use]
    pub fn multiplex(con: C, n: usize) -> (Self, Vec<MuxConn>) {
        tracing::debug!("Multiplexing Connnection into {n}");
        let (mut gateway, link) = Self::new(con);
        let muxes: Vec<_> = (0..n).map(|_| gateway.add_mux(link.clone())).collect();
        (gateway, muxes)
    }

    /// Multiplex a channel to share into `N` new connections
    ///
    /// * `con`: connection to use
    ///
    /// Returns a gateway which the [`MuxConn`] communicate through, along with the [`MuxConn`]'s
    #[must_use]
    pub fn multiplex_array<const N: usize>(con: C) -> (Self, [MuxConn; N]) {
        let (mut gateway, link) = Self::new(con);
        let muxes = std::array::from_fn(|_| gateway.add_mux(link.clone()));
        (gateway, muxes)
    }

    fn add_mux(&mut self, gateway: UnboundedSender<MultiplexedMessage>) -> MuxConn {
        let id = self.mailboxes.len();
        let (errors_coms1, error) = oneshot::channel();
        let sender = MuxedSender { id, gateway, error };
        let (outbox, mailbox) = tokio::sync::mpsc::unbounded_channel();
        let (errors_coms2, error) = oneshot::channel();
        let receiver = MuxedReceiver { id, mailbox, error };
        self.errors.push([errors_coms1, errors_coms2]);
        self.mailboxes.push(outbox);
        MuxConn(sender, receiver)
    }

    /// Add a new muxed connection
    pub fn muxify(&mut self) -> MuxConn {
        let gateway = self
            .outbox
            .clone()
            .upgrade()
            .expect("We are holding the receiver");
        self.add_mux(gateway)
    }
}

pub struct ActiveGateway<C: SplitChannel>(
    tokio::task::JoinHandle<Result<Gateway<C>, GatewayError>>,
);

impl<C: SplitChannel + Send + 'static> Gateway<C> {
    pub fn go(self) -> ActiveGateway<C> {
        ActiveGateway(tokio::spawn(self.drive()))
    }
}

impl<C: SplitChannel + Send + 'static> ActiveGateway<C> {
    pub async fn deactivate(self) -> Result<Gateway<C>, GatewayError> {
        self.0.await.unwrap()
    }
}

pub struct NetworkGateway<C: SplitChannel> {
    gateways: Vec<Gateway<C>>,
    index: usize,
}

pub type MultiplexedNetwork = Network<MuxConn>;

impl MultiplexedNetwork {
    pub async fn shutdown(self) {
        let _: Vec<_> = self
            .connections
            .into_co_stream()
            .map(|con| con.shutdown())
            .collect()
            .await;
    }
}

impl<C> NetworkGateway<C>
where
    C: SplitChannel + Send,
{
    #[must_use]
    pub fn multiplex(net: Network<C>, n: usize) -> (NetworkGateway<C>, Vec<MultiplexedNetwork>) {
        tracing::debug!("Multiplexing Network into {n}");
        let mut gateways = Vec::new();
        let mut matrix = Vec::new();
        let index = net.index;
        for conn in net.connections {
            let (gateway, muxes) = Gateway::multiplex(conn, n);
            matrix.push(muxes);
            gateways.push(gateway);
        }
        let gateway = NetworkGateway { gateways, index };

        let matrix = help::transpose(matrix);
        let muxnets: Vec<_> = matrix
            .into_iter()
            .map(|connections| MultiplexedNetwork { connections, index })
            .collect();

        tracing::debug!("Network successfully multiplexed");
        (gateway, muxnets)
    }

    pub fn multiplex_borrow(
        net: &mut Network<C>,
        n: usize,
    ) -> (NetworkGateway<&mut C>, Vec<MultiplexedNetwork>) {
        let net = net.as_mut();
        NetworkGateway::<&mut C>::multiplex(net, n)
    }

    pub fn multiplexify<'tun, T>(
        tuneable: &'tun mut T,
        n: usize,
    ) -> (NetworkGateway<&'tun mut C>, Vec<Network<MuxConn>>)
    where
        T: Tuneable<Channel = C>,
    {
        let index = tuneable.id().0;
        let channels: &'tun mut [C] = tuneable.channels();
        let connections: Vec<&'tun mut C> = channels.iter_mut().collect();
        let network = Network { connections, index };
        NetworkGateway::<&'tun mut C>::multiplex(network, n)
    }

    pub async fn drive(self) -> Result<Self, GatewayError> {
        let connections = self.gateways.len();
        let gateways = try_join_all(self.gateways.into_iter().map(Gateway::drive))
            .instrument(tracing::debug_span!("Driving gateway", connections))
            .await?;
        Ok(Self {
            gateways,
            index: self.index,
        })
    }

    #[must_use]
    pub fn destroy(mut self) -> Network<C> {
        let index = self.index;
        let connections: Vec<_> = self.gateways.drain(..).map(Gateway::destroy).collect();
        Network { connections, index }
    }

    pub fn new_mux(&mut self) -> MultiplexedNetwork {
        let connections = self.gateways.iter_mut().map(Gateway::muxify).collect();
        MultiplexedNetwork {
            connections,
            index: self.index,
        }
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

            let (s, gateway) = join!(s, gateway.drive());
            let mut gateway = gateway.unwrap();
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
            let (s, gateway) = (s, gateway.drive()).join().await;
            let mut gateway = gateway.unwrap();
            let _: String = gateway.channel.recv().await.unwrap();
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

        let (_, ()) = futures::join!(p1, p2);
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
                let gateway = gateway.unwrap();
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

    #[tokio::test]
    async fn network_borrowed_no_tasks() {
        let res = crate::testing::Cluster::new(3)
            .run(|mut net| async move {
                let net_ref = net.as_mut();
                let (gateway, mut muxed) = NetworkGateway::multiplex(net_ref, 2);
                let (m1, m2) = muxed.drain(..).collect_tuple().unwrap();
                let h1 = async move {
                    let mut m = m1;
                    let res = m.symmetric_broadcast(String::from("Hello")).await.unwrap();
                    assert_eq!(res, vec!["Hello"; 3]);
                    true
                };
                let h2 = async move {
                    let mut m = m2;
                    let res = m.symmetric_broadcast(String::from("World")).await.unwrap();
                    assert_eq!(res, vec!["World"; 3]);
                    true
                };
                let (r1, r2, _) = futures::join!(h1, h2, gateway.drive());
                net.shutdown().await.unwrap();
                r1 && r2
            })
            .await
            .unwrap();

        assert!(res.into_iter().all(|x| x));
    }
}
