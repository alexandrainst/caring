use std::sync::Arc;
use std::{error::Error, vec::Drain};

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
    net::{
        connection::{RecvBytes, SendBytes},
        network::Network,
        Channel, SplitChannel,
    },
};

#[derive(Debug, Error)]
#[error(transparent)]
pub struct MuxError(Arc<dyn Error + Send + Sync + 'static>);

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

impl super::connection::SendBytes for MuxedSender {
    type SendError = MuxError;

    async fn send_bytes(&mut self, bytes: tokio_util::bytes::Bytes) -> Result<(), Self::SendError> {
        futures::select! {
            res = &mut self.error => {
                let err = res.expect("Gateway should always be alive if MuxConn is alive");
                Err(err)
            },
            default => {
                self.gateway
                    .send(MultiplexedMessage(bytes, self.id))
                    .await
                    .unwrap();
                Ok(())
            },
        }
    }
}

use tokio_util::bytes::{Buf, BufMut, Bytes, BytesMut};
impl super::connection::RecvBytes for MuxedReceiver {
    type RecvError = MuxError;

    async fn recv_bytes(&mut self) -> Result<tokio_util::bytes::BytesMut, Self::RecvError> {
        futures::select! {
            msg = self.mailbox.next() => {
                let msg = msg.expect("Should not be empty");
                Ok(msg)
            },
            res = &mut self.error => {
                let err = res.expect("Should not be canceled");
                Err(err)
            }
        }
    }
}

pub struct MuxConn(MuxedSender, MuxedReceiver);

impl Channel for MuxConn {
    type Error = MuxError;

    fn send<T: serde::Serialize + Sync>(
        &mut self,
        msg: &T,
    ) -> impl futures::prelude::Future<Output = Result<(), Self::Error>> + Send {
        self.0.send_thing(msg)
    }

    fn recv<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> impl futures::prelude::Future<Output = Result<T, Self::Error>> + Send {
        self.1.recv_thing()
    }
}

impl SplitChannel for MuxConn {
    type Sender = MuxedSender;
    type Receiver = MuxedReceiver;

    fn split(&mut self) -> (&mut Self::Sender, &mut Self::Receiver) {
        (&mut self.0, &mut self.1)
    }
}

pub struct GatewayInner<'a, C>
where
    C: SplitChannel,
{
    channel: &'a mut C,
    mailboxes: Vec<mpsc::UnboundedSender<BytesMut>>,
    inbox: mpsc::UnboundedReceiver<MultiplexedMessage>,
    errors: Vec<[oneshot::Sender<MuxError>; 2]>,
}

pub struct Gateway<'a, C: SplitChannel> {
    //handle: tokio::task::JoinHandle<GatewayInner<&'a mut C>>,
    handle: GatewayInner<'a, C>,
    muxes: Vec<MuxConn>,
}

impl<'a, C: SplitChannel + Send + 'static> Gateway<'a, C> {
    /// Multiplex a connection to share it into `n` new connections.
    ///
    /// * `net`: Connection to use as a gateway for multiplexing
    /// * `n`: Number of new connections to multiplex into
    ///
    /// Returns a gateway which the MuxConn communicate through, along with the MuxConn
    ///
    /// # Example
    /// ```
    /// # use caring::net::connection::Connection;
    /// # use caring::net::mux::Gateway;
    /// # use crate::caring::net::Channel;
    /// # tokio_test::block_on(async {
    /// # let (c1, c2) = Connection::in_memory();
    /// # let first = async {
    /// # let con =c1;
    /// let (gateway, mut muxs) = Gateway::multiplex(con, 2);
    /// let mut m1 = muxs.remove(1);
    /// let mut m2 = muxs.remove(0);
    /// let t1 = async move {
    ///     m1.send(&String::from("Hello")).await.unwrap();
    /// };
    /// let t2 = async move {
    ///     m2.send(&String::from("Friend")).await.unwrap();
    /// };
    /// futures::join!(t1, t2);
    /// let con : Connection<_,_> = gateway.takedown().await;
    /// # };
    /// #
    /// # let second = async {
    /// # let con = c2;
    /// # let (gateway, mut muxs) = Gateway::multiplex(con, 2);
    /// # let mut m1 = muxs.remove(1);
    /// # let mut m2 = muxs.remove(0);
    /// # let t1 = async move {
    /// #     let _ : String = m1.recv().await.unwrap();
    /// # };
    /// # let t2 = async move {
    /// #     let _ : String = m2.recv().await.unwrap();
    /// # };
    /// # futures::join!(t1, t2);
    /// # let con : Connection<_,_> = gateway.takedown().await;
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

        let gateway = GatewayInner {
            mailboxes: sends,
            inbox,
            channel: con,
            errors,
        };

        let handle = gateway;
        //let handle = tokio::spawn(gateway.run());

        Self {
            handle,
            muxes: channels,
        }
    }

    pub async fn for_each<T, F: Future<Output = T>>(
        self,
        func: impl FnMut(MuxConn) -> F,
    ) -> Vec<T> {
        let res = join_all(self.muxes.into_iter().map(func));
        let (res, _) = join!(res, self.handle.run());
        res
    }

    pub fn drain(&mut self) -> Drain<MuxConn> {
        self.muxes.drain(..)
    }

    //pub async fn takedown(self) -> C {
    //    let gateway = self.handle.await.unwrap();
    //    let GatewayInner { channel, .. } = gateway;
    //    channel
    //}
}

impl<'a, C: SplitChannel + Send> GatewayInner<'a, C> {
    async fn run(self) -> Self {
        let mut gateway = self;
        {
            let (sending, recving) = gateway.channel.split();

            let send_out = async {
                while let Some(msg) = gateway.inbox.next().await {
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
            futures::select! {
                () = send_out => {},
                err = recv_in => {
                    let err = Arc::new(err);
                    for [c1, c2] in gateway.errors.drain(..) {
                        // ignore dropped connections,
                        // they can't handle errors when they don't exist.
                        let _ = c1.send(MuxError(err.clone()));
                        let _ = c2.send(MuxError(err.clone()));
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
    // We should really borrow instead.
    pub fn multiplex(net: &'a mut Network<C>, n: usize) -> (Self, Vec<MuxNet>) {
        let mut gateways = Vec::new();
        let mut matrix = Vec::new();
        let index = net.index;
        for conn in net.connections.iter_mut() {
            let mut gateway = Gateway::multiplex(conn, n);
            let muxes = gateway.drain().collect_vec();
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

    //pub async fn takedown(self) -> Network<C> {
    //    let index = self.index;
    //    let iter = self.gateways.into_iter().map(|g| g.takedown());
    //    let res = join_all(iter).await;
    //    Network {
    //        connections: res,
    //        index,
    //    }
    //}

    pub async fn run(self) {
        let _ = join_all(self.gateways.into_iter().map(|c| c.handle.run())).await;
    }
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use itertools::Itertools;
    use tokio::join;

    use crate::net::{
        connection::Connection,
        mux::{Gateway, NetworkGateway},
        Channel, RecvBytes, SendBytes, SplitChannel,
    };

    async fn chat(c: &mut impl SplitChannel, text: &'static str) -> String {
        let text = String::from(text);
        let (s, r) = c.split();
        let (res, msg) = join!(s.send_thing(&text), r.recv_thing());
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
            let (mut m1, mut m2, mut m3) = gateway.drain().collect_tuple().unwrap();

            let s = async move {
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };

            let s = s;
            let (s, _) = join!(s, gateway.handle.run());
            s
        };

        let p2 = async {
            let mut gateway = Gateway::multiplex(&mut c2, 3);
            let (mut m1, mut m2, mut m3) = gateway.drain().collect_tuple().unwrap();
            let s = async move {
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };
            let s = s; //tokio::spawn(s);
            let (s, _) = join!(s, gateway.handle.run());
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
            let (mut m1, mut m2, mut m3) = gateway.drain().collect_tuple().unwrap();
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
            join!(h, gateway.handle.run())
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
                let (r1, r2, _) = futures::join!(h1, h2, gateway.run());
                r1.unwrap();
                r2.unwrap();
            })
            .await
            .unwrap();
    }
}
