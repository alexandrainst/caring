use std::sync::Arc;

use futures::{
    channel::{mpsc, oneshot},
    future::{join_all, try_join_all},
    join, FutureExt, SinkExt, StreamExt,
};
use itertools::multiunzip;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::{
    help,
    net::{agency::{Broadcast, Unicast}, connection::ConnectionError, network::Network, Channel},
};


// TODO: Handle errors back in MuxConn
// TODO: Make it work over arbitrary Channel instead of Connection.

#[derive(Debug, Error)]
#[error(transparent)]
pub struct MuxError(Arc<ConnectionError>);

pub struct MuxConn {
    mailbox: mpsc::UnboundedReceiver<BytesMut>,
    gateway: mpsc::UnboundedSender<MultiplexedMessage>,
    error: oneshot::Receiver<MuxError>,
    id: usize,
}

struct MuxedSender {
    id: usize,
    gateway: mpsc::UnboundedSender<MultiplexedMessage>,
    error: oneshot::Receiver<MuxError>,
}

struct MuxedReceiver {
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

    fn to_bytes(self) -> Bytes {
        let bytes = self.0;
        let id = self.1;
        let mut msg  = BytesMut::new();
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

impl Channel for MuxConn {
    type Error = MuxError;

    async fn send<T: serde::Serialize + Sync>(&mut self, msg: &T) -> Result<(), Self::Error> {
        futures::select! {
            res = &mut self.error => {
                let err = res.expect("Gateway should always be alive if MuxConn is alive");
                Err(err)
            },
            default => {
                let msg = bincode::serialize(msg).unwrap().into();
                self.gateway
                    .send(MultiplexedMessage(msg, self.id))
                    .await
                    .unwrap();
                Ok(())
            },
        }
    }

    async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, Self::Error> {
        futures::select! {
            msg = self.mailbox.next() => {
                let buf = std::io::Cursor::new(msg.unwrap());
                let msg: T = bincode::deserialize_from(buf).unwrap();
                Ok(msg)
            },
            res = &mut self.error => {
                let err = res.expect("Should not be canceled");
                Err(err)
            }
        }
    }
}

pub struct GatewayInner<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    channel: super::connection::Connection<R, W>,
    mailboxes: Vec<mpsc::UnboundedSender<BytesMut>>,
    inbox: mpsc::UnboundedReceiver<MultiplexedMessage>,
    errors: Vec<oneshot::Sender<MuxError>>,
}

pub struct Gateway<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    handle: tokio::task::JoinHandle<GatewayInner<R, W>>,
}


impl<R, W> Gateway<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
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
    pub fn multiplex(con: super::connection::Connection<R, W>, n: usize) -> (Self, Vec<MuxConn>) {
        let (gateway, inbox) = mpsc::unbounded();

        let (sends, channels, errors) = multiunzip((0..n).map(|id| {
            let (send, recv) = mpsc::unbounded();
            let gateway = gateway.clone();
            let (error_coms, error) = oneshot::channel();

            let chan = MuxConn {
                mailbox: recv,
                error,
                gateway,
                id,
            };
            (send, chan, error_coms)
        }));

        let gateway = GatewayInner {
            mailboxes: sends,
            inbox,
            channel: con,
            errors,
        };

        let handle = tokio::spawn(gateway.run());

        let gateway = Self { handle };
        (gateway, channels)
    }

    pub async fn takedown(self) -> super::connection::Connection<R, W> {
        let gateway = self.handle.await.unwrap();
        let GatewayInner { channel, .. } = gateway;
        channel
    }
}

impl<R, W> GatewayInner<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    async fn run(self) -> Self {
        let mut gateway = self;
        {
            let (recving, sending) = gateway.channel.split();

            let send_out = async {
                while let Some(msg) = gateway.inbox.next().await {
                    sending.send_bytes(msg.to_bytes()).await.unwrap();
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
                    for c in gateway.errors.drain(..) {
                        // ignore dropped connections,
                        // they can't handle errors when they don't exist.
                        let _ = c.send(MuxError(err.clone()));
                    }
                },
            };
        }

        gateway
    }
}
// TODO: Multiplex generic channel
// requires that Channels can be split (introduce splittable trait?)
// probable needs to have a trait for sender and a trait for receiver.

// TODO: Multiplex Network
// Should be pretty easy, just do multiplexing for all the connections at once.

// NOTE: We could also mux the AsyncWrite/AsyncRead streams themselves.

struct NetworkGateway<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    gateways: Vec<Gateway<R, W>>,
    index: usize,
}

struct MuxNet {
    mux_conn: Vec<MuxConn>,
    index: usize,
}

impl<R, W> NetworkGateway<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    pub fn multiplex(net: Network<R, W>, n: usize) -> (Self, Vec<MuxNet>) {
        let mut gateways = Vec::new();
        let mut matrix = Vec::new();
        let index = net.index;
        for conn in net.connections.into_iter() {
            let (gateway, muxs) = Gateway::multiplex(conn, n);
            gateways.push(gateway);
            matrix.push(muxs);
        }
        let gateway = NetworkGateway { gateways, index };

        let matrix = help::transpose(matrix);
        let muxnets: Vec<_> = matrix
            .into_iter()
            .map(|mux_conn| MuxNet { mux_conn, index })
            .collect();

        (gateway, muxnets)
    }

    pub async fn takedown(self) -> Network<R, W> {
        let index = self.index;
        let iter = self.gateways.into_iter().map(|g| g.takedown());
        let res = join_all(iter).await;
        Network {
            connections: res,
            index,
        }
    }
}

impl Broadcast for MuxNet {
    type Error = MuxError;

    async fn broadcast(&mut self, msg: &impl serde::Serialize) -> Result<(), Self::Error> {
        let packet = bincode::serialize(msg).unwrap();
        let iter = self.mux_conn.iter_mut().map(|c| c.send(&packet));
        try_join_all(iter).await?;
        Ok(())
    }

    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, Self::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let packet = bincode::serialize(&msg).unwrap();
        let iter = self.mux_conn.iter_mut().map(|c| async {
            let mailbox = &mut c.mailbox;
            let gateway = &mut c.gateway;
            //let err = &mut c.error; // TODO: Handle errors
            let msg = MultiplexedMessage(packet.clone().into(), c.id);
            let success = join!(gateway.send(msg), mailbox.next());
            success
        });
        let resp: Result<Vec<T>, _> = join_all(iter)
            .await
            .into_iter()
            .map(|(_, s)| s.expect("Errors handling somewhere"))
            .map(|bytes| {
                let msg: Result<T, _> = bincode::deserialize(&bytes);
                msg
            })
            .collect();

        let mut resp =
            resp.map_err(|e| MuxError(Arc::new(ConnectionError::MalformedMessage(e))))?;
        resp.insert(self.index, msg);
        Ok(resp)
    }

    async fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> Result<T, Self::Error> {
        let idx = if idx > self.index { idx - 1 } else { idx };
        self.mux_conn[idx].recv().await
    }

    fn size(&self) -> usize {
        self.mux_conn.len() + 1
    }
}

impl Unicast for MuxNet {
    type Error = MuxError;

    async fn unicast(
        &mut self,
        msgs: &[impl serde::Serialize + Sync],
    ) -> Result<(), Self::Error> {
        let outgoing = self
            .mux_conn
            .iter_mut()
            .zip(msgs.iter())
            .map(|(conn, msg)| {
                conn.send(msg)
            });
        try_join_all(outgoing).await?;
        Ok(())
    }

    async fn symmetric_unicast<T>(
        &mut self,
        mut msgs: Vec<T>,
    ) -> Result<Vec<T>, Self::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Sync {
        let own = msgs.remove(self.index);
        let iter = self.mux_conn.iter_mut().zip(msgs.iter()).map(|(c,m)| async {
            let mailbox = &mut c.mailbox;
            let gateway = &mut c.gateway;
            //let err = &mut c.error; // TODO: Handle errors
            let packet = bincode::serialize(m).unwrap();
            let msg = MultiplexedMessage(packet.clone().into(), c.id);
            let success = join!(gateway.send(msg), mailbox.next());
            success
        });
        let resp: Result<Vec<T>, _> = join_all(iter)
            .await
            .into_iter()
            .map(|(_, s)| s.expect("Errors handling somewhere"))
            .map(|bytes| {
                let msg: Result<T, _> = bincode::deserialize(&bytes);
                msg
            })
            .collect();

        let mut resp =
            resp.map_err(|e| MuxError(Arc::new(ConnectionError::MalformedMessage(e))))?;
        resp.insert(self.index, own);
        Ok(resp)
    }

    async fn receive_all<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> Result<Vec<T>, Self::Error> {
        todo!()
    }

    fn size(&self) -> usize {
        todo!()
    }
}


#[cfg(test)]
mod test {
    use std::time::Duration;

    use itertools::Itertools;

    use crate::net::{connection::Connection, mux::Gateway, Channel};

    async fn chat(c: &mut impl Channel, text: &'static str) -> String {
        let text = String::from(text);
        c.send(&text).await.unwrap();
        let resp: String = c.recv().await.unwrap();
        assert_eq!(text, resp);
        resp
    }

    // TODO: Better names for tests.

    #[tokio::test]
    async fn sunshine() {
        let (c1, c2) = Connection::in_memory();
        let p1 = async {
            let (gateway, mut muxes) = Gateway::multiplex(c1, 3);

            let s = {
                let (mut m1, mut m2, mut m3) = muxes.drain(0..3).collect_tuple().unwrap();
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };
            gateway.takedown().await;
            s
        };

        let p2 = async {
            let (gateway, mut muxes) = Gateway::multiplex(c2, 3);
            let s = {
                let (mut m1, mut m2, mut m3) = muxes.drain(0..3).collect_tuple().unwrap();
                let (s1, s2, s3) = futures::join!(
                    chat(&mut m1, "Hello, "),
                    chat(&mut m2, "how are you? "),
                    chat(&mut m3, "Great!"),
                );
                s1 + &s2 + &s3
            };
            gateway.takedown().await;
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
            {
                let (mut m1, mut m2, mut m3) = muxes.drain(0..3).collect_tuple().unwrap();
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
            gateway.takedown().await;
        };

        let p2 = async {
            drop(c2);
        };

        let (_, _) = futures::join!(p1, p2);
    }
}
