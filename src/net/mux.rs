use futures::{channel::mpsc, FutureExt, SinkExt, StreamExt};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncWrite};

use crate::net::Channel;

type Bytes = Vec<u8>;

// TODO: Handle errors back in MuxConn
// TODO: Make it work over arbitrary Channel instead of Connection.

pub struct MuxConn {
    mailbox: mpsc::UnboundedReceiver<Bytes>,
    gateway: mpsc::UnboundedSender<MultiplexedMessage>,
    id: usize,
}

#[derive(Debug, Error)]
#[error("Need better errors")]
pub struct MuxError;

impl Channel for MuxConn {
    type Error = MuxError;

    async fn send<T: serde::Serialize + Sync>(&mut self, msg: &T) -> Result<(), Self::Error> {
        let msg = bincode::serialize(msg).unwrap();
        self.gateway
            .send(MultiplexedMessage(msg, self.id))
            .await
            .unwrap();
        Ok(())
    }

    async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> Result<T, Self::Error> {
        let res = self.mailbox.next().await.unwrap();
        let buf = std::io::Cursor::new(res);
        let msg: T = bincode::deserialize_from(buf).unwrap();
        Ok(msg)
    }
}

pub struct GatewayInner<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    channel: super::connection::Connection<R, W>,
    mailboxes: Vec<mpsc::UnboundedSender<Bytes>>,
    inbox: mpsc::UnboundedReceiver<MultiplexedMessage>,
}

pub struct Gateway<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    handle: tokio::task::JoinHandle<GatewayInner<R, W>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct MultiplexedMessage(Bytes, usize);

impl<R, W> Gateway<R, W>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    pub fn multiplex(net: super::connection::Connection<R, W>, n: usize) -> (Self, Vec<MuxConn>) {
        let (gateway, inbox) = mpsc::unbounded();

        let (sends, channels) = (0..n)
            .map(|id| {
                let (send, recv) = mpsc::unbounded();
                let gateway = gateway.clone();

                let chan = MuxConn {
                    mailbox: recv,
                    gateway,
                    id,
                };
                (send, chan)
            })
            .unzip();

        let gateway = GatewayInner {
            mailboxes: sends,
            inbox,
            channel: net,
        };

        let handle = tokio::spawn(gateway.run());

        let gateway = Self {
            handle,
        };
        (gateway, channels)
    }


    pub async fn takedown(self) -> super::connection::Connection<R, W> {
        let gateway = self.handle.await.unwrap();
        let GatewayInner { channel, mailboxes, inbox } = gateway;
        drop(inbox);
        drop(mailboxes);
        channel
    }
}


impl<R,W> GatewayInner<R,W>
where 
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    async fn run(self) -> Self {
        let mut gateway = self;
        {
            let (mut recving, mut sending) = gateway.channel.split();

            let send_out = async {
                while let Some(msg) = gateway.inbox.next().await {
                    sending.send(&msg).await.unwrap();
                }
            }.fuse();

            let recv_in = async {
                while let Ok(msg) = recving.recv().await {
                    let MultiplexedMessage(bytes, id) = msg;
                    gateway.mailboxes[id].send(bytes).await.unwrap();
                }
            }.fuse();

            futures::pin_mut!(send_out, recv_in);

            futures::select! {
                () = send_out => {},
                () = recv_in => {},
            };
        }

        gateway
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::net::{connection::Connection, mux::Gateway, Channel};

    async fn chat(c: &mut impl Channel, text: &'static str) -> String {
        let text = String::from(text);
        c.send(&text).await.unwrap();
        let resp : String = c.recv().await.unwrap();
        assert_eq!(text, resp);
        resp
    }

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
}
