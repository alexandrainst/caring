use std::{collections::BTreeMap, marker::PhantomData, io::Cursor, sync::{Arc, Mutex, RwLock}};

use futures::{future, SinkExt, StreamExt};
use tokio::{io::{AsyncReadExt, AsyncWrite, AsyncRead, DuplexStream, ReadHalf, WriteHalf}, sync::{mpsc::Sender}, net::{TcpStream, tcp::{OwnedReadHalf, OwnedWriteHalf}}};
use tokio_util::codec::{FramedRead, LengthDelimitedCodec, FramedWrite};


pub struct Connection<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> {
    task: tokio::task::JoinHandle<()>,
    input: tokio::sync::mpsc::Sender<Box<[u8]>>,
    reader: FramedRead<R, LengthDelimitedCodec>,
    phantom: PhantomData<W> // Not needed, but nice
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin + Send + 'static> Connection<R,W> {
    pub fn new(reader: R, writer: W) -> Self {
        let (input, mut outgoing) : (Sender<Box<[u8]>>, _) = tokio::sync::mpsc::channel(8);
        let codec = LengthDelimitedCodec::new();
        let reader = FramedRead::new(reader, codec.clone());
        let mut writer = FramedWrite::new(writer, codec);

        let task = tokio::spawn(async move {
            while let Some(msg) = outgoing.recv().await {
                writer.send(msg.into()).await.unwrap();
            }
        });
        Connection {task, input, reader, phantom: PhantomData}
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Connection<R,W> {
    pub fn send(&self, msg: &impl serde::Serialize) {
        let msg = bincode::serialize(msg).unwrap();
        self.input.try_send(msg.into()).unwrap();
    }

    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> T {
        let buf = self.reader.next().await.unwrap().unwrap();
        let buf = std::io::Cursor::new(buf);
        bincode::deserialize_from(buf).unwrap()
    }
}


impl Connection<OwnedReadHalf, OwnedWriteHalf> {
    pub fn from_tcp(stream: TcpStream) -> Self {
        let (reader, writer) = stream.into_split();
        Self::new(reader, writer)
    }
}


impl Connection<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>> {
    pub fn in_memory() -> (Self, Self) {
        let (s1, s2) = tokio::io::duplex(64);

        let (r1, w1) = tokio::io::split(s1);
        let (r2, w2) = tokio::io::split(s2);

        (Self::new(r1, w2), Self::new(r2, w1))
    }
}

pub struct Network<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin> {
    // connections should be sorted after their index.
    pub connections: Vec<Connection<R, W>>,
    pub index: usize,
}

//TODO: struct representing a group of messages from a broadcast?
impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Network<R,W> {
    pub async fn broadcast(&mut self, msg: &impl serde::Serialize) {
        // TODO: Concurrency
         self.connections.iter_mut().map(|conn| {
            conn.send(msg)
        });
    }

    pub async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Vec<T> {
        // TODO: Concurrency
        let messages = self.connections.iter_mut().enumerate().map(|(i, conn)| {
            let msg = conn.recv();
            async move {(i, msg.await)}
        });
        let mut messages = future::join_all(messages).await;
        // Maybe we should pass the id with it?
        // Idk, it doesn't seem like there is a single good way for this.
        messages.sort_unstable_by_key(|(i,_)| *i);
        messages.into_iter().map(|(_,m)| m).collect()
    }

    pub async fn symmetric_broadcast<T>(&mut self, msg: T) -> Vec<T> where T: serde::Serialize + serde::de::DeserializeOwned {
        self.broadcast(&msg).await;
        let mut messages = self.receive_all().await;
        messages.insert(self.index, msg);
        messages
    }
}
impl Network<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>> {
    fn in_memory(player_count: usize) -> Vec<Self> {
        // This could probably be created nicer,
        // but upper-triangular matrices are hard to construct.
        let mut internet = BTreeMap::new();
        for i in 0..player_count {
            for j in 0..i {
                if i == j {continue;}
                let (c1, c2) = Connection::in_memory();
                internet.insert((i,j), c1);
                internet.insert((j,i), c2);
            }
        }

        let mut networks = Vec::new();
        for i in 0..player_count {
            let mut network = Vec::new();
            for j in 0..player_count {
                if i == j { continue;}
                println!("({i}, {j})");
                let conn = internet.remove(&(i,j)).unwrap();
                network.push(conn);
            }
            let network = Network {connections: network, index: i};
            networks.push(network);
        }
        networks
    }
}

#[cfg(test)]
mod test {

    use std::{net::SocketAddrV4, time::Duration};

    use tokio::net::{TcpSocket, TcpListener, TcpStream};

    use super::*;

    // #[tokio::test]
    // async fn send_recv() {
    //     let mut conn = Connection::loopback();

    //     conn.send(&"hello").await;
    //     let msg : Box<str> = conn.recv().await;
    //     let msg : &str = &msg;
    //     assert_eq!(msg, "hello");
    // }

    // #[tokio::test]
    // async fn send_recv_weird() {
    //     let mut conn = Connection::loopback();

    //     use curve25519_dalek::Scalar;
    //     let a = Scalar::from(32u32);
    //     conn.send(&a).await;
    //     let b : Scalar = conn.recv().await;
    //     assert_eq!(a, b);
    // }

    #[tokio::test]
    async fn network() {
        let players = Network::in_memory(1000);
        for p in players {
            tokio::spawn(async move {
                let mut network = p;
                let msg = "Joy to the world!".to_owned();
                network.broadcast(&msg).await;
                let post : Vec<String> = network.receive_all().await;
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
            });
        }
    }

    #[tokio::test]
    async fn tcp() {
        console_subscriber::init();
        let addr = "127.0.0.1:4321".parse::<SocketAddrV4>().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let h1 = tokio::spawn(async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            let mut conn = Connection::from_tcp(stream);
            conn.send(&"Hello");
            println!("[1] Message sent");
            conn.send(&"Buddy");
            println!("[1] Message sent");
            let msg : Box<str> = conn.recv().await;
            println!("[1] Messge received");
            assert_eq!(msg, "Greetings friend".into());

        });
        let h2 = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut conn = Connection::from_tcp(stream);
            let msg : Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Hello".into());
            let msg : Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Buddy".into());
            conn.send(&"Greetings friend");
            println!("[2] Message sent");
        });


        h2.await.unwrap();
        h1.await.unwrap();
    }
}
