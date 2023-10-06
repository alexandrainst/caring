use std::{collections::BTreeMap, cell::RefCell, sync::Arc};

use futures::{future, join, SinkExt, StreamExt};
use tokio::{io::{AsyncWriteExt, AsyncReadExt, BufReader, BufWriter}, sync::{Mutex, mpsc::Sender}, net::{TcpStream, tcp::OwnedReadHalf}};
use tokio_util::codec::{FramedRead, Decoder, LengthDelimitedCodec, FramedWrite, self};


// There is also an argument for splitting this into three structs
// with a common trait.
#[non_exhaustive]
pub enum Connection {
    // We could end up in a problem where we want to send 
    // and receive at the same time. In this case we might need
    // to 'split' the connection. However, that is not nice.
    // We would still end-up with borrowing problems, if we don't form a seperate send/recv
    // struct. Another way could be just use interoir mutability checks with RefCell,
    // thus allowing some degree of freedom,
    // when we know we ore only using half of the connection.
    Tcp2{
        reader: tokio::net::tcp::OwnedReadHalf,
        writer: tokio::net::tcp::OwnedWriteHalf,
    },
    // Unix(tokio::net::UnixStream),
    InMemory{
        sender: tokio::sync::mpsc::Sender<Box<[u8]>>,
        receiver: tokio::sync::mpsc::Receiver<Box<[u8]>>,
    }
    // TODO: Add TLS-backed stream
}


impl Connection {
    /// Constructs an in-memory connection which loop's back onto itself
    /// i.e., when sending with it, you also receive from it.
    ///
    /// By default it has a buffer of 8 items.
    pub fn loopback() -> Self {
        let (send,recv) = tokio::sync::mpsc::channel(8);
        Connection::InMemory { sender: send, receiver: recv }
    }

    /// Construct a pair of in-memory connections which point to each other.
    pub fn in_memory() -> (Self, Self) {
        let (s1,r1) = tokio::sync::mpsc::channel(8);
        let (s2,r2) = tokio::sync::mpsc::channel(8);
        let c1 = Connection::InMemory { sender: s1, receiver: r2 };
        let c2 = Connection::InMemory { sender: s2, receiver: r1 };
        (c1, c2)
    }

    pub fn from_tcp(stream: tokio::net::TcpStream) -> Self {
        let (reader, writer) = stream.into_split();
        Connection::Tcp2{reader, writer}
    }

    // pub fn from_unix(stream: tokio::net::UnixStream) -> Self {
    //     Connection::Unix(stream)
    // }


    pub async fn send(&mut self, msg: &impl serde::Serialize) {
        let msg = bincode::serialize(msg).unwrap();
        match self {
            Connection::InMemory { sender, .. } => {
                sender.send(msg.into_boxed_slice()).await.unwrap();
            },
            Connection::Tcp2{ reader: _, writer } => {
                // Should be fine?
                writer.write_all(&msg).await.unwrap();
            },
        }

    }

    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> T {
        match self {
            Connection::InMemory { sender: _, receiver } => {
                // Technically we could skip serialization here
                // and just pass pointer around and clone.
                let msg = receiver.recv().await.unwrap();
                let buf = std::io::Cursor::new(msg);
                bincode::deserialize_from(buf).unwrap()
            },
            Connection::Tcp2{ reader, writer: _ } => {
                let mut buf = Vec::new();
                reader.read_to_end(&mut buf).await.unwrap();
                let buf = std::io::Cursor::new(buf);
                bincode::deserialize_from(buf).unwrap()
            },
        }
    }
}


pub struct TcpChannel {
    task: tokio::task::JoinHandle<()>,
    input: tokio::sync::mpsc::Sender<Box<[u8]>>,
    reader: FramedRead<OwnedReadHalf, LengthDelimitedCodec>,
}

impl TcpChannel {
    pub fn from_tcp(stream: TcpStream) -> Self {
        let (reader, writer) = stream.into_split();
        let (input, mut outgoing) : (Sender<Box<[u8]>>, _) = tokio::sync::mpsc::channel(8);
        let codec = LengthDelimitedCodec::new();
        let reader = FramedRead::new(reader, codec.clone());
        let mut writer = FramedWrite::new(writer, codec);

        let task = tokio::spawn(async move {
            while let Some(msg) = outgoing.recv().await {
                writer.send(msg.into()).await.unwrap();
            }
        });
        TcpChannel {task, input, reader}
    }

    #[async_backtrace::framed]
    pub fn send(&self, msg: &impl serde::Serialize) {
        let msg = bincode::serialize(msg).unwrap();
        self.input.try_send(msg.into()).unwrap();
    }

    #[async_backtrace::framed]
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> T {
        let buf = self.reader.next().await.unwrap().unwrap();
        let buf = std::io::Cursor::new(buf);
        bincode::deserialize_from(buf).unwrap()
    }
}

pub struct Network {
    // connections should be sorted after their index.
    pub connections: Vec<Connection>,
    pub index: usize,
}

// TODO: struct representing a group of messages from a broadcast?
impl Network {
    pub async fn broadcast(&mut self, msg: &impl serde::Serialize) {
        // TODO: Concurrency
        let futures = self.connections.iter_mut().map(|conn| {
            conn.send(msg)
        });
        future::join_all(futures).await;
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

    pub fn in_memory(player_count: usize) -> Vec<Network> {
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

    #[tokio::test]
    async fn send_recv() {
        let mut conn = Connection::loopback();

        conn.send(&"hello").await;
        let msg : Box<str> = conn.recv().await;
        let msg : &str = &msg;
        assert_eq!(msg, "hello");
    }

    #[tokio::test]
    async fn send_recv_weird() {
        let mut conn = Connection::loopback();

        use curve25519_dalek::Scalar;
        let a = Scalar::from(32u32);
        conn.send(&a).await;
        let b : Scalar = conn.recv().await;
        assert_eq!(a, b);
    }

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
            let mut conn = TcpChannel::from_tcp(stream);
            conn.send(&"Hello");
            println!("[1] Message sent");
            conn.send(&"Buddy");
            println!("[1] Message sent");
            // let msg : Box<str> = conn.recv().await;
            // println!("[1] Messge received");
            // assert_eq!(msg, "Greetings".into());

        });
        let h2 = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut conn = TcpChannel::from_tcp(stream);
            let msg : Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Hello".into());
            let msg : Box<str> = conn.recv().await;
            println!("[2] Message received");
            assert_eq!(msg, "Buddy".into());
            // conn.send(&"Greetings\0");
            // println!("[2] Message sent");
        });


        h2.await.unwrap();
        h1.await.unwrap();
    }
}
