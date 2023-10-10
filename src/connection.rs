//! Module for doing arbitrary communication in 'some' medium.
//! This 'medium' can be anything that implements `AsyncRead`/`AsyncWrite`.
//! There is built-in support for TCP and in-memory duplex-based connections.

use std::{collections::BTreeMap, marker::PhantomData, net::SocketAddr, ops::Range};

use futures::{future, SinkExt, StreamExt};
use itertools::Itertools;
use rand::{thread_rng, Rng};
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
    // task: tokio::task::JoinHandle<()>,
    phantom: PhantomData<W>,           // Not needed, but nice
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

        let _task = tokio::spawn(async move {
            // This self-drops after the sender is gone.
            while let Some(msg) = outgoing.recv().await {
                writer.send(msg.into()).await.unwrap();
            }
        });
        Connection {
            // task,
            input,
            reader,
            phantom: PhantomData,
        }
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Connection<R, W> {
    /// Send a message without waiting
    ///
    /// * `msg`: Message to send
    pub fn send(&self, msg: &impl serde::Serialize) {
        let msg = bincode::serialize(msg).unwrap();
        self.input.try_send(msg.into()).unwrap();
    }

    /// Receive a message waiting for arrival
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> T {
        // TODO: Handle unstable connections
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
}

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

/// Peer-2-peer network
///
/// * `connections`: Connections for each peer, sorted by their index.
/// * `index`: My own index
pub struct Network<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin> {
    pub connections: Vec<Connection<R, W>>,
    pub index: usize,
}

// TODO: Do timeouts?
impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Network<R, W> {
    /// Broadcast a message to all other parties.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msg`: Message to send
    pub fn broadcast(&mut self, msg: &impl serde::Serialize) {
        for conn in &self.connections {
            conn.send(msg);
        }
    }

    /// Unicast messages to each party
    ///
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`, skipping your own index.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msgs`: Messages to send
    pub fn unicast(&mut self, msgs: &[impl serde::Serialize]) {
        for (conn, msg) in self.connections.iter().zip(msgs.iter()) {
            conn.send(msg);
        }
    }

    /// Receive a message for each party.
    ///
    /// Asymmetric, waiting
    ///
    /// Returns: A list sorted by the connections (skipping yourself)
    pub async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Vec<T> {
        // TODO: Concurrency
        let messages = self.connections.iter_mut().enumerate().map(|(i, conn)| {
            let msg = conn.recv();
            async move { (i, msg.await) }
        });
        let mut messages = future::join_all(messages).await;
        // Maybe we should pass the id with it?
        // Idk, it doesn't seem like there is a single good way for this.
        messages.sort_unstable_by_key(|(i, _)| *i);
        messages.into_iter().map(|(_, m)| m).collect()
    }

    /// Broadcast a message to all parties and await their messages
    /// Messages are ordered by their index.
    ///
    /// * `msg`: message to send and receive
    pub async fn symmetric_broadcast<T>(&mut self, msg: T) -> Vec<T>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        self.broadcast(&msg);
        let mut messages = self.receive_all().await;
        messages.insert(self.index, msg);
        messages
    }

    /// Unicast a message to each party and await their messages
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`.
    ///
    /// * `msg`: message to send and receive
    pub async fn symmetric_unicast<T>(&mut self, mut msgs: Vec<T>) -> Vec<T>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let mine = msgs.remove(self.index);
        self.unicast(&msgs);
        let mut messages = self.receive_all().await;
        messages.insert(self.index, mine);
        messages
    }

    /// Resolute IDs
    ///
    /// Each party picks a random number then broadcasts it.
    /// We then (all) sort the connection list by the numbers picked.
    ///
    /// * `rng`: Random number generator to use
    pub async fn resolute_ids(&mut self, rng: &mut impl Rng) {
        self.index = 0; // reset index.
        let num: u64 = rng.gen();
        let results = self.symmetric_broadcast(num).await;
        // Enumerate the results, sort by the ballots and return the indices.
        let mut results: Vec<_> = results.into_iter().enumerate().collect();
        results.sort_unstable_by_key(|(_, n): &(usize, u64)| *n);
        // Results have `n` items
        let mut results: Vec<_> = results.into_iter().map(|(i, _)| i).collect();

        // Connections only have `n-1` items, since we do not connect to ourselves.
        // Thus we need to add ourselves.
        let connections = std::mem::take(&mut self.connections);
        let mut connections: Vec<_> = connections.into_iter().map(Option::Some).collect();
        connections.insert(0, None);

        // Remove our own position, remember we are at zero.
        let (i, _) = results.iter().find_position(|&i| *i == 0).unwrap();
        results.remove(i); // remember we have removed it to not remove it twice.
        self.index = i;

        // Now we sort the connections by their index.
        let sorted = results.into_iter().map(|i| {
            std::mem::take(&mut connections[i]).expect("No element should be removed twice")
        });
        // Add it back.
        self.connections.extend(sorted);
    }

    /// Returns a range for representing the participants.
    pub fn participants(&self) -> Range<u32> {
        let n = self.connections.len() as u32;
        let n = n + 1; // We need to count ourselves.
        0..n
    }
}

pub type InMemoryNetwork = Network<ReadHalf<DuplexStream>, WriteHalf<DuplexStream>>;

impl InMemoryNetwork {
    /// Construct a list of networks for each 'peer' in the peer-2-peer network.
    ///
    /// * `player_count`: Size of the network in terms of peers.
    pub fn in_memory(player_count: usize) -> Vec<Self> {
        // This could probably be created nicer,
        // but upper-triangular matrices are hard to construct.
        let mut internet = BTreeMap::new();
        for i in 0..player_count {
            for j in 0..i {
                if i == j {
                    continue;
                }
                let (c1, c2) = Connection::in_memory();
                internet.insert((i, j), c1);
                internet.insert((j, i), c2);
            }
        }

        let mut networks = Vec::new();
        for i in 0..player_count {
            let mut network = Vec::new();
            for j in 0..player_count {
                if i == j {
                    continue;
                }
                let conn = internet.remove(&(i, j)).unwrap();
                network.push(conn);
            }
            let network = Network {
                connections: network,
                index: i,
            };
            networks.push(network);
        }
        networks
    }
}

pub type TcpNetwork = Network<OwnedReadHalf, OwnedWriteHalf>;

impl TcpNetwork {
    /// Construct a TCP-based network by opening a socket and connecting to peers.
    /// If peers cannot be connected to, we wait until we wait until they have
    /// connected to us.
    ///
    /// * `me`: Socket address to open to
    /// * `peers`: Socket addresses of other peers
    pub async fn connect(me: SocketAddr, peers: &[SocketAddr]) -> Self {
        let n = peers.len();

        println!("Connecting to parties");
        let results = future::join_all(
            peers
                .iter()
                .map(|addr| tokio::task::spawn(tokio::net::TcpStream::connect(*addr))),
        )
        .await;

        let mut parties: Vec<_> = results
            .into_iter()
            .map(|x| x.unwrap())
            .filter_map(|x| x.ok())
            .collect();

        // If we are not able to connect to some, they will connect to us.
        println!("Accepting connections");
        let incoming = tokio::net::TcpListener::bind(me).await.unwrap();
        loop {
            if parties.len() >= n {
                break;
            };
            let (stream, _) = incoming.accept().await.unwrap();
            parties.push(stream);
        }

        // Not sure if a good idea?
        // Fresco does it
        for stream in &mut parties {
            stream.set_nodelay(true).unwrap();
        }

        let connections = parties.into_iter()
            .map(Connection::from_tcp).collect();


        let mut network = Self {
            connections,
            index: 0,
        };
        network.resolute_ids(&mut thread_rng()).await;

        network
    }
}

#[cfg(test)]
mod test {

    use std::net::SocketAddrV4;

    use tokio::net::{TcpListener, TcpStream};

    use super::*;

    #[tokio::test]
    async fn network() {
        println!("Spawning network!");
        let players = Network::in_memory(100);
        // remember it's n^n messages, one for each party, to each party.
        println!("Done!");
        for p in players {
            tokio::spawn(async move {
                let mut network = p;
                let msg = "Joy to the world!".to_owned();
                network.broadcast(&msg);
                let post: Vec<String> = network.receive_all().await;
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
            });
        }
    }

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
