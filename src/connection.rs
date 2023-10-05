use std::collections::BTreeMap;

use tokio::io::{AsyncWriteExt, AsyncReadExt};

#[non_exhaustive]
pub enum Connection {
    // We could end up in a problem where we want to send 
    // and receive at the same time. In this case we might need
    // to 'split' the connection. However, that is not nice.
    Tcp(tokio::net::TcpStream),
    Unix(tokio::net::UnixStream),
    InMemory{
        sender: tokio::sync::mpsc::Sender<Box<[u8]>>,
        receiver: tokio::sync::mpsc::Receiver<Box<[u8]>>,
    }
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

    pub fn from_tcp_stream(stream: tokio::net::TcpStream) -> Self {
        Connection::Tcp(stream)
    }


    pub async fn send(&mut self, msg: &impl serde::Serialize) {
        let msg = bincode::serialize(msg).unwrap();
        match self {
            Connection::InMemory { sender, .. } => {
                sender.send(msg.into_boxed_slice()).await.unwrap();
            },
            Connection::Tcp(stream) => {
                stream.write_all(&msg).await.unwrap();
            },
            Connection::Unix(stream) => {
                stream.write_all(&msg).await.unwrap();
            },
        }

    }

    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> T {
        match self {
            Connection::InMemory { sender: _, receiver } => {
                let msg = receiver.recv().await.unwrap();
                let buf = std::io::Cursor::new(msg);
                bincode::deserialize_from(buf).unwrap()
            },
            Connection::Tcp(stream) => {
                let mut buf = Vec::new();
                stream.read_to_end(&mut buf).await.unwrap();
                let buf = std::io::Cursor::new(buf);
                bincode::deserialize_from(buf).unwrap()
            },
            Connection::Unix(stream) => {
                let mut buf = Vec::new();
                stream.read_to_end(&mut buf).await.unwrap();
                let buf = std::io::Cursor::new(buf);
                bincode::deserialize_from(buf).unwrap()
            },
        }
    }
}

pub struct Network {
    connections: Vec<Connection>,
}

impl Network {
    pub async fn broadcast(&mut self, msg: &impl serde::Serialize) {
        // TODO: Concurrency
        for conn in &mut self.connections {
            conn.send(msg).await;
        }
    }

    pub async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Vec<T> {
        // TODO: Concurrency
        let mut messages = Vec::new();
        for conn in &mut self.connections {
            let msg : T = conn.recv().await;
            messages.push(msg);
        }
        messages
    }

    pub async fn symmetric_broadcast<T: serde::de::DeserializeOwned + serde::Serialize>(&mut self, msg: T) -> Vec<T> {
        // TODO: Concurrency
        self.broadcast(&msg).await;
        let mut messages = self.receive_all().await;
        messages.push(msg);
        messages
    }

    pub fn in_memory(num: usize) -> Vec<Network> {
        // This could probably be created nicer,
        // but upper-triangular matrices are hard to construct.
        let mut internet = BTreeMap::new();
        for i in 0..num {
            for j in 0..i {
                if i == j {continue;}
                let (c1, c2) = Connection::in_memory();
                internet.insert((i,j), c1);
                internet.insert((j,i), c2);
            }
        }

        let mut networks = Vec::new();
        for i in 0..num {
            let mut network = Vec::new();
            for j in 0..num {
                if i == j { continue;}
                println!("({i}, {j})");
                let conn = internet.remove(&(i,j)).unwrap();
                network.push(conn);
            }
            let network = Network {connections: network};
            networks.push(network);
        }
        networks
    }
}

#[cfg(test)]
mod test {
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
        let players = Network::in_memory(2);
        for p in players {
            tokio::spawn(async move {
                let mut network = p;
                let msg = "Joy to the world!".to_owned();
                let post = network.symmetric_broadcast(msg).await;
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
            });
        }
    }

}
