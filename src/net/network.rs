use std::{collections::BTreeMap, net::SocketAddr, ops::Range, time::Duration};

use futures::future::join_all;
use futures::prelude::*;
use itertools::Itertools;
use rand::{thread_rng, Rng};
use tokio::{
    io::{AsyncRead, AsyncWrite, AsyncWriteExt, DuplexStream, ReadHalf, WriteHalf},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
};

use crate::net::{
    agency::{Broadcast, Unicast},
    connection::{Connection, ConnectionError}, Tuneable,
};

/// Peer-2-peer network
///
/// This acts as a single waypoint to all other (connected) parties in the network.
/// Ideally the connection list should be the same for all parties, however it could differ.
///
/// * `connections`: Connections, one for each peer, sorted by their index, skipping our own index.
/// * `index`: My own index
pub struct Network<R: tokio::io::AsyncRead + Unpin, W: tokio::io::AsyncWrite + Unpin> {
    // NOTE:
    // We could also insert a 'fake' Connection into the set for the representation of ourselves.
    // However that is probably a less efficient, if nicer, abstraction.
    connections: Vec<Connection<R, W>>,
    pub index: usize,
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin,
        W: tokio::io::AsyncWrite + std::marker::Unpin,
    > std::ops::Index<usize> for Network<R, W>
{
    type Output = Connection<R, W>;

    fn index(&self, index: usize) -> &Self::Output {
        let i = self.id_to_index(index);
        &self.connections[i]
    }
}

impl<
        R: tokio::io::AsyncRead + std::marker::Unpin,
        W: tokio::io::AsyncWrite + std::marker::Unpin,
    > std::ops::IndexMut<usize> for Network<R, W>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i = self.id_to_index(index);
        &mut self.connections[i]
    }
}

#[derive(thiserror::Error, Debug)]
#[error("Error communicating with {id}: {source}")]
pub struct NetworkError {
    id: u32,
    source: ConnectionError,
}

// TODO: Do timeouts?
impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Network<R, W> {
    fn id_to_index(&self, index: usize) -> usize {
        let n = self.connections.len() + 1;
        if index < self.index {
            index
        } else if index == self.index {
            // You probably didn't mean to do that.
            panic!("Trying to reference self connection, id = {index}")
        } else if index < n {
            index - 1
        } else {
            // Out of bounds
            panic!("Only {n} in network, but referenced id = {index}")
        }
    }

    /// Broadcast a message to all other parties.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msg`: Message to send
    pub async fn broadcast(&mut self, msg: &impl serde::Serialize) -> Result<(), NetworkError> {
        let my_id = self.index;
        let outgoing = self.connections.iter_mut().enumerate().map(|(i, conn)| {
            let id = if i < my_id { i } else { i + 1 } as u32;
            conn.send(&msg)
                .map_err(move |e| NetworkError { source: e, id })
        });
        future::try_join_all(outgoing).await?;
        Ok(())
    }

    /// Unicast messages to each party
    ///
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`, skipping your own index.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msgs`: Messages to send
    pub async fn unicast(&mut self, msgs: &[impl serde::Serialize]) -> Result<(), NetworkError> {
        let my_id = self.index;
        let outgoing = self
            .connections
            .iter_mut()
            .zip(msgs.iter())
            .enumerate()
            .map(|(i, (conn, msg))| {
                let id = if i < my_id { i } else { i + 1 } as u32;
                conn.send(msg)
                    .map_err(move |e| NetworkError { source: e, id })
            });
        future::try_join_all(outgoing).await?;
        Ok(())
    }

    /// Receive a message for each party.
    ///
    /// Asymmetric, waiting
    ///
    /// Returns: A list sorted by the connections (skipping yourself)
    pub async fn receive_all<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> Result<Vec<T>, NetworkError> {
        let my_id = self.index;
        let messages = self.connections.iter_mut().enumerate().map(|(i, conn)| {
            let msg = conn.recv::<T>();
            let msg = tokio::time::timeout(Duration::from_secs(5), msg);
            let id = if i < my_id { i } else { i + 1 } as u32;
            async move { (id, msg.await) }
        });
        let messages = future::join_all(messages).await;
        // Maybe we should pass the id with it?
        // Idk, it doesn't seem like there is a single good way for this.
        let messages : Vec<_> = messages
            .into_iter()
            .map(|(id, m)| match m {
                Ok(m) => m.map_err(|e| NetworkError { id, source: e }),
                Err(duration) => Err(NetworkError {
                    id,
                    source: ConnectionError::TimeOut(duration),
                }),
            })
            .collect::<Result<_,_>>()?;

            assert!(messages.len() == self.connections.len(), "Too few messages received");

            Ok(messages)
    }

    /// Broadcast a message to all parties and await their messages
    /// Messages are ordered by their index.
    ///
    /// * `msg`: message to send and receive
    pub async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, NetworkError>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let my_id = self.index;
        let (mut rx, mut tx): (Vec<_>, Vec<_>) =
            self.connections.iter_mut().map(|c| c.split()).unzip();

        let outgoing = tx.iter_mut().enumerate().map(|(id, conn)| {
            let id = if id < my_id { id } else { id + 1 } as u32;
            conn.send_async(&msg)
                .map_err(move |e| NetworkError { source: e, id })
        });

        let messages = rx.iter_mut().enumerate().map(|(i, conn)| {
            let msg = conn.recv::<T>();
            let msg = tokio::time::timeout(Duration::from_secs(5), msg);
            let id = if i < my_id { i } else { i + 1 } as u32;
            async move { (id, msg.await) }
        });
        let (receipts, messages) =
            futures::join!(future::try_join_all(outgoing), future::join_all(messages));
        receipts?;

        // Maybe we should pass the id with it?
        // Idk, it doesn't seem like there is a single good way for this.
        // messages.sort_unstable_by_key(|(i, _)| *i);
        let mut messages : Vec<_> = messages
            .into_iter()
            .map(|(i, m)| {
                let id = i;
                match m {
                    Ok(m) => m.map_err(|e| NetworkError { id, source: e }),
                    Err(duration) => Err(NetworkError {
                        id,
                        source: ConnectionError::TimeOut(duration),
                    }),
                }
            })
            .collect::<Result<_,_>>()?;

        messages.insert(self.index, msg);
        Ok(messages)
    }

    /// Unicast a message to each party and await their messages
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`.
    ///
    /// * `msg`: message to send and receive
    pub async fn symmetric_unicast<T>(&mut self, mut msgs: Vec<T>) -> Result<Vec<T>, NetworkError>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let my_id = self.index;
        let my_own_msg = msgs.remove(my_id);

        let (mut rx, mut tx): (Vec<_>, Vec<_>) =
            self.connections.iter_mut().map(|c| c.split()).unzip();

        let outgoing = tx
            .iter_mut()
            .zip(msgs.iter())
            .enumerate()
            .map(|(id, (conn, msg))| {
                let id = id as u32;
                conn.send_async(msg)
                    .map_err(move |e| NetworkError { source: e, id })
            });

        let messages = rx.iter_mut().enumerate().map(|(i, conn)| {
            let id = if i < my_id { i } else { i + 1 } as u32;
            let msg = conn.recv::<T>();
            let msg = tokio::time::timeout(Duration::from_secs(5), msg);
            async move { (id, msg.await) }
        });
        let (receipts, messages) =
            futures::join!(future::try_join_all(outgoing), future::join_all(messages));
        receipts?;

        // Maybe we should pass the id with it?
        // Idk, it doesn't seem like there is a single good way for this.
        // messages.sort_unstable_by_key(|(i, _)| *i);
        let mut messages : Vec<_> = messages
            .into_iter()
            .map(|(id, m)| match m {
                Ok(m) => m.map_err(|e| NetworkError { id, source: e }),
                Err(duration) => Err(NetworkError {
                    id,
                    source: ConnectionError::TimeOut(duration),
                }),
            })
            .collect::<Result<_,_>>()?;

        messages.insert(self.index, my_own_msg);
        Ok(messages)
    }

    /// Resolute IDs
    ///
    /// Each party picks a random number then broadcasts it.
    /// We then (all) sort the connection list by the numbers picked.
    ///
    /// * `rng`: Random number generator to use
    pub async fn resolute_ids(&mut self, rng: &mut impl Rng) -> Result<(), NetworkError> {
        self.index = 0; // reset index.
        let num: u64 = rng.gen();
        let results = self.symmetric_broadcast(num).await?;
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
        Ok(())
    }

    /// Returns a range for representing the participants.
    pub fn participants(&self) -> Range<u32> {
        let n = self.connections.len() as u32;
        let n = n + 1; // We need to count ourselves.
        0..n
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Unicast for Network<R, W> {
    type Error = NetworkError;

    #[tracing::instrument(skip_all)]
    async fn unicast(&mut self, msgs: &[impl serde::Serialize]) -> Result<(), Self::Error> {
        self.unicast(msgs).await
    }

    #[tracing::instrument(skip_all)]
    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Result<Vec<T>, Self::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        self.symmetric_unicast(msgs).await
    }

    #[tracing::instrument(skip_all)]
    async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> Result<Vec<T>, Self::Error> {
        self.receive_all().await
    }

    fn size(&self) -> usize {
        self.connections.len()
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Broadcast for Network<R, W> {
    type Error = NetworkError;

    #[tracing::instrument(skip_all)]
    async fn broadcast(&mut self, msg: &impl serde::Serialize) -> Result<(), Self::Error> {
        self.broadcast(msg).await
    }

    #[tracing::instrument(skip_all)]
    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, Self::Error>
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        self.symmetric_broadcast(msg).await
    }

    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        idx: usize,
    ) -> impl Future<Output = Result<T, Self::Error>> {
            Tuneable::recv_from(self, idx)
                .map_err(move |e| NetworkError{id: idx as u32, source: e})
    }

    fn size(&self) -> usize {
        self.connections.len()
    }
}

/// Network containing only duplex connections.
/// Used for local testing.
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

/// TCP Network based on TCP Streams.
pub type TcpNetwork = Network<OwnedReadHalf, OwnedWriteHalf>;

impl TcpNetwork {
    /// Construct a TCP-based network by opening a socket and connecting to peers.
    /// If peers cannot be connected to, we wait until we wait until they have
    /// connected to us.
    ///
    /// * `me`: Socket address to open to
    /// * `peers`: Socket addresses of other peers
    pub async fn connect(me: SocketAddr, peers: &[SocketAddr]) -> Result<Self, NetworkError> {
        let n = peers.len();

        // Connecting to parties
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
        // Accepting connections
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

        let connections = parties.into_iter().map(Connection::from_tcp).collect();

        let mut network = Self {
            connections,
            index: 0,
        };
        network.resolute_ids(&mut thread_rng()).await?;

        Ok(network)
    }

    pub async fn shutdown(self) -> Result<(), NetworkError> {
        let futs = self
            .connections
            .into_iter()
            .enumerate()
            .map(|(i, conn)| async move {
                match conn.to_tcp().await {
                    Ok(mut tcp) => {
                        tcp.shutdown().await.unwrap();
                        Ok(())
                    }
                    Err(e) => Err(NetworkError {
                        id: i as u32,
                        source: e,
                    }),
                }
            });
        join_all(futs).await.into_iter().map_ok(|_| {}).collect()
    }
}

#[cfg(test)]
mod test {
    use std::collections::VecDeque;

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
                network.broadcast(&msg).await.unwrap();
                let post: Vec<String> = network.receive_all().await.unwrap();
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
            });
        }
    }


    #[tokio::test]
    async fn broadcasting() {
        const N : usize = 3;
        let players = Network::in_memory(N);
        let mut messages : VecDeque<_> = vec!["Over".to_string(), "And".to_string(), "Out".to_string()].into();

        let mut tasks = Vec::new();
        for p in players {
            let msg = messages.pop_front().unwrap();
            let t = tokio::spawn(async move {
                let mut network = p;
                let post = network.symmetric_broadcast(msg).await.unwrap();
                assert!(post.len() == N);
                assert_eq!(post[0], "Over");
                assert_eq!(post[1], "And");
                assert_eq!(post[2], "Out");
            });
            tasks.push(t);
        }
        let res = future::try_join_all(tasks.into_iter()).await;
        res.unwrap();
    }


    #[tokio::test]
    async fn unicasting() {
        const N : usize = 3;
        let players = Network::in_memory(N);
        let mut messages : VecDeque<_> = vec![[0,1,2], [0,1,2], [0,1,2]].into();
        // Each party sends each other party a message.
        // To test this we send each party their id as the message.
        let mut tasks = Vec::new();
        for p in players {
            let msg = messages.pop_front().unwrap();
            let t = tokio::spawn(async move {
                let mut network = p;
                let post = network.symmetric_unicast(msg.to_vec()).await.unwrap();
                dbg!(network.index, &post);
                assert!(post.len() == N);
                assert_eq!(post[0], network.index);
                assert_eq!(post[1], network.index);
                assert_eq!(post[2], network.index);
            });
            tasks.push(t);
        }
        let res = future::try_join_all(tasks.into_iter()).await;
        res.unwrap();
    }
}
