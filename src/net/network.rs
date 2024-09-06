use crate::net::{Channel, Communicate, Id, ReceiverError, RecvBytes, SendBytes};
use std::{collections::BTreeMap, net::SocketAddr, ops::Range, time::Duration};

use futures::future::join_all;
use futures::prelude::*;
use itertools::Itertools;
use rand::{thread_rng, Rng};
use tokio::time::error::Elapsed;
use tokio_util::bytes::Bytes;

use crate::net::{
    agency::{Broadcast, Unicast},
    connection::{Connection, DuplexConnection, TcpConnection},
    SplitChannel, Tuneable,
};

// NOTE: We should probably find a way to include drop-outs in the broadcasts, since threshold
// schemes will continue to function if we lose connections underway. Maybe this is just handled by
// the network? But that would require the ability to resume a protocol after handling the drop-out.
// Another method is just ignore drop-outs, and as such the network will never error out.
// Otherwise we could do something totally different, which is let the network just have a
// threshold, ignoring drop-outs until then, then returning errors.
//
// In the same manner we can let the network have re-try strategies and the like.
// It is probably better handled in that layer anyway.
//
// One could still be for the broadcast/unicast operations to have a `size` function
// which gives the current network size. However I am not sure if this will be relevant?

/// Peer-to-Peer network
///
/// This acts as a single waypoint to all other (connected) parties in the network.
/// Ideally the connection list should be the same for all parties, however it could differ.
///
/// * `connections`: Connections, one for each peer, sorted by their index, skipping our own index.
/// * `index`: My own index
pub struct Network<C: SplitChannel> {
    // NOTE:
    // We could also insert a 'fake' Connection into the set for the representation of ourselves.
    // However that is probably a less efficient, if nicer, abstraction.
    pub(crate) connections: Vec<C>,
    pub(crate) index: usize,
}

impl<C: SplitChannel> std::fmt::Debug for Network<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.connections.len();
        f.debug_struct("Network")
            .field("connections", &n)
            .field("index", &self.index)
            .finish()
    }
}

#[derive(thiserror::Error, Debug)]
pub enum NetworkError<E, U> {
    #[error("Error receiving from {id}: {source}")]
    Incoming { id: u32, source: ReceiverError<E> },
    #[error("Error sending to {id}: {source}")]
    Outgoing { id: u32, source: U },
    #[error("{id} timeouted after {elapsed:?}")]
    TimeOut { id: u32, elapsed: Elapsed },
}

#[allow(type_alias_bounds)] // It clearly matters, stop complaining
type NetResult<T, C: Channel> = std::result::Result<T, NetworkError<C::RecvError, C::SendError>>;

// PERFORMANCE: serialize in network once when broadcasting.
impl<C: SplitChannel> Network<C> {
    fn id_to_index(&self, Id(id): Id) -> usize {
        let n = self.connections.len() + 1;
        if id < self.index {
            id
        } else if id == self.index {
            // You probably didn't mean to do that.
            panic!("Trying to reference self connection, id = {id}")
        } else if id < n {
            id - 1
        } else {
            // Out of bounds
            panic!("Only {n} in network, but referenced id = {id}")
        }
    }

    pub fn id(&self) -> Id {
        Id(self.index)
    }

    pub fn set_id(&mut self, id: Id) {
        tracing::info!("Obtained new id = {id:?}");
        self.index = id.0;
    }

    pub fn prev_neighbour(&self) -> Id {
        let n = self.connections.len();
        Id((self.index + n - 1) % n)
    }

    pub fn next_neighbour(&self) -> Id {
        let n = self.connections.len();
        Id((self.index + n + 1) % n)
    }

    pub fn peers(&self) -> Vec<Id> {
        let n = self.connections.len();
        (0..=n).map(Id).filter(|id| *id != self.id()).collect_vec()
    }

    /// Returns a range for representing the participants.
    pub fn participants(&self) -> Range<u32> {
        let n = self.connections.len() as u32;
        let n = n + 1; // We need to count ourselves.
        0..n
    }

    /// Broadcast a message to all other parties.
    ///
    /// Asymmetric, non-waiting
    ///
    /// * `msg`: Message to send
    pub async fn broadcast(&mut self, msg: &(impl serde::Serialize + Sync)) -> NetResult<(), C> {
        let my_id = self.index;
        let packet: Bytes = bincode::serialize(&msg).unwrap().into();
        let outgoing = self.connections.iter_mut().enumerate().map(|(i, conn)| {
            let id = if i < my_id { i } else { i + 1 } as u32;
            conn.send_bytes(packet.clone())
                .map_err(move |e| NetworkError::Outgoing { id, source: e })
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
    pub async fn unicast(&mut self, msgs: &[impl serde::Serialize + Sync]) -> NetResult<(), C> {
        let my_id = self.index;
        let outgoing = self
            .connections
            .iter_mut()
            .zip(msgs.iter())
            .enumerate()
            .map(|(i, (conn, msg))| {
                let id = if i < my_id { i } else { i + 1 } as u32;
                conn.send(msg)
                    .map_err(move |e| NetworkError::Outgoing { id, source: e })
            });
        future::try_join_all(outgoing).await?;
        Ok(())
    }

    /// Receive a message for each party.
    ///
    /// Asymmetric, waiting
    ///
    /// Returns: A list sorted by the connections (skipping yourself)
    pub async fn receive_all<T: serde::de::DeserializeOwned>(&mut self) -> NetResult<Vec<T>, C> {
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
        let messages: Vec<_> = messages
            .into_iter()
            .map(|(id, m)| match m {
                Ok(m) => m.map_err(|e| NetworkError::Incoming { id, source: e }),
                Err(_duration) => todo!("handle it"),
            })
            .collect::<NetResult<_, C>>()?;

        assert!(
            messages.len() == self.connections.len(),
            "Too few messages received"
        );

        Ok(messages)
    }

    /// Broadcast a message to all parties and await their messages
    /// Messages are ordered by their index.
    ///
    /// * `msg`: message to send and receive
    pub async fn symmetric_broadcast<T>(&mut self, msg: T) -> NetResult<Vec<T>, C>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Sync,
    {
        let my_id = self.index;
        let (mut tx, mut rx): (Vec<_>, Vec<_>) = self
            .connections
            .iter_mut()
            .map(super::SplitChannel::split)
            .unzip();

        let packet: Bytes = bincode::serialize(&msg).unwrap().into();
        let outgoing = tx.iter_mut().enumerate().map(|(id, conn)| {
            let id = if id < my_id { id } else { id + 1 } as u32;
            conn.send_bytes(packet.clone())
                .map_err(move |e| NetworkError::Outgoing { id, source: e })
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
        let mut messages: Vec<_> = messages
            .into_iter()
            .map(|(i, m)| {
                let id = i;
                match m {
                    Ok(m) => m.map_err(|e| NetworkError::Incoming { id, source: e }),
                    Err(_duration) => {
                        todo!("handle it")
                        //Err(NetworkError {
                        //id,
                        //source: ConnectionError::TimeOut(duration),
                        //})
                    }
                }
            })
            .collect::<NetResult<_, C>>()?;

        messages.insert(self.index, msg);
        Ok(messages)
    }

    /// Unicast a message to each party and await their messages
    /// Messages are supposed to be in order, meaning message `i`
    /// will be send to party `i`.
    ///
    /// * `msg`: message to send and receive
    ///
    /// # Errors
    ///
    ///  Returns a [``NetworkError``] with an ``id`` if the underlying connection to that ``id`` fails.
    ///
    pub async fn symmetric_unicast<T>(&mut self, mut msgs: Vec<T>) -> NetResult<Vec<T>, C>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Sync,
    {
        let my_id = self.index;
        let my_own_msg = msgs.remove(my_id);

        let (mut tx, mut rx): (Vec<_>, Vec<_>) = self
            .connections
            .iter_mut()
            .map(super::SplitChannel::split)
            .unzip();

        let outgoing = tx
            .iter_mut()
            .zip(msgs.iter())
            .enumerate()
            .map(|(id, (conn, msg))| {
                let id = id as u32;
                conn.send(msg)
                    .map_err(move |e| NetworkError::Outgoing { id, source: e })
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
        let mut messages: Vec<_> = messages
            .into_iter()
            .map(|(id, m)| match m {
                Ok(m) => m.map_err(|e| NetworkError::Incoming { id, source: e }),
                Err(elapsed) => Err(NetworkError::TimeOut { id, elapsed }),
            })
            .collect::<NetResult<_, C>>()?;

        messages.insert(self.index, my_own_msg);
        Ok(messages)
    }

    /// Resolute IDs
    ///
    /// Each party picks a random number then broadcasts it.
    /// We then (all) sort the connection list by the numbers picked.
    ///
    /// * `rng`: Random number generator to use
    pub async fn resolute_ids(&mut self, rng: &mut impl Rng) -> NetResult<(), C> {
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
        let (i, _) = results
            .iter()
            .find_position(|&i| *i == 0)
            .expect("Id zero should be in there");
        results.remove(i); // remember we have removed it to not remove it twice.
        self.index = i;

        // Now we sort the connections by their index.
        let sorted = results.into_iter().map(|i| {
            std::mem::take(&mut connections[i]).expect("No element should be removed twice")
        });
        // Add it back.
        self.connections.extend(sorted);

        let id = self.id();
        tracing::info!("Obtained new id = {id:?} from id-resolution");
        Ok(())
    }

    async fn drop_party(_id: usize) -> Result<(), ()> {
        todo!("Initiate a drop vote");
    }

    pub(crate) fn as_mut(&mut self) -> Network<&mut C> {
        let connections = self.connections.iter_mut().collect();
        Network {
            connections,
            index: self.index,
        }
    }
}

// TODO: Implement a handler system, such we can handle ad-hoc requests from other parties,
// such as dropping/kicking other parties for cheating, being slow, etc.
//
// Outline:
// Currently we do not handle any unprepared protocols, but only expected 'happy path' behaviour.
// In case of protocols or communication failure we return an error, but we do not provide a solution.
// The current expection is for the downstream user to handle it themselves, instead of doing
// something automatic. However, we currently do not have any methods for removing parties,
// and if we had we still need all other parties to come to the same conclusion.
//
// First,
// we are in need of some voting protocols, such we can initiate a 'drop' vote.
// How this should be done is not clear-cut, but we can start with something simple.
//
// Second,
// We need the ability to handle these ad-hoc as these voting requests can come at any point in
// time, while we could check for votes manually each 'round' between each protocol, this would not
// probably not suffice.
//
// We can use asyncness to run these in the back, racing/selecting between the happy-path and
// incoming vote requests. A handler should be able to be set up so the policies/code for how to
// react on these requests should be handled.
//
// The issue here becomes that we need to process channels before deciding to relay them or handle
// it with the vote handler.
//
//
//

impl<C: SplitChannel> Unicast for Network<C> {
    type UnicastError = NetworkError<C::RecvError, C::SendError>;

    #[tracing::instrument(level = "trace", skip(msgs))]
    async fn unicast(
        &mut self,
        msgs: &[impl serde::Serialize + Sync],
    ) -> Result<(), Self::UnicastError> {
        self.unicast(msgs).await
    }

    #[tracing::instrument(level = "debug", skip(msgs))]
    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Result<Vec<T>, Self::UnicastError>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Sync,
    {
        self.symmetric_unicast(msgs).await
    }

    #[tracing::instrument(level = "trace")]
    async fn receive_all<T: serde::de::DeserializeOwned + Send>(
        &mut self,
    ) -> Result<Vec<T>, Self::UnicastError> {
        self.receive_all().await
    }

    fn size(&self) -> usize {
        self.connections.len() + 1
    }
}

impl<C: SplitChannel> Broadcast for Network<C> {
    type BroadcastError = NetworkError<C::RecvError, C::SendError>;

    #[tracing::instrument(level = "trace", skip(msg))]
    async fn broadcast(
        &mut self,
        msg: &(impl serde::Serialize + Sync),
    ) -> Result<(), Self::BroadcastError> {
        self.broadcast(msg).await
    }

    #[tracing::instrument(level = "trace", skip(msg))]
    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, Self::BroadcastError>
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Sync,
    {
        self.symmetric_broadcast(msg).await
    }

    #[tracing::instrument(level = "trace")]
    fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        id: Id,
    ) -> impl Future<Output = Result<T, Self::BroadcastError>> + Send {
        Tuneable::recv_from(self, id)
    }

    fn size(&self) -> usize {
        self.connections.len() + 1
    }
}

impl<C: SplitChannel> Tuneable for Network<C> {
    type TuningError = NetworkError<C::RecvError, C::SendError>;

    fn id(&self) -> Id {
        self.id()
    }

    #[tracing::instrument(level = "trace")]
    async fn recv_from<T: serde::de::DeserializeOwned>(
        &mut self,
        id: Id,
    ) -> Result<T, Self::TuningError> {
        let idx = self.id_to_index(id);
        self.connections[idx]
            .recv()
            .await
            .map_err(|e| NetworkError::Incoming {
                id: idx as u32,
                source: e,
            })
    }

    #[tracing::instrument(level = "trace", skip(msg))]
    async fn send_to<T: serde::Serialize + Sync>(
        &mut self,
        id: Id,
        msg: &T,
    ) -> Result<(), Self::TuningError> {
        let idx = self.id_to_index(id);
        self.connections[idx]
            .send(msg)
            .await
            .map_err(|e| NetworkError::Outgoing {
                id: idx as u32,
                source: e,
            })
    }

    type Channel = C;
    fn channels(&mut self) -> &mut [C] {
        &mut self.connections
    }
}

impl<C: SplitChannel> Communicate for Network<C> {}

/// Network containing only duplex connections.
/// Used for local testing.
pub type InMemoryNetwork = Network<DuplexConnection>;

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

    #[tracing::instrument(level = "trace")]
    pub async fn shutdown(self) -> Result<(), std::io::Error> {
        let futs = self
            .connections
            .into_iter()
            .map(|conn| async move { conn.shutdown().await });
        join_all(futs).await.into_iter().map_ok(|_| {}).collect()
    }

    pub fn add_read_latency(self, delay: Duration) -> Network<impl SplitChannel> {
        let connections = self
            .connections
            .into_iter()
            .map(|c| c.delayed_read(delay))
            .collect();
        Network {
            connections,
            index: self.index,
        }
    }

    pub fn add_write_latency(self, delay: Duration) -> Network<impl SplitChannel> {
        let connections = self
            .connections
            .into_iter()
            .map(|c| c.delayed_write(delay))
            .collect();
        Network {
            connections,
            index: self.index,
        }
    }
}

/// TCP Network based on TCP Streams.
pub type TcpNetwork = Network<TcpConnection>;

impl TcpNetwork {
    /// Construct a TCP-based network by opening a socket and connecting to peers.
    /// If peers cannot be connected to, we wait until we wait until they have
    /// connected to us.
    ///
    /// * `me`: Socket address to open to
    /// * `peers`: Socket addresses of other peers
    #[tracing::instrument]
    pub async fn connect(me: SocketAddr, peers: &[SocketAddr]) -> NetResult<Self, TcpConnection> {
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

        let connections = parties
            .into_iter()
            .map(Connection::from_tcp_stream)
            .collect();

        let mut network = Self {
            connections,
            index: 0,
        };
        network.resolute_ids(&mut thread_rng()).await?;

        Ok(network)
    }

    #[tracing::instrument]
    pub async fn shutdown(self) -> Result<(), std::io::Error> {
        let futs = self
            .connections
            .into_iter()
            .map(|conn| async move { conn.shutdown().await });
        join_all(futs).await.into_iter().map_ok(|_| {}).collect()
    }
}

mod builder {
    use std::{net::SocketAddr, time::Duration};

    use crate::net::{
        connection::TcpConnection,
        network::{NetResult, TcpNetwork},
    };

    pub struct NetworkBuilder {
        delay: Option<Duration>,
    }

    pub struct TcpNetworkBuilder {
        parent: NetworkBuilder,
        addr: SocketAddr,
        parties: Vec<SocketAddr>,
    }

    impl NetworkBuilder {
        pub fn add_delay(mut self, lag: Duration) -> Self {
            self.delay = Some(lag);
            self
        }

        pub fn tcp(self, addr: SocketAddr) -> TcpNetworkBuilder {
            TcpNetworkBuilder {
                parent: self,
                addr,
                parties: vec![],
            }
        }
    }

    impl TcpNetworkBuilder {
        pub fn add_party(mut self, addr: SocketAddr) -> Self {
            self.parties.push(addr);
            self
        }

        pub fn add_parties(mut self, addrs: &[SocketAddr]) -> Self {
            self.parties.extend_from_slice(addrs);
            self
        }

        pub async fn connect(self) -> NetResult<TcpNetwork, TcpConnection> {
            TcpNetwork::connect(self.addr, &self.parties).await
        }
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
        const N: usize = 3;
        let players = Network::in_memory(N);
        let mut messages: VecDeque<_> =
            vec!["Over".to_string(), "And".to_string(), "Out".to_string()].into();

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
        const N: usize = 3;
        let players = Network::in_memory(N);
        let mut messages: VecDeque<_> = vec![[0, 1, 2], [0, 1, 2], [0, 1, 2]].into();
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
