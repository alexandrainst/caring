//! This a experiment for a structure to run MPC programs

use std::{sync::Arc, collections::HashMap};
use std::net::SocketAddr;

use futures::future::{join_all, self};
use rand::{thread_rng, Rng};
use tokio::{sync::Mutex, net::tcp::{OwnedReadHalf, OwnedWriteHalf}};
use tokio::task;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// TODO: Find a better abstraction of streams.
// It would be nice if we could swap-out the underlying stream,
// i.e. TCP vs UDP or use TLS or maybe Unix sockets, or just an in-memory data stream for testing.
//
// The structure of this could either be a 'Stream/Channel' with Stream<T> for T being an
// underlying type. Or it could be an enum Stream with variants for each.
// The latter probably provides the best flexibility.

struct Party {
    id: PartyID,
    // TODO: Be generic over stream types
    channel: (Mutex<OwnedReadHalf>, Mutex<OwnedWriteHalf>),
}

#[repr(transparent)]
#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct PartyID(pub u32);

impl PartyID {
    pub fn idx(self) -> usize {
        (self.0) as usize
    }
}

pub struct Engine {
    // Should probably just be a map with ID's to parties.
    // However we still need a good way to get IDs.
    // One way could be to do as Fresco does and just supply them in the beginning.
    // I, however, find that wildly annoying when testing.
    // Sure, in the real world we would know who we talk with before hand,
    // but then we would also have certificates and other stuff.
    //
    // A quick and dirty solution to ID resolution would just let everyone
    // pick a random number and then sort and enumerate them.
    // Then we just need a broadcast to ensure that everyone have the same IDs.
    // (Probably also sending the party with ID 'i' a message that we know they are 'i')
    //
    // If everything goes well we should then have some IDs, however they wouldn't be 'fair' IDs,
    // in the sense that a given party can always choose a low random value.
    // So don't depend anything on being number one!
    parties: Vec<Arc<Party>>,
    // runtime: Runtime,
    pub id: PartyID,
}



impl Engine {

    pub fn participants(&self) -> Vec<PartyID> {
        let mut vec : Vec<_> = self.parties.iter().map(|p| p.id).collect();
        vec.push(self.id);
        vec.sort_unstable();
        vec
    }

    pub async fn connect(my_addr: SocketAddr, peers: &[SocketAddr]) -> Self {
        // Connect to the initial parties.
        let n = peers.len();

        println!("Connecting to parties");
        let results = join_all(peers.iter()
            .map(|addr| tokio::task::spawn(tokio::net::TcpStream::connect(*addr)))
        ).await;

        let mut parties : Vec<_> = results.into_iter().map(|x| x.unwrap())
            .filter_map(|x| x.ok())
            .collect();

        // If we are not able to connect to some, they will connect to us.
        println!("Accepting connections");
        let incoming = tokio::net::TcpListener::bind(my_addr).await.unwrap();
        loop {
            if parties.len() >= n { break };
            let (stream, _) = incoming.accept().await.unwrap();
            parties.push(stream);
        }

        // ID resolution
        // It is unlikely to choose the same u64 twice.
        let mut rng = thread_rng();
        let num : u64 = rng.gen();
        let mut nums = Vec::new();
        nums.push((my_addr, num));

        let (mut readers, mut writers) : (Vec<_>, Vec<_>) = parties.iter_mut().map(|s| s.split()).unzip();
        for party in &mut writers {
            party.write_u64(num).await.unwrap();
        }
        for party in &mut readers {
            let num = party.read_u64().await.unwrap();
            let addr = party.peer_addr().unwrap();
            nums.push((addr, num));
        }
        nums.sort_unstable_by_key(|(_, n)| *n);
        let map : HashMap<_,_> = nums.iter().enumerate().map(|(i, (addr, _))| (addr, i)).collect();

        let mut parties : Vec<_> = parties.into_iter()
            .map(|channel| channel.into_split())
            .map(|(read, write)| {
                let id = *map.get(&read.peer_addr().unwrap()).unwrap();
                Party {id: PartyID(id as u32), channel: (Mutex::new(read), Mutex::new(write))}
            }).map(Arc::new).collect();

        let id = PartyID(*map.get(&my_addr).unwrap() as u32);
        parties.sort_unstable_by_key(|p| p.id.0);
        Engine { id, parties }
    }


    // pub async fn execute<'a,T,P,F>(&'a mut self, prg: P) -> Option<T>
    //     where F : Future<Output=T> + Send, P: Fn(&'a mut Self) -> F, T: Send,
    // {
    //     let prg = prg(self);
    //     Some(task::spawn(prg).await.unwrap())
    // }

    // TODO: I am not sure if we should provide 'asymmetric' functions
    // in which parties can either send or recv, or if we only need to provide 'symmetric'
    // in which all parties do the same.

    // send the same value to all parties
    // this should be received with a corresponding check to ensure consistency.

    /// Asymmetric broadcast
    ///
    /// Broadcasts (concurrently) the message with a given size.
    /// Combine with `recv_for_all` for a symmetric version
    ///
    /// * `msg`: Message to be broadcast
    pub fn broadcast<const N: usize>(&mut self, msg: &[u8; N]) {
        let msg = Arc::new(*msg);
        for party in &mut self.parties {
            let party = party.clone();
            let msg = msg.clone();
            // This awaits until the other parts receives the message.
            // Thus we spawn tasks so we can leave immediatly.
            task::spawn(async move {
                party.channel.1.lock().await.write_all(&*msg).await.unwrap();
            });
        }
    }

    /// Asymmetric receive from all
    ///
    /// This receives a `N`-sized byte message from all parties,
    /// except yourself.
    pub async fn recv_from_all<const N: usize>(&mut self) -> Vec<(PartyID, [u8; N])> {
        let mut results = Vec::new();

        for party in &mut self.parties {
            let handle = async move {
                let mut buf = [0u8; N];
                party.channel.0.lock().await.read_exact(&mut buf).await.unwrap();
                (party.id, buf)
            };
            results.push(handle);
        }

        future::join_all(results.into_iter()).await
    }

    // TODO: change msg to be anything serializeble.
    pub async fn symmetric_broadcast<const N: usize>(&mut self, msg: [u8; N]) -> Vec<(PartyID, [u8; N])> {
        self.broadcast(&msg);
        let mut vec = self.recv_from_all().await;
        vec.push((self.id, msg));
        vec
    }


    // commit to a given value and broadcast the commitment.
    pub async fn commit(&mut self) {}

    // publish a commited value.
    pub async fn publish(&mut self) {}

    /// Assymmetric send.
    pub async fn send(&mut self, msg: &impl serde::Serialize, recipiant: u32) {
        let i  = recipiant as usize;
        let msg = bincode::serialize(msg).unwrap();
        let mut writer = self.parties[i].channel.1.lock().await;
        writer.write_all(&msg).await.unwrap();
    }

    /// Assymetric receive.
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self, sender: u32) -> T {
        let i  = sender as usize;
        let mut reader = self.parties[i].channel.0.lock().await;
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).await.unwrap();
        let buf = std::io::Cursor::new(buf);
        bincode::deserialize_from(buf).unwrap()
    }
}

