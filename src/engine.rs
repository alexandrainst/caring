//! This a experiment for a structure to run MPC programs

use std::{future::Future, sync::Arc};
use std::net::SocketAddr;

use futures::future::{join_all, self};
use tokio::{sync::Mutex, net::tcp::{OwnedReadHalf, OwnedWriteHalf}};
use tokio::task;
use tokio::{net::TcpStream, runtime::Runtime, io::{AsyncReadExt, AsyncWriteExt}};


struct Party {
    id: u32,
    channel: (Mutex<OwnedReadHalf>, Mutex<OwnedWriteHalf>),
}

pub struct Engine {
    parties: Vec<Arc<Party>>,
    // runtime: Runtime,
}



impl Engine {

    pub async fn connect(my_addr: SocketAddr, peers: &[SocketAddr]) -> Self {
        // Connect to the initial parties.
        let n = peers.len();
        let parties = async {

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

            // We need a good method to distribute IDs
            
            parties.into_iter()
                .map(|channel| channel.into_split())
                .map(|(read, write)| Party {id: 0, channel: (Mutex::new(read), Mutex::new(write))})
                .map(Arc::new)
                .collect()
        };

        Engine { parties: parties.await }
    }


    // pub async fn execute<'a,T,P,F>(&'a mut self, prg: P) -> Option<T>
    //     where F : Future<Output=T> + Send, P: Fn(&'a mut Self) -> F, T: Send,
    // {
    //     let prg = prg(self);
    //     Some(task::spawn(prg).await.unwrap())
    // }

    // TODO: I am not sure if we should provide 'asyncronous' functions (not in the async sense),
    // in which parties can either send or recv, or if we only need to provide 'syncronous'
    // in which all parties do the same.

    // send the same value to all parties
    // this should be received with a corresponding check to ensure consistency.
    pub fn broadcast<const N: usize>(&mut self, msg: &[u8; N]) {
        let msg = Arc::new(*msg);
        for party in &mut self.parties {
            let party = party.clone();
            let msg = msg.clone();
            task::spawn(async move {
                party.channel.1.lock().await.write_all(&*msg).await.unwrap();
            });
        }
    }

    // recv from a broadcast 
    pub async fn recv_from_all<const N: usize>(&mut self) -> Vec<[u8; N]> {
        let mut results = Vec::new();

        for party in &mut self.parties {
            let party = party.clone();
            let handle = async move {
                let mut buf = [0u8; N];
                party.channel.0.lock().await.read_exact(&mut buf).await.unwrap();
                buf
            };
            results.push(handle);
        }

        future::join_all(results.into_iter()).await
    }


    // commit to a given value and broadcast the commitment.
    pub async fn commit(&mut self) {}

    // publish a commited value.
    pub async fn publish(&mut self) {}
}
