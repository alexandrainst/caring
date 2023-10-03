//! This a experiment for a structure to run MPC programs

use std::{future::Future, net::{IpAddr, TcpStream, SocketAddr, TcpListener, Shutdown}, collections::HashSet, io::{Write, Read}, pin::pin};


enum Channel {
    TcpChannel(TcpStream)
}

struct Party {
    id: u32,
    channel: Channel,
}

impl Party {
    pub async fn send<const N: usize>(&mut self, msg: impl Into<[u8; N]>) {
        let _payload : [u8; N] = msg.into();
        todo!()
    }

    pub async fn recv<const N: usize, T>(&mut self) -> T where T: From<[u8; N]> {
        let zeroes = [0; N];
        T::from(zeroes)
    }
}


pub struct Engine {
    parties: Vec<Party>,
}



impl Engine {

    pub fn connect(my_addr: SocketAddr, peers: &[SocketAddr]) -> Option<Self> {
        let mut parties = Vec::new();
        let mut missing = HashSet::new();

        // Connect to the initial parties.
        let mut buf = String::new();
        for addr in peers {
            if let Ok(mut stream) = TcpStream::connect(addr) {
                println!("Connected to {addr}");
                stream.read_to_string(&mut buf).unwrap();
                println!("{addr} says '{buf}'");
                parties.push(stream);
            } else {
                missing.insert(addr.ip());
            }
        }

        // If we are not able to connect to some, they will connect to us.
        while let Ok(incoming) = TcpListener::bind(my_addr) {
            if let Ok((mut stream, addr)) = incoming.accept() {
                if missing.take(&addr.ip()).is_none() {
                    eprintln!("Error!, {addr} is not supposed connect to us");
                    stream.shutdown(Shutdown::Both).unwrap();
                } else {
                    println!("{addr} connected");
                    write!(stream, "Hi buddy!").unwrap();
                    parties.push(stream);
                }
                if missing.is_empty() {
                    break
                }
            }
        }

        // Yea. We need async here.

        // We need a good method to distribute IDs
        let parties = parties.into_iter().map(Channel::TcpChannel)
            .map(|channel| Party {id: 0, channel}).collect();

        Some(Engine { parties })
    }


    pub fn execute<'a,T,P,F>(&'a mut self, prg: P) -> Option<T>
        where F : Future<Output=T>, P: Fn(&'a mut Self) -> F,
    {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let prg = prg(self);
        Some(runtime.block_on(prg))
    }
    // TODO: I am not sure if we should provide 'asyncronous' functions (not in the async sense),
    // in which parties can either send or recv, or if we only need to provide 'syncronous'
    // in which all parties do the same.

    // send the same value to all parties
    // this should be received with a corresponding check to ensure consistency.
    pub async fn broadcast<const N: usize>(&mut self, msg: impl Into<[u8; N]>) {
        let msg : [u8; N] = msg.into();
        let fut = self.parties.iter_mut()
            .map(|p| p.send(msg));
        futures::future::join_all(fut).await;
    }

    // commit to a given value and broadcast the commitment.
    pub async fn commit(&mut self) {}

    // publish a commited value.
    pub async fn publish(&mut self) {}

    // recv from a broadcast by a specific party
    pub async fn recv_broadcast(&mut self) {}

}
