//! This a experiment for a structure to run MPC programs

use std::future::Future;

struct Channel {}

struct Party {
    id: u32,
    channel: Channel,
}

impl Party {
    pub async fn send<const N: usize>(&mut self, msg: impl Into<[u8; N]>) {
        let payload : [u8; N] = msg.into();
        todo!()
    }

    pub async fn recv<const N: usize, T>(&mut self) -> T where T: From<[u8; N]> {
        let zeroes = [0; N];
        T::from(zeroes)
    }
}


struct Engine {
    parties: Vec<Party>,
}

impl Engine {

    pub fn new() -> Self { Engine { parties: Vec::new() }}

    pub fn execute<Prg>(&mut self, prg: Prg) where Prg: FnMut(&mut Engine) {


    }
    // TODO: I am not sure if we should provide 'asyncronous' functions (not in the async sense),
    // in which parties can either send or recv, or if we only need to provide 'syncronous'
    // in which all parties do the same.

    // send the same value to all parties
    // this should be received with a corresponding check to ensure consistency.
    pub async fn broadcast<const N: usize>(&mut self, msg: impl Into<[u8; N]>) {
        let msg : [u8; N] = msg.into();
        for party in &mut self.parties {
            party.send(msg);
        }
    }

    // commit to a given value and broadcast the commitment.
    pub async fn commit() {}

    // publish a commited value.
    pub async fn publish() {}

    // recv from a broadcast by a specific party
    pub async fn recv_broadcast() {}

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple() {
        let mut engine = Engine::new();
        engine.execute(|eng| {
            println!("hello world");
        });

    }

}
