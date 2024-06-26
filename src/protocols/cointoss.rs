//! WIP: Very incomplete
use std::error::Error;

use crate::net::agency::Broadcast;

pub struct CoinToss<Rng: rand::RngCore> {
    rng: Rng,
    // hashing_alg: D,
}

impl<Rng: rand::RngCore> CoinToss<Rng> {
    pub fn new(rng: Rng) -> Self {
        Self { rng }
    }

    /// NOT COMPLETE
    pub async fn toss<const N: usize>(
        &mut self,
        cx: &mut impl Broadcast,
    ) -> Result<[u8; N], Box<dyn Error>> {
        let mut my_seed = [0u8; N];
        self.rng.fill_bytes(&mut my_seed);
        // TODO: Commitment protocol
        let my_seed: Box<[u8]> = Box::new(my_seed);
        let seeds = cx.symmetric_broadcast(my_seed).await.unwrap();

        fn xor(acc: &mut [u8], ins: &[u8]) {
            acc.iter_mut().zip(ins).for_each(|(acc, ins)| *acc ^= ins);
        }
        let mut acc = [0u8; N];
        seeds.into_iter().for_each(|seed| xor(&mut acc, &seed));

        Ok(acc)
    }
}
