/// This is a simple commitment scheme.
/// It is outsorsed to a module, so it can be easily relapaced
// ToDo: consider making a trait.
use ff::PrimeField;
use rand::RngCore;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::{
    //schemes::spdz::{self, SpdzContext},
    algebra::element::Element32,
    net::agency::Broadcast,
    schemes::spdz,
};
// TODO: Find a hashing algorithm of cryptograpic standart to use in commitments
// TODO: Consider whether we need a more complex commitment scheme
// TODO: Make a module for the commitment scheme and move it to another file.
#[derive(PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Commitment {
    commitment: u64,
}
#[derive(PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Salt {
    salt: u64,
}
#[derive(Hash)]
struct Commitable {
    val: u64,
    salt: u64,
}
fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
// TODO: The randoness should be based on some randomness generator that is pased on to the commitment
pub fn commit<F: PrimeField + std::convert::Into<u64>>(val: F) -> (Commitment, Salt) {
    // TODO: This is not a propper way to make a random value - there should be a randomness generator in the context
    use ff::Field;
    let mut rng = rand::rngs::mock::StepRng::new(42, 7);
    let salt = Element32::random(&mut rng).into();
    (make_commit(val.into(), salt), Salt { salt })
}
fn make_commit(val: u64, salt: u64) -> Commitment {
    let c = Commitable { val, salt };
    Commitment {
        commitment: calculate_hash(&c),
    }
}
pub fn verify_commit<F: PrimeField + std::convert::Into<u64>>(
    val: &F,
    commitment: &Commitment,
    salt: &Salt,
) -> bool {
    let commitment_2 = make_commit((*val).into(), salt.salt);
    *commitment == commitment_2
}
