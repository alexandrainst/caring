/// This is a simple commitment scheme.
/// It is outsorsed to a module, so it can be easily relapaced
// ToDo: consider making a trait.
// ToDo: The  sollution with the hashing is very hacky - consider making something nicer.
use ff::PrimeField;
use ff::Field;
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
#[derive(PartialEq, serde::Serialize, serde::Deserialize, Hash, Clone, Copy)]
pub struct Salt {
    salt: u64,
}
#[derive(Hash)]
struct Commitable {
    vals: Values,
    salt: Salt,
}
#[derive(Hash)]
struct Values{
    values: Vec<u64>
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
// TODO: The randoness should be based on some randomness generator that is pased on to the commitment
pub fn commit<F: PrimeField + std::convert::Into<u64>>(val: &F) -> (Commitment, Salt) {
    // TODO: This is not a propper way to make a random value - there should be a randomness generator in the context
    let mut rng = rand::rngs::mock::StepRng::new(42, 7);
    let salt = Salt{salt: Element32::random(&mut rng).into()};
    let vals = Values{values: vec![(*val).into()]};
    (make_commit(vals, salt), salt)
}
pub fn commit_many<F: PrimeField + std::convert::Into<u64>>(vals: &Vec<F>) -> (Commitment, Salt){
    // TODO: This is not a propper way to make a random value - there should be a randomness generator in the context
    let mut rng = rand::rngs::mock::StepRng::new(42, 7);
    let salt = Salt{salt:Element32::random(&mut rng).into()};
    let mut values = vec![];
    for v in vals {
        values.push((*v).into());
    }
    (make_commit(Values {values}, salt), salt)
}
fn make_commit(vals: Values, salt: Salt) -> Commitment {
    //let val = Values{values: vec![val]};
    let c = Commitable { vals, salt };
    Commitment {
        commitment: calculate_hash(&c),
    }
}
pub fn verify_commit<F: PrimeField + std::convert::Into<u64>>(
    val: &F,
    commitment: &Commitment,
    salt: &Salt,
) -> bool {
    let vals = Values{values: vec![(*val).into()]};
    let commitment_2 = make_commit(vals, *salt);
    *commitment == commitment_2
}

pub fn verify_many<F: PrimeField + std::convert::Into<u64>>(
    vals: &Vec<F>,
    commitment: &Commitment,
    salt: &Salt,
) -> bool {
    let mut values = vec![];
    for v in vals {
        values.push((*v).into());
    }
    let commitment_2 = make_commit(Values {values}, *salt);
    *commitment == commitment_2
}
// TODO: consider unit testing commitments
