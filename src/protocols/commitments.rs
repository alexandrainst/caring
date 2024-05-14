//use ff::derive::bitvec::view::AsBits;
//use ff::derive::bitvec::view::AsMutBits;
/// This is a simple commitment scheme.
/// It is outsorsed to a module, so it can be easily relapaced
//use ff::Field;
use ff::PrimeField;
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use rand::thread_rng;


// TODO: Find a hashing algorithm of cryptograpic standart to use in commitments
// TODO: Consider whether we need a "real" commitment scheme
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
struct Values {
    values: Vec<u8>,
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
// TODO: The randoness should be based on some randomness generator that is pased on to the commitment
pub fn commit<F: PrimeField  + serde::Serialize>(val: &F) -> (Commitment, Salt) {
    // TODO: consider whether this is the way we wish to make the random value.
    let mut rng = thread_rng();
    let r_salt: u64 = rng.gen();
    let salt = Salt {
        salt: r_salt,
    };
    let vals = Values{values: bincode::serialize(val).unwrap()};
    (make_commit(vals, salt), salt)
}
pub fn commit_many<F: PrimeField + serde::Serialize>(vals: &Vec<F>) -> (Commitment, Salt) {
    let mut rng = thread_rng();
    let r_salt: u64 = rng.gen();
    let salt = Salt {
        salt: r_salt,
    };
    let mut values = vec![];
    for v in vals {
        values.append(&mut bincode::serialize(v).unwrap());
    }
    (make_commit(Values { values }, salt), salt)
}
fn make_commit(vals: Values, salt: Salt) -> Commitment {
    let c = Commitable { vals, salt };
    Commitment {
        commitment: calculate_hash(&c),
    }
}

pub fn verify_commit<F: PrimeField + serde::Serialize>(
    val: &F,
    commitment: &Commitment,
    salt: &Salt,
) -> bool {
    let vals = Values {
        values: bincode::serialize(val).unwrap(),
    };
    let commitment_2 = make_commit(vals, *salt);
    *commitment == commitment_2
}

pub fn verify_many<F: PrimeField+ serde::Serialize>(
    vals: &Vec<F>,
    commitment: &Commitment,
    salt: &Salt,
) -> bool {
    let mut values = vec![];
    for v in vals {
        values.append(&mut bincode::serialize(v).unwrap());
    }
    let commitment_2 = make_commit(Values { values }, *salt);
    *commitment == commitment_2
}
// TODO: consider unit testing commitments
