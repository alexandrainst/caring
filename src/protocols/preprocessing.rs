// Preprocessing 
    // Making triplets
    // Making random values known to some specific party
    // MAC'ing elements

// To do the preprosessing we need some HE scheme 

// This has to be done interactivly - look at triplets for inspiration

use ff::PrimeField;
use rand::RngCore;

use crate::{
    net::agency::Broadcast, 
    schemes::spdz::{self, SpdzContext}
};

#[derive(Debug)]
pub struct PreprocessedValues<F: PrimeField> {
    pub triplets: Vec<MultiplicationTriple<F>>,
    pub rand_known_to_i: RandomKnownToPi<F>, // consider boxed slices for the outer vec
    pub rand_known_to_me: RandomKnownToMe<F>,
}

#[derive(Debug)]
pub struct MultiplicationTriple<F: PrimeField> {
    pub shares: (spdz::Share<F>, spdz::Share<F>, spdz::Share<F>),
}

#[derive(Debug)]
pub struct RandomKnownToPi<F: PrimeField>{
    pub shares: Vec<Vec<spdz::Share<F>>>,
}

#[derive(Debug)]
pub struct RandomKnownToMe<F: PrimeField>{
    pub shares_and_vals: Vec<(spdz::Share<F>, F)>,
}