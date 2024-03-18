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

pub struct MultiplicationTriple<F: PrimeField> {
    pub shares: (spdz::Share<F>, spdz::Share<F>, spdz::Share<F>),
}