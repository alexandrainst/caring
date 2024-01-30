use ff::Field;
use rand::RngCore;

use derive_more::{Add, Sub};

#[repr(transparent)]
#[derive(Clone, Copy, Add, Sub)]
pub struct Share<F: Field>(F);


pub fn share<F: Field>(secret: F, rng: &mut impl RngCore) -> [Share<F>; 3] {
    let mut shares = [F::random(rng); 3];
    let sum = shares[0] + shares[1] + shares[2];
    shares[0] = secret - sum;

    [Share(shares[0]), Share(shares[1]), Share(shares[2])]
}


pub fn recombine<F: Field>(shares: &[Share<F>; 3]) -> F {
    shares[0].0 + shares[1].0 + shares[2].0
}
