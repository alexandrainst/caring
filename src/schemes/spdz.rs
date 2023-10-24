//! (Some version of) SPDZ
//!
//! The idea is to implement the bare-bones version of spdz, probably a more recent one.
//! We will probably need some homomorphic encryption or oblivious transfer to enable this.
//!
//! Further more stands the issue of sharing the input in a nice way, since a lot of
//! preprocessing is needed in spdz beforehand.
//!
//! The spdz(2k) scheme is actually not that difficult in the online phase,
//! however it requres some heavy machinery in the offline phase.
//!
use ff::Field;

use derive_more::{Add, AddAssign, Sub, SubAssign};
use rand::RngCore;

#[derive(Clone, Copy, Add, AddAssign)]
pub struct Share<F: Field>{val: F, mac: F}

impl<F: Field> Share<F> {

    pub fn validate(&self, key: F) -> bool {
        let Share{val, mac} = *self;
        val * key == mac
    }
}

impl<F: Field> std::ops::Mul<F> for Share<F> {
    type Output = Share<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Share{val:self.val * rhs, mac: self.mac * rhs}
    }
}

// TODO: Implement multiplication between shares.

pub fn share<F: Field>(val: F, n: usize, key: F, mut rng: &mut impl RngCore) -> Vec<Share<F>> {
    // HACK: This is really not secure at all.
    let mut shares : Vec<_> = (0..n).map(|_| {
        F::random(&mut rng)
    }).collect();

    let sum : F = shares.iter().sum();
    shares[0] -=  sum - val;
    // In Fresco, this is all very interactive

    shares.into_iter().map(|x| {
        Share{val: x, mac: key * x}
    }).collect()
}

pub fn input<F: Field>(val: F, n: usize) -> Vec<Share<F>> {

    // 1. Everyone sends party `i` their share (partial opening)
    // 2. Party `i` then broadcasts `x^i - r`.
    //    Party 1 sets their share to `x^i_1 = r1 + x^i - r`.
    //    (In practice this might not always be 1)
    // 3. Profit

    todo!("Implement the function")
}

pub fn reconstruct<F: Field>(shares: &[Share<F>]) -> F {
    shares.iter().map(|x| x.val).sum()
}

// TODO: Round syncronization

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn sharing() {
        use crate::element::Element32;
        let mut rng  = rand::rngs::mock::StepRng::new(42, 7);
        // Key should probablu be derived from a random somewhere
        let key  = 7u32.into();
        let v  = 42u32;
        let shares = share(Element32::from(v), 3, key, &mut rng);

        let v2 : u32 = reconstruct(&shares).into();
        assert_eq!(v, v2);
    }

}
