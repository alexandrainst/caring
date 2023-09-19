use std::ops::{Add, Mul, Sub, AddAssign, MulAssign, SubAssign, Neg};

use rand::RngCore;

pub trait Field: Eq + Neg<Output=Self> + Add<Output=Self> + Mul<Output=Self> + Sub<Output=Self> + AddAssign + MulAssign + SubAssign + Sized + Clone + Copy {
    const ZERO : Self;

    const ONE : Self;

    fn random(rng: &mut impl RngCore) -> Self;

    fn pow(self, exp: u64) -> Self;

    fn invert(self) -> Self;
}


impl<F: ff::Field> Field for F {
    const ZERO : F = F::ZERO;
    const ONE : F = F::ONE;

    fn random(rng: &mut impl RngCore) -> Self {
        F::random(rng)
    }

    fn pow(self, exp: u64) -> Self {
        F::pow(&self, &[exp])
    }

    fn invert(self) -> Self {
        F::invert(&self).unwrap()
    }
}
