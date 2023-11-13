//! Replacement trait for ff::Field, since it might be abit overkill.
use rand::RngCore;
use std::ops::*;

trait Field: Clone + Copy + Sized + Add<Output=Self> + Sub<Output=Self> + Neg<Output=Self> + Mul<Output=Self> {
    const ONE: Self;
    const ZERO: Self;

    fn random(rng: impl RngCore) -> Self;
    fn pow(&self, exp: u64) -> Self;
}


impl<T: ff::Field> Field for T {
    const ONE: Self = Self::ONE;

    const ZERO: Self = Self::ZERO;

    fn random(rng: impl RngCore) -> Self {
        ff::Field::random(rng)
    }

    fn pow(&self, exp: u64) -> Self {
        ff::Field::pow_vartime(self, [exp])
    }
}
