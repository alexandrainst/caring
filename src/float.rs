use crate::field;

// HACK: Hack together a field to represent real-numbers.
impl field::MyField for f64 {
    const ZERO : Self = 0.0;

    const ONE : Self = 1.0;

    fn random(rng: &mut impl rand::RngCore) -> Self {
        0.0 + (rng.next_u64() as i64) as f64
    }

    fn pow(self, exp: u64) -> Self {
        f64::powi(self, exp as i32)
    }

    fn invert(self) -> Self {
        1.0 / self
    }
}
