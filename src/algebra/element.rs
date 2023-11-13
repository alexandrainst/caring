//! Concrete mathematical field example.
//! Here we have a prime field that is very close to 2^32.
use ff::{PrimeField, derive::subtle::{Choice, ConstantTimeEq, ConditionallySelectable}};
use rand::Rng;

#[derive(PrimeField, serde::Serialize, serde::Deserialize)]
#[PrimeFieldModulus = "4294967291"]
#[PrimeFieldGenerator = "2"]
#[PrimeFieldReprEndianness = "little"]
pub struct Element32([u64; 1]);

impl From<Element32> for u32 {
    /// Convert a element into u32
    ///
    /// * `val`: Element to convert
    fn from(val: Element32) -> Self {
        let arr = val.to_repr().0;
        let arr = [arr[0], arr[1], arr[2], arr[3]];
        u32::from_le_bytes(arr)
        // val.0[0] as u32
    }
}

impl From<u32> for Element32 {
    /// Create a element element from a u32
    ///
    /// * `val`: Element to convert
    fn from(val: u32) -> Self {
        let val = val.to_le_bytes();
        // NOTE: Should probably mention that this is vartime.
        // TODO: Maybe this fails if the integer is bigger than the modulus?
        Element32::from_repr_vartime(Element32Repr([val[0], val[1], val[2], val[3], 0, 0, 0, 0]))
            .unwrap()
    }
}

impl From<Element32> for u64 {
    fn from(val: Element32) -> Self {
        val.0[0]
    }
}

#[derive(Default, PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Mod11(pub(crate) u8);

// Here be lot's of trait implementations
// I would like either specialization or macros please.
impl std::ops::Add for Mod11 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self((self.0 + rhs.0) % 11)
    }
}

impl<'a> std::ops::Add for &'a Mod11 {
    type Output = Mod11;

    fn add(self, rhs: Self) -> Self::Output {
        Mod11((self.0 + rhs.0) % 11)
    }
}

impl<'a> std::ops::Add<&'a Mod11> for Mod11 {
    type Output = Mod11;

    fn add(self, rhs: &'a Self) -> Self::Output {
        Self((self.0 + rhs.0) % 11)
    }
}

impl std::ops::Sub for Mod11 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> std::ops::Sub for &'a Mod11 {
    type Output = Mod11;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-*rhs)
    }
}

impl<'a> std::ops::Sub<&'a Mod11> for Mod11 {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self::Output {
        self + (-*rhs)
    }
}

impl std::ops::Neg for Mod11 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(11 - self.0)
    }
}


impl std::ops::Mul for Mod11 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<'a> std::ops::Mul for &'a Mod11 {
    type Output = Mod11;

    fn mul(self, rhs: Self) -> Self::Output {
        Mod11((self.0 * rhs.0) % 11)
    }
}

impl<'a> std::ops::Mul<&'a Mod11> for Mod11 {
    type Output = Self;

    fn mul(self, rhs: &'a Self) -> Self::Output {
        Self((self.0 * rhs.0) % 11)
    }
}


impl std::ops::MulAssign<&Mod11> for Mod11 {
    fn mul_assign(&mut self, rhs: &Mod11) {
        self.0 *= rhs.0;
        self.0 %= 11;
    }
}

impl std::ops::MulAssign<Mod11> for Mod11 {
    fn mul_assign(&mut self, rhs: Mod11) {
        self.0 *= rhs.0;
        self.0 %= 11;
    }
}

impl std::ops::SubAssign<&Mod11> for Mod11 {
    fn sub_assign(&mut self, rhs: &Mod11) {
        self.0 += (-*rhs).0;
        self.0 %= 11;
    }
}

impl std::ops::SubAssign<Mod11> for Mod11 {
    fn sub_assign(&mut self, rhs: Mod11) {
        self.0 += (-rhs).0;
        self.0 %= 11;
    }
}

impl std::ops::AddAssign<&Mod11> for Mod11 {
    fn add_assign(&mut self, rhs: &Mod11) {
        self.0 += rhs.0;
        self.0 %= 11;
    }
}

impl std::ops::AddAssign<Mod11> for Mod11 {
    fn add_assign(&mut self, rhs: Mod11) {
        self.0 += rhs.0;
        self.0 %= 11;
    }
}

impl ConstantTimeEq for Mod11 {
    fn ct_eq(&self, other: &Self) -> Choice {
        ((self == other) as u8).into()
    }
}

impl std::iter::Sum for Mod11 {
    fn sum<I: Iterator<Item = Mod11>>(iter: I) -> Self {
        todo!()
    }
}

impl std::iter::Product for Mod11 {
    fn product<I: Iterator<Item = Mod11>>(iter: I) -> Self {
        todo!()
    }
}


impl<'a> std::iter::Sum<&'a Mod11> for &'a Mod11 {
    fn sum<I: Iterator<Item = &'a Mod11>>(iter: I) -> Self {
        todo!()
    }
}


impl<'a> std::iter::Sum<&'a Mod11> for Mod11 {
    fn sum<I: Iterator<Item = &'a Mod11>>(iter: I) -> Self {
        todo!()
    }
}

impl<'a> std::iter::Product<&'a Mod11> for &'a Mod11 {
    fn product<I: Iterator<Item = &'a Mod11>>(iter: I) -> Self {
        todo!()
    }
}

impl<'a> std::iter::Product<&'a Mod11> for Mod11 {
    fn product<I: Iterator<Item = &'a Mod11>>(iter: I) -> Self {
        todo!()
    }
}
impl ConditionallySelectable for Mod11 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let c = choice.unwrap_u8();
        Self(c * a.0 + (1-c) * b.0)
    }
}


impl ff::Field for Mod11 {
    const ZERO: Self = Self(0);

    const ONE: Self = Self(1);

    fn random(mut rng: impl rand::RngCore) -> Self {
        let x : u8 = rng.gen();
        Self(x % 11)
    }

    fn square(&self) -> Self {
        Self((self.0 * self.0) % 11)
    }

    fn double(&self) -> Self {
        Self((self.0 * 2) % 11)
    }

    fn invert(&self) -> ff::derive::subtle::CtOption<Self> {
        // let lookup : [u8; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let lookup : [u8; 11] = [0, 1, 6, 4, 3, 9, 2, 8, 7, 5, 10];
        ff::derive::subtle::CtOption::new(Self(lookup[self.0 as usize]), 1.into())
    }

    fn sqrt_ratio(_num: &Self, _div: &Self) -> (ff::derive::subtle::Choice, Self) {
        todo!()
    }

    fn is_zero(&self) -> Choice {
        self.ct_eq(&Self::ZERO)
    }

    fn is_zero_vartime(&self) -> bool {
        self.is_zero().into()
    }

    fn cube(&self) -> Self {
        self.square() * self
    }

    fn sqrt_alt(&self) -> (Choice, Self) {
        Self::sqrt_ratio(self, &Self::ONE)
    }

    fn sqrt(&self) -> ff::derive::subtle::CtOption<Self> {
        let (is_square, res) = Self::sqrt_ratio(self, &Self::ONE);
        ff::derive::subtle::CtOption::new(res, is_square)
    }

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                res = res.square();
                let mut tmp = res;
                tmp *= self;
                res.conditional_assign(&tmp, (((*e >> i) & 1) as u8).into());
            }
        }
        res
    }

    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                res = res.square();

                if ((*e >> i) & 1) == 1 {
                    res *= self;
                }
            }
        }

        res
    }
}
