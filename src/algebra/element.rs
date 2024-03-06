//! Concrete mathematical field example.
//! Here we have a prime field that is very close to 2^32.
use ff::{
    derive::subtle::{Choice, ConditionallySelectable, ConstantTimeEq},
    PrimeField,
};
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
        u64::from_le_bytes(val.to_repr().0)
    }
}

#[cfg(test)]
mod test {
    use crate::algebra::element::Element32;

    #[test]
    fn from_into_u64() {
        let n0 : u64 = 7;
        let e = Element32::from(n0);
        let n1 : u64 = e.into();
        assert_eq!(n0, n1);
    }

    #[test]
    fn from_into_u32() {
        let n0 : u32 = 7;
        let e = Element32::from(n0);
        let n1 : u32 = e.into();
        assert_eq!(n0, n1);
    }
}


use derive_more::{Product, Sum};

/// Modulo 11 Arithmetic
///
/// Not constant-time add-all.
/// Possibly inefficient with the amount of modulus operations used.
#[derive(
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    Clone,
    Copy,
    serde::Serialize,
    serde::Deserialize,
    Sum,
    Product,
)]
pub struct Mod11(pub(crate) u8);

use overload::overload;
use std::ops;

// Now instead of three implementations for each we just have one!
overload!((a: ?Mod11) + (b: ?Mod11) -> Mod11 {Mod11((a.0 + b.0) % 11)});
overload!((a: ?Mod11) - (b: ?Mod11) -> Mod11 {Mod11((a.0 + 11 - b.0) % 11)});
overload!((a: ?Mod11) * (b: ?Mod11) -> Mod11 {Mod11((a.0 * b.0) % 11)});
overload!(- (a: ?Mod11) -> Mod11 { Mod11(11 - a.0) });
overload!((a: &mut Mod11) += (b: ?Mod11) {
    a.0 += b.0;
    a.0 %= 11;
});
overload!((a: &mut Mod11) -= (b: ?Mod11) {
    a.0 += 11 - b.0;
    a.0 %= 11;
});
overload!((a: &mut Mod11) *= (b: ?Mod11) {
    a.0 *= b.0;
    a.0 %= 11;
});

impl ConstantTimeEq for Mod11 {
    fn ct_eq(&self, other: &Self) -> Choice {
        ((self == other) as u8).into()
    }
}

impl<'a> std::iter::Sum<&'a Mod11> for Mod11 {
    fn sum<I: Iterator<Item = &'a Mod11>>(iter: I) -> Self {
        iter.into_iter()
            .fold(<Mod11 as ff::Field>::ZERO, |acc, x| acc + x)
    }
}

impl<'a> std::iter::Product<&'a Mod11> for Mod11 {
    fn product<I: Iterator<Item = &'a Mod11>>(iter: I) -> Self {
        iter.into_iter()
            .fold(<Mod11 as ff::Field>::ONE, |acc, x| acc * x)
    }
}

impl ConditionallySelectable for Mod11 {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let c = choice.unwrap_u8();
        Self(c * a.0 + (1 - c) * b.0)
    }
}

impl ff::Field for Mod11 {
    const ZERO: Self = Self(0);

    const ONE: Self = Self(1);

    fn random(mut rng: impl rand::RngCore) -> Self {
        let x: u8 = rng.gen();
        Self(x % 11)
    }

    fn square(&self) -> Self {
        Self((self.0 * self.0) % 11)
    }

    fn double(&self) -> Self {
        Self((self.0 * 2) % 11)
    }

    fn invert(&self) -> ff::derive::subtle::CtOption<Self> {
        let lookup: [u8; 11] = [0, 1, 6, 4, 3, 9, 2, 8, 7, 5, 10];
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
