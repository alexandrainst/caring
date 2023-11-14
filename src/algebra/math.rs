//! Module for helper maths (mostly vectors)
//!
//! The key item here is the `Vector<F>` struct not to be confused with `std::vec::Vec`,
//! which is a growable array type. `Vector<F>` is backed by a boxed slice, since we usally don't
//! want to grow or shrink our mathematical vectors. (Subject to change if we want to use the stack
//! instead)
//!
//! Addition and subtraction function elementwise, while multiplication is only allowed with scalar
//! values.
//!
use std::ops::AddAssign;

use rayon::prelude::*;
use derive_more::*;

// TODO: Consider smallvec or tinyvec

/// Represention of a mathematical vector of type `F`.
///
/// This provides implementations for addition, subtraction and multiplication
/// as in ordinary linear algebra fashion.
///
/// If the rayon feature is enabled the operations will be parallelized.
#[derive(Clone, Debug, Index, PartialEq, Eq)]
struct Vector<F: Send + Sync> (
    Box<[F]>
);

impl<F: Send + Sync> Vector<F> {
    pub fn from_boxed_slice(slice: Box<[F]>) -> Self {Self(slice)}
    pub fn from_vec(v: Vec<F>) -> Self {
        Self(v.into_boxed_slice())
    }
    pub fn from_array<const N: usize>(v: [F; N]) -> Self {
        Self(Box::new(v))
    }
}

impl<'a, F: Send + Sync> IntoIterator for &'a Vector<F> {
    type Item = &'a F;

    type IntoIter = std::slice::Iter<'a, F>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}



impl<F: Send + Sync> IntoIterator for Vector<F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}


impl<T: Send + Sync> AsRef<[T]> for Vector<T> {
    fn as_ref(&self) -> &[T] { &self.0 }
}

impl<T: Send + Sync> AsMut<[T]> for Vector<T> {
    fn as_mut(&mut self) -> &mut [T] { &mut self.0 }
}

impl<A: Send + Sync, B: Send + Sync> std::ops::Add for &Vector<A> where for<'a> &'a A: std::ops::Add<Output=B> {
    type Output = Vector<B>;

    fn add(self, rhs: Self) -> Self::Output {
        let internal = if cfg!(feature = "rayon") {
            self.0
                .par_iter()
                .zip(rhs.0.par_iter())
                .map(|(a, b)| a + b)
                .collect()
        } else {
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(a, b)| a + b)
                .collect()
        };
        Vector(internal)
    }
}


impl<A: Send + Sync> std::ops::AddAssign<&Vector<A>> for Vector<A> where A: for<'a> std::ops::AddAssign<&'a A> {
    fn add_assign(&mut self, rhs: &Vector<A>) {
        if cfg!(feature = "rayon") {
            self.0
                .par_iter_mut()
                .zip(rhs.0.par_iter())
                .for_each(|(a, b)| *a += b)
        } else {
            self.0
                .iter_mut()
                .zip(rhs.0.iter())
                .for_each(|(a, b)| *a += b)
        }
    }
}

impl<A: Send + Sync> std::ops::AddAssign for Vector<A> where for<'a> A: std::ops::AddAssign<&'a A> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<A: Send + Sync> std::ops::Add<&Self> for Vector<A> where A: for<'a> std::ops::AddAssign<&'a A> {
    type Output = Vector<A>;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<A: Send + Sync> std::ops::Add<Self> for Vector<A> where A: for<'a> std::ops::AddAssign<&'a A> {
    type Output = Vector<A>;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}


impl<A: Send + Sync, B: Send + Sync> std::ops::Sub for &Vector<A> where for<'a> &'a A: std::ops::Sub<Output=B> {
    type Output = Vector<B>;

    fn sub(self, rhs: Self) -> Self::Output {
        let internal = if cfg!(feature = "rayon") {
            self.0
                .par_iter()
                .zip(rhs.0.par_iter())
                .map(|(a, b)| a - b)
                .collect()
        } else {
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(a, b)| a - b)
                .collect()
        };
        Vector(internal)
    }
}

impl<A: Send + Sync> std::ops::SubAssign<&Vector<A>> for Vector<A> where A: for<'a> std::ops::SubAssign<&'a A> {
    fn sub_assign(&mut self, rhs: &Vector<A>) {
        if cfg!(feature = "rayon") {
            self.0
                .par_iter_mut()
                .zip(rhs.0.par_iter())
                .for_each(|(a, b)| *a -= b)
        } else {
            self.0
                .iter_mut()
                .zip(rhs.0.iter())
                .for_each(|(a, b)| *a -= b)
        }
    }
}

impl<A: Send + Sync> std::ops::Sub<&Self> for Vector<A> where A: for<'a> std::ops::SubAssign<&'a A> {
    type Output = Vector<A>;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
    }
}


impl<A: Send + Sync> std::ops::Sub<Self> for Vector<A> where A: for<'a> std::ops::SubAssign<&'a A> {
    type Output = Vector<A>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}



impl<A: Send + Sync, B: Send + Sync> std::ops::MulAssign<&B> for Vector<A> where for<'b> A: std::ops::MulAssign<&'b B>{
    fn mul_assign(&mut self, rhs: &B) {
        let b = rhs;
        if cfg!(feature = "rayon") {
            // self.0
            //     .par_iter_mut()
            //     .for_each(|a| *a *= b)
            todo!("error")
        } else {
            self.0
                .iter_mut()
                .for_each(|a| *a *= b)
        }
    }
}


impl<A: Send + Sync, B: Send + Sync> std::ops::Mul<B> for &Vector<A> where for<'b, 'a> &'a A: std::ops::Mul<&'b B, Output=A>{
    type Output = Vector<A>;

    fn mul(self, rhs: B) -> Self::Output {
        let b = rhs;
        let internal = if cfg!(feature = "rayon") {
            self.0
                .par_iter()
                .map(|a| a * &b)
                .collect()
        } else {
            self.0
                .iter()
                .map(|a| a * &b)
                .collect()
        };
        Vector(internal)
    }
}


impl<A: Send + Sync, B: Send + Sync> std::ops::Mul<B> for Vector<A> where for<'b> A: std::ops::MulAssign<&'b B> {
    type Output = Vector<A>;

    fn mul(mut self, rhs: B) -> Self::Output {
        self *= &rhs;
        self
    }
}


impl<F: Send + Sync> std::iter::Sum for Vector<F> where F: for<'a> std::ops::AddAssign<&'a F> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut first = iter.next().expect("Please don't sum zero items together");
        for elem in iter {
            first += elem;
        }
        first
    }
}

#[cfg(test)]
mod test {
    use crate::algebra::element::Mod11;

    use super::*;

    #[test]
    fn add() {
        let a = Vector::from_array([1u32,2u32]);
        let b = Vector::from_array([3u32,4u32]);
        let c : Vector<_> = a + b;
        let c2 = Vector::from_array([4u32,6u32]);
        assert_eq!(c, c2);
    }


    #[test]
    fn sub() {
        let a = Vector::from_array([Mod11(3), Mod11(4)]);
        let b = Vector::from_array([Mod11(1), Mod11(2)]);
        let c : Vector<_> = a - b;
        let c2 = Vector::from_array([Mod11(2), Mod11(2)]);
        assert_eq!(c, c2);
    }


    #[test]
    fn mul() {
        let a = Vector::from_array([Mod11(3), Mod11(4)]);
        let b = Mod11(2);
        let c : Vector<_> = a * b;
        let c2 = Vector::from_array([Mod11(6), Mod11(8)]);
        assert_eq!(c, c2);
    }
}

