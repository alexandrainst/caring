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

// TODO:
// Reconsider most of this, as this is just a more primitive manual implementation of something,
// like `ndarray`.

// TODO: Consider smallvec or tinyvec
// TODO: Make parallel version it's own type switch on them using cfg..

use std::ops::{Add, AddAssign, MulAssign, Sub, SubAssign};

use ff::Field;

/// Represention of a mathematical vector of type `F`.
///
/// This provides implementations for addition, subtraction and multiplication
/// as in ordinary linear algebra fashion.
///
/// If the rayon feature is enabled the operations will be parallelized.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Vector<F>(pub(super) Box<[F]>);

impl<F> Vector<F> {
    pub const fn from_boxed_slice(slice: Box<[F]>) -> Self {
        Self(slice)
    }

    pub fn from_vec(v: Vec<F>) -> Self {
        Self(v.into_boxed_slice())
    }

    pub fn from_array<const N: usize>(v: [F; N]) -> Self {
        Self(Box::new(v))
    }

    pub fn size(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<F> {
        self.0.iter()
    }

    pub fn into_boxed_slice(self) -> Box<[F]> {
        self.0
    }
}

impl<A> Vector<A> {
    pub fn scalar_mul<B>(&mut self, scalar: &B)
    where
        for<'b> A: MulAssign<&'b B>,
    {
        self.0.iter_mut().for_each(|a| *a *= scalar);
    }
}

pub trait RowMult<T> {
    fn row_wise_mult(&mut self, slice: &[T]);
}

impl<A, B> RowMult<B> for Vector<A>
where
    for<'b> A: MulAssign<&'b B>,
{
    fn row_wise_mult(&mut self, slice: &[B]) {
        self.0
            .iter_mut()
            .zip(slice.iter())
            .for_each(|(a, b)| *a *= b);
    }
}

impl<F> super::Length for Vector<F> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<F> std::ops::Deref for Vector<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> std::ops::DerefMut for Vector<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F> std::ops::Index<usize> for Vector<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<F> std::ops::IndexMut<usize> for Vector<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F> IntoIterator for Vector<F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}

impl<F> FromIterator<F> for Vector<F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let boxed = iter.into_iter().collect();
        Self(boxed)
    }
}

impl<T: Send + Sync> AsRef<[T]> for Vector<T> {
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T: Send + Sync> AsMut<[T]> for Vector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value.into_boxed_slice())
    }
}

impl<T> From<Vector<T>> for Vec<T> {
    fn from(value: Vector<T>) -> Self {
        value.0.into()
    }
}

macro_rules! inherent {
    ($trait2:ident, $fun2:ident, $trait1:ident, $fun1:ident) => {
        impl<A> $trait1<&Vector<A>> for Vector<A>
        where
            A: for<'a> std::ops::$trait1<&'a A>,
        {
            fn $fun1(&mut self, rhs: &Self) {
                self.0
                    .iter_mut()
                    .zip(rhs.0.iter())
                    .for_each(|(a, b)| $trait1::$fun1(a, b));
            }
        }

        impl<A> $trait1 for Vector<A>
        where
            for<'a> A: $trait1<&'a A>,
        {
            fn $fun1(&mut self, rhs: Self) {
                $trait1::$fun1(self, &rhs)
            }
        }

        impl<A> $trait2<&Self> for Vector<A>
        where
            A: for<'a> $trait1<&'a A>,
        {
            type Output = Vector<A>;

            fn $fun2(mut self, rhs: &Self) -> Self::Output {
                $trait1::$fun1(&mut self, rhs);
                self
            }
        }

        impl<A> $trait2<Self> for Vector<A>
        where
            A: for<'a> $trait1<&'a A>,
        {
            type Output = Vector<A>;

            fn $fun2(mut self, rhs: Self) -> Self::Output {
                $trait1::$fun1(&mut self, rhs);
                self
            }
        }

        impl<A> $trait2<Vector<A>> for &Vector<A>
        where
            A: for<'a> $trait1<&'a A>,
        {
            type Output = Vector<A>;

            fn $fun2(self, mut rhs: Vector<A>) -> Self::Output {
                $trait1::$fun1(&mut rhs, self);
                rhs
            }
        }
    };
}

inherent!(Add, add, AddAssign, add_assign);
inherent!(Sub, sub, SubAssign, sub_assign);

impl<F> std::ops::Neg for Vector<F>
where
    F: std::ops::Neg<Output = F>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.into_iter().map(|x| -x).collect()
    }
}

/// Generic implementation of row-wise mult-assign
///
/// Vector<A> =* B
/// if A *= B exists
///
impl<A, B> std::ops::MulAssign<&B> for Vector<A>
where
    for<'b> A: std::ops::MulAssign<&'b B>,
{
    fn mul_assign(&mut self, rhs: &B) {
        let b = rhs;
        self.0.iter_mut().for_each(|a| *a *= b)
    }
}

/// Generic implementation of row-wise multiplication
///
/// Vector<A> * B -> Vector<B>
/// if A * B -> B
///
/// See `Polynomial` for usage scenario
///
/// This is actually super-useful
///
impl<A, B> std::ops::Mul<B> for &Vector<A>
where
    for<'a, 'b> &'a A: std::ops::Mul<&'b B, Output = B>,
{
    type Output = Vector<B>;

    fn mul(self, rhs: B) -> Self::Output {
        let b = rhs;
        let internal = self.0.iter().map(|a| a * &b).collect();
        Vector(internal)
    }
}

/// Generic implemtentation of row-wise multiplication
///
/// Vector<A> * B -> Vector<A>
/// if A *= B exists
///
impl<A, B> std::ops::Mul<B> for Vector<A>
where
    for<'b> A: std::ops::MulAssign<&'b B>,
{
    type Output = Vector<A>;

    fn mul(mut self, rhs: B) -> Self::Output {
        self *= &rhs;
        self
    }
}

/// Generic implemetantion of sum
///
/// Sum(...Vector<A>) -> Vector<A>
/// if A += A exists
///
impl<F: Send + Sync> std::iter::Sum for Vector<F>
where
    F: for<'a> std::ops::AddAssign<&'a F>,
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut first = iter.next().expect("Please don't sum zero items together");
        for elem in iter {
            first += elem;
        }
        first
    }
}

impl<F: Field> Vector<F> {
    pub fn inner_product(&self, other: &Self) -> F {
        self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum()
    }
}

pub fn lagrange_coefficients<F: Field>(xs: &[F], x: F) -> Vec<F> {
    xs.iter()
        .map(|&i| {
            let mut prod = F::ONE;
            for &m in xs.iter() {
                if m != i {
                    prod *= x - m * (i - m).invert().unwrap_or(F::ZERO);
                }
            }
            prod
        })
        .collect()
}

pub fn lagrange_interpolation<F: Field>(x: F, xs: &[F], ys: &[F]) -> F {
    // Lagrange interpolation:
    // L(x) = sum( y_i * l_i(x) )
    // where l_i(x) = prod( (x - x_k)/(x_i - x_k) | k != i)
    // here we always evaluate with x = 0
    let mut sum = F::ZERO;
    let ls = lagrange_coefficients(xs, x);
    for (&li, &yi) in ls.iter().zip(ys) {
        sum += yi * li;
    }
    sum
}

#[cfg(test)]
mod test {
    use crate::algebra::element::Mod11;

    use super::*;

    #[test]
    fn add() {
        let a = Vector::from_array([1u32, 2u32]);
        let b = Vector::from_array([3u32, 4u32]);
        let c: Vector<_> = a + b;
        let c2 = Vector::from_array([4u32, 6u32]);
        assert_eq!(c, c2);
    }

    #[test]
    fn sub() {
        let a = Vector::from_array([Mod11(3), Mod11(4)]);
        let b = Vector::from_array([Mod11(1), Mod11(2)]);
        let c: Vector<_> = a - b;
        let c2 = Vector::from_array([Mod11(2), Mod11(2)]);
        assert_eq!(c, c2);
    }

    #[test]
    fn mul() {
        let a = Vector::from_array([Mod11(3), Mod11(4)]);
        let b = Mod11(2);
        let c: Vector<_> = a * b;
        let c2 = Vector::from_array([Mod11(6), Mod11(8)]);
        assert_eq!(c, c2);
    }
}
