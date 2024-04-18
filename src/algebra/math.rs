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

use ff::Field;
use rayon::prelude::*;

/// Represention of a mathematical vector of type `F`.
///
/// This provides implementations for addition, subtraction and multiplication
/// as in ordinary linear algebra fashion.
///
/// If the rayon feature is enabled the operations will be parallelized.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Vector<F: Send + Sync>(Box<[F]>);

impl<F: Send + Sync> Vector<F> {
    pub const fn from_boxed_slice(slice: Box<[F]>) -> Self {
        Self(slice)
    }

    pub fn from_vec(v: Vec<F>) -> Self {
        Self(v.into_boxed_slice())
    }

    pub fn from_array<const N: usize>(v: [F; N]) -> Self {
        Self(Box::new(v))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<F> {
        self.0.iter()
    }
}

impl<F: Send + Sync> std::ops::Deref for Vector<F> {
    type Target = [F];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Send + Sync> std::ops::Index<usize> for Vector<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<F: Send + Sync> std::ops::IndexMut<usize> for Vector<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<F: Send + Sync> IntoIterator for Vector<F> {
    type Item = F;

    type IntoIter = std::vec::IntoIter<F>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_vec().into_iter()
    }
}

impl<F: Send + Sync> FromIterator<F> for Vector<F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        let boxed = iter.into_iter().collect();
        Self(boxed)
    }
}

impl<'a, F: Send + Sync> IntoParallelIterator for &'a Vector<F> {
    type Item = &'a F;
    type Iter = rayon::slice::Iter<'a, F>;

    fn into_par_iter(self) -> Self::Iter {
        self.0.par_iter()
    }
}

impl<F: Send + Sync> FromParallelIterator<F> for Vector<F> {
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = F>,
    {
        let boxed = par_iter.into_par_iter().collect();
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

// // Inefficient.
// impl<A: Send + Sync + Copy> std::ops::Add for &Vector<A>
// where
//     A: std::ops::Add<Output = A>,
// {
//     type Output = Vector<A>;

//     fn add(self, rhs: Self) -> Self::Output {
//         let internal = if cfg!(feature = "rayon") {
//             self.0
//                 .par_iter()
//                 .zip(rhs.0.par_iter())
//                 .map(|(&a, &b)| a + b)
//                 .collect()
//         } else {
//             self.0
//                 .iter()
//                 .zip(rhs.0.iter())
//                 .map(|(&a, &b)| a + b)
//                 .collect()
//         };
//         Vector(internal)
//     }
// }

impl<A: Send + Sync> std::ops::AddAssign<&Vector<A>> for Vector<A>
where
    A: for<'a> std::ops::AddAssign<&'a A>,
{
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

impl<A: Send + Sync> std::ops::AddAssign for Vector<A>
where
    for<'a> A: std::ops::AddAssign<&'a A>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<A: Send + Sync> std::ops::Add<&Self> for Vector<A>
where
    A: for<'a> std::ops::AddAssign<&'a A>,
{
    type Output = Vector<A>;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<A: Send + Sync> std::ops::Add<Self> for Vector<A>
where
    A: for<'a> std::ops::AddAssign<&'a A>,
{
    type Output = Vector<A>;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += &rhs;
        self
    }
}

// Inefficient
// impl<A: Send + Sync, B: Send + Sync> std::ops::Sub for &Vector<A>
// where
//     for<'a> &'a A: std::ops::Sub<Output = B>,
// {
//     type Output = Vector<B>;

//     fn sub(self, rhs: Self) -> Self::Output {
//         let internal = if cfg!(feature = "rayon") {
//             self.0
//                 .par_iter()
//                 .zip(rhs.0.par_iter())
//                 .map(|(a, b)| a - b)
//                 .collect()
//         } else {
//             self.0
//                 .iter()
//                 .zip(rhs.0.iter())
//                 .map(|(a, b)| a - b)
//                 .collect()
//         };
//         Vector(internal)
//     }
// }

impl<A: Send + Sync> std::ops::SubAssign<&Vector<A>> for Vector<A>
where
    A: for<'a> std::ops::SubAssign<&'a A>,
{
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

impl<A: Send + Sync> std::ops::Sub<&Self> for Vector<A>
where
    A: for<'a> std::ops::SubAssign<&'a A>,
{
    type Output = Vector<A>;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<A: Send + Sync> std::ops::Sub<Self> for Vector<A>
where
    A: for<'a> std::ops::SubAssign<&'a A>,
{
    type Output = Vector<A>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= &rhs;
        self
    }
}

/// Generic implementation of row-wise mult-assign
///
/// Vector<A> =* B
/// if A *= B exists
///
impl<A: Send + Sync, B: Send + Sync> std::ops::MulAssign<&B> for Vector<A>
where
    for<'b> A: std::ops::MulAssign<&'b B>,
{
    fn mul_assign(&mut self, rhs: &B) {
        let b = rhs;
        if cfg!(feature = "rayon") {
            // self.0
            //     .par_iter_mut()
            //     .for_each(|a| *a *= b)
            todo!("error")
        } else {
            self.0.iter_mut().for_each(|a| *a *= b)
        }
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
impl<A: Send + Sync, B: Send + Sync> std::ops::Mul<B> for &Vector<A>
where
    for<'a, 'b> &'a A: std::ops::Mul<&'b B, Output = B>,
{
    type Output = Vector<B>;

    fn mul(self, rhs: B) -> Self::Output {
        let b = rhs;
        let internal = if cfg!(feature = "rayon") {
            self.0.par_iter().map(|a| a * &b).collect()
        } else {
            self.0.iter().map(|a| a * &b).collect()
        };
        Vector(internal)
    }
}

/// Generic implemtentation of row-wise multiplication
///
/// Vector<A> * B -> Vector<A>
/// if A *= B exists
///
impl<A: Send + Sync, B: Send + Sync> std::ops::Mul<B> for Vector<A>
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
        self.par_iter().zip(other).map(|(&a, &b)| a * b).sum()
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
