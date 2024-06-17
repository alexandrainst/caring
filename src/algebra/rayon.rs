use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Vector<F>(Box<[F]>);

macro_rules! inherent {
    ($trait2:ident, $fun2:ident, $trait1:ident, $fun1:ident) => {
        impl<A: Send + Sync> $trait1<&Vector<A>> for Vector<A>
        where
            A: for<'a> std::ops::$trait1<&'a A>,
        {
            fn $fun1(&mut self, rhs: &Self) {
                self.0
                    .par_iter_mut()
                    .zip(rhs.0.par_iter())
                    .for_each(|(a, b)| $trait1::$fun1(a, b));
            }
        }

        impl<A: Send + Sync> $trait1 for Vector<A>
        where
            for<'a> A: $trait1<&'a A>,
        {
            fn $fun1(&mut self, rhs: Self) {
                $trait1::$fun1(self, &rhs)
            }
        }

        impl<A: Send + Sync> $trait2<&Self> for Vector<A>
        where
            A: for<'a> $trait1<&'a A>,
        {
            type Output = Vector<A>;

            fn $fun2(mut self, rhs: &Self) -> Self::Output {
                $trait1::$fun1(&mut self, rhs);
                self
            }
        }

        impl<A: Send + Sync> $trait2<Self> for Vector<A>
        where
            A: for<'a> $trait1<&'a A>,
        {
            type Output = Vector<A>;

            fn $fun2(mut self, rhs: Self) -> Self::Output {
                $trait1::$fun1(&mut self, rhs);
                self
            }
        }

        impl<A: Send + Sync> $trait2<Vector<A>> for &Vector<A>
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
inherent!(Div, div, DivAssign, div_assign);

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
        self.0.par_iter_mut().for_each(|a| *a *= b);
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
        let internal = self.0.par_iter().map(|a| a * &b).collect();
        Vector(internal)
    }
}

impl<F: ff::Field> Vector<F> {
    pub fn inner_product(&self, other: &Self) -> F {
        self.par_iter().zip(other).map(|(&a, &b)| a * b).sum()
    }
}
