use ff::Field;
use std::ops::{self};

use itertools::{self, Itertools};
use rand::RngCore;

use crate::algebra::math::Vector;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Polynomial<G: Send + Sync>(pub Vector<G>);

// // HACK: We really should not implement Deref/DerefMut, but operator-overloading is annoying.
// // If we can derive Add/Sub/Mul instead that would make me happy.
// impl<G: Send + Sync> Deref for Polynomial<G> {
//     type Target = Vector<G>;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

// impl<G: Send + Sync> DerefMut for Polynomial<G> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }

impl<G: Field> Polynomial<G> {
    /// Evaluate `x` in the polynomial `f`, such you obtain `f(x)`
    ///
    /// * `x`: value to map from
    pub fn eval<F: Field>(&self, x: &F) -> G
    where
        G: ops::Mul<F, Output = G>,
    {
        self.0
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                // evaluate: a * x^i
                a * x.pow([i as u64])
            }) // sum: s + a1 x + a2 x^2 + ...
            .fold(G::ZERO, |sum, x| sum + x)
    }
}

impl<G: Send + Sync> Polynomial<G> {
    pub fn degree(&self) -> usize {
        // a0 + a1x1 is degree(1)
        self.0.len() - 1
    }
}

impl<G: Send + Sync> FromIterator<G> for Polynomial<G> {
    fn from_iter<T: IntoIterator<Item = G>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<F: Field> Polynomial<F> {
    /// Sample a random polynomial
    ///
    /// * `degree`: the degree of the polynomial
    /// * `rng`: random number generator to use
    pub fn random(degree: usize, mut rng: &mut impl RngCore) -> Self {
        (0..=degree).map(|_| F::random(&mut rng)).collect()
    }
}

// impl<F: Copy, G> ops::Mul<&G> for Polynomial<F>
// where F: for<'a> ops::Mul<&'a G>, Box<[G]>: for<'a> FromIterator<<F as ops::Mul<&'a G>>::Output> {
//     type Output = Polynomial<G>;

//     fn mul(self, rhs: &G) -> Self::Output {
//         Polynomial(self.0.iter().map(|&a| a * rhs).collect())
//     }
// }

impl<F: Send + Sync, G: Send + Sync> std::ops::Mul<G> for &Polynomial<F>
where
    for<'a, 'b> &'a F: std::ops::Mul<&'b G, Output = G>,
{
    type Output = Polynomial<G>;

    fn mul(self, rhs: G) -> Self::Output {
        let me = &self.0;
        let res: Vector<G> = me * rhs;
        Polynomial(res)
    }
}

/// Implementation of cartesian product for polyminials
///
/// f * g -> h
///
/// where degree(h) = degree(f) + degree(g)
impl<
        F: Send + Sync + ops::AddAssign + num_traits::Zero + Clone,
        G: Send + Sync + ops::Mul<Output = F> + Copy,
    > Polynomial<G>
{
    pub fn mult(&self, other: &Self) -> Polynomial<F> {
        // degree is length - 1.
        let n = self.0.len() + other.0.len();
        let iter = self
            .0
            .iter()
            .enumerate()
            .cartesian_product(other.0.iter().enumerate())
            .map(|((i, &a), (j, &b))| (i + j, a * b));

        let mut vec = vec![F::zero(); n - 1];
        for (i, a) in iter {
            vec[i] += a;
        }

        Polynomial(Vector::from_vec(vec))
    }
}
