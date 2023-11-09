use ff::Field;
use std::ops::{self, AddAssign};

use itertools::{self, Itertools};
use rand::RngCore;
// NOTE: Consider basing a lot of these generic vectorized data structeres on a common
// one as to limit the mental and code overhead of duplicated implementations of
// addition, multiplication, etc.
// Maybe this task should be delegated out to a crate itself?
// --
// Currently, the common functionaly is a dynamically allocated but constant sized vector,
// which supports addition with itself and multiplication with another type in which it maps over
// to. There is also the randomness, and iterating over values.

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Polynomial<G>(pub Box<[G]>);

impl<G: ops::AddAssign + Clone> Polynomial<G> {
    fn add_self(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(a, b)| *a += b.clone())
    }
}

impl<G: ops::SubAssign + Clone> Polynomial<G> {
    fn sub_self(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(a, b)| *a -= b.clone())
    }
}

impl<G: Field> Polynomial<G> {
    /// Evaluate `x` in the polynomial `f`, such you obtain `f(x)`
    ///
    /// * `x`: value to map from
    pub fn eval<F: Field>(&self, x: F) -> G
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

impl<G> FromIterator<G> for Polynomial<G> {
    fn from_iter<T: IntoIterator<Item = G>>(iter: T) -> Self {
        let poly: Box<[_]> = iter.into_iter().collect();
        Polynomial(poly)
    }
}

impl<F: Field> Polynomial<F> {
    /// Sample a random polynomial
    ///
    /// * `degree`: the degree of the polynomial
    /// * `rng`: random number generator to use
    pub fn random(degree: usize, mut rng: &mut impl RngCore) -> Self {
        (0..degree).map(|_| F::random(&mut rng)).collect()
    }
}

// impl<F: Copy, G> ops::Mul<&G> for Polynomial<F>
// where F: for<'a> ops::Mul<&'a G>, Box<[G]>: for<'a> FromIterator<<F as ops::Mul<&'a G>>::Output> {
//     type Output = Polynomial<G>;

//     fn mul(self, rhs: &G) -> Self::Output {
//         Polynomial(self.0.iter().map(|&a| a * rhs).collect())
//     }
// }

impl<F: Copy, G: Clone> ops::Mul<G> for &Polynomial<F>
where
    F: ops::Mul<G, Output = G>,
    Box<[G]>: FromIterator<<F as ops::Mul<G>>::Output>,
{
    type Output = Polynomial<G>;

    fn mul(self, rhs: G) -> Self::Output {
        Polynomial(self.0.iter().map(|&a| -> G { a * rhs.clone() }).collect())
    }
}

impl<'a, 'b, F: Copy, G: Clone> ops::MulAssign<&'a G> for &'b mut Polynomial<F>
where
    &'b mut F: ops::MulAssign<&'a G>,
{
    fn mul_assign(&mut self, rhs: &'a G) {
        for mut a in self.0.iter_mut() {
            a *= rhs;
        }
    }
}

impl<F: AddAssign + Clone> std::iter::Sum for Polynomial<F> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        // This is sort of a hack, sum() should work on empty iterators.
        // This requires some sort of default/'zero polynomial'
        // While we could construct a 'zero' polynomial instead, where each
        // coefficient is zero, this still leaves the question of the degree.
        // This begs to differ if we want an empty polynomial (degree -1) or a constant polynomial
        // (degree 0) with the single element zero.
        //
        // However neither these are particular usefull, so in most cases
        // the sum would be an mistake, since you would not be able to perform very
        // meaning operations between these and other polynomials.
        let mut acc = iter.next().expect("Can't sum zero polynomials");
        for poly in iter {
            acc += poly;
        }
        acc
    }
}

impl<F: ops::AddAssign + num_traits::Zero + Clone, G: ops::Mul<Output = F> + Copy> Polynomial<G> {
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

        Polynomial(vec.into())
    }
}

impl<G: ops::AddAssign + Clone> ops::AddAssign for Polynomial<G> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_self(&rhs)
    }
}

impl<G: ops::AddAssign + Clone> ops::AddAssign<&Polynomial<G>> for Polynomial<G> {
    fn add_assign(&mut self, rhs: &Polynomial<G>) {
        self.add_self(rhs)
    }
}

impl<G: ops::AddAssign + Clone> ops::AddAssign<&Polynomial<G>> for &mut Polynomial<G> {
    fn add_assign(&mut self, rhs: &Polynomial<G>) {
        self.add_self(rhs)
    }
}

impl<G: ops::AddAssign + Clone> ops::Add for Polynomial<G> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new = self;
        new += rhs;
        new
    }
}

impl<'a, 'b, G: ops::AddAssign + Clone> ops::Add<&'a Polynomial<G>> for &'b Polynomial<G> {
    type Output = Polynomial<G>;

    fn add(self, rhs: &Polynomial<G>) -> Self::Output {
        let mut new = self.clone();
        new += rhs;
        new
    }
}

impl<G: ops::AddAssign + Clone> ops::Add<Polynomial<G>> for &Polynomial<G> {
    type Output = Polynomial<G>;

    fn add(self, rhs: Polynomial<G>) -> Self::Output {
        let mut new = self.clone();
        new += rhs;
        new
    }
}

impl<G: ops::SubAssign + Clone> ops::SubAssign for Polynomial<G> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_self(&rhs)
    }
}

impl<G: ops::SubAssign + Clone> ops::SubAssign<&Polynomial<G>> for Polynomial<G> {
    fn sub_assign(&mut self, rhs: &Polynomial<G>) {
        self.sub_self(rhs)
    }
}

impl<G: ops::SubAssign + Clone> ops::SubAssign<&Polynomial<G>> for &mut Polynomial<G> {
    fn sub_assign(&mut self, rhs: &Polynomial<G>) {
        self.sub_self(rhs)
    }
}

impl<G: ops::SubAssign + Clone> ops::Sub for Polynomial<G> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();
        new -= rhs;
        new
    }
}

impl<G: ops::SubAssign + Clone> ops::Sub<&Polynomial<G>> for Polynomial<G> {
    type Output = Self;

    fn sub(self, rhs: &Polynomial<G>) -> Self::Output {
        let mut new = self.clone();
        new -= rhs;
        new
    }
}

impl<'a, 'b, G: ops::SubAssign + Clone> ops::Sub<&'a Polynomial<G>> for &'b Polynomial<G> {
    type Output = Polynomial<G>;

    fn sub(self, rhs: &Polynomial<G>) -> Self::Output {
        let mut new = self.clone();
        new -= rhs;
        new
    }
}
