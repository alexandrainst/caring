use std::ops;
use group::Group;


#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Polynomial<G : Group>(pub Box<[G]>);

impl<G: Group> Polynomial<G> {
    pub fn add_self(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(a, &b)| *a += b)
    }

    pub fn sub_self(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(a, &b)| *a -= b)
    }
}

impl<G: Group> ops::Add for Polynomial<G> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res: Box<[_]> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Self(res)
    }
}

impl<G: Group> ops::Add for &Polynomial<G> {
    type Output = Polynomial<G>;

    fn add(self, rhs: Self) -> Self::Output {
        let res: Box<[_]> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Polynomial(res)
    }
}

impl<G: Group> ops::AddAssign for Polynomial<G> {

    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(a, &b)| *a += b)
    }
}

impl<G: Group> ops::AddAssign for &mut Polynomial<G> {

    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(a, &b)| *a += b)
    }
}

impl<G: Group> ops::Sub for Polynomial<G> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let res: Box<[_]> = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Self(res)
    }
}

impl<F: Group> std::iter::Sum for Polynomial<F> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut acc = iter.next().expect("Can't sum zero polynomials");
        for poly in iter {
            acc += poly;
        }
        acc
    }
}

