/// Pseudo-parsing direct to an array-backed AST which just is a bytecode stack.
use std::{
    array,
    ops::{Add, Mul, Sub},
};

use crate::{
    algebra::math::Vector,
    net::Id,
    vm::{Instruction, Script, Value},
};

pub struct Exp<F> {
    exp: Vec<Instruction<F>>,
}

impl<F> Exp<F> {
    pub fn share(secret: impl Into<F>) -> Self {
        Self {
            exp: vec![Instruction::Share(Value::Single(secret.into()))],
        }
    }

    pub fn share_vec(secret: impl Into<Vector<F>>) -> Self {
        Self {
            exp: vec![Instruction::Share(Value::Vector(secret.into()))],
        }
    }

    pub fn receive_input(id: Id) -> Self {
        Self {
            exp: vec![Instruction::Recv(id)],
        }
    }

    pub fn open(mut self) -> Self {
        self.exp.push(Instruction::Recombine);
        self
    }

    pub fn finalize(self) -> Script<F> {
        Script(self.exp)
    }

    fn empty() -> Self {
        Self { exp: vec![] }
    }

    pub fn share_or_receive<const N: usize>(input: impl Into<F>, me: Id) -> [Self; N] {
        let mut input: Option<F> = Some(input.into());
        array::from_fn(|i| {
            let id = Id(i);
            if id == me {
                let f = input.take().expect("We only do this once.");
                Exp::share(f)
            } else {
                Exp::receive_input(id)
            }
        })
    }
}

impl<F> Add for Exp<F> {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.exp.append(&mut rhs.exp);
        self.exp.push(Instruction::Add);
        self
    }
}

impl<F> Mul for Exp<F> {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        self.exp.append(&mut rhs.exp);
        self.exp.push(Instruction::Mul);
        self
    }
}

impl<F> Mul<F> for Exp<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self.exp.push(Instruction::MulCon(Value::Single(rhs)));
        self
    }
}

impl<F> Sub for Exp<F> {
    type Output = Self;

    fn sub(mut self, mut rhs: Self) -> Self::Output {
        self.exp.append(&mut rhs.exp);
        self.exp.push(Instruction::Sub);
        self
    }
}

#[cfg(test)]
mod test {
    use crate::{
        algebra::{self, element::Element32},
        net::Id,
        testing::{self, Cluster},
        vm::parsing::Exp,
    };

    type F = algebra::element::Element32;
    type Share = testing::mock::Share<F>;

    #[tokio::test]
    async fn addition() {
        let inputs = [3, 7, 13u32];
        let res = Cluster::new(3)
            .with_args(
                (0..3)
                    .map(|id| {
                        let me = Id(id);
                        type E = Exp<Element32>;
                        let [a, b, c] = E::share_or_receive(inputs[id], me);
                        let sum = a + b + c;
                        let res = sum.open();
                        res.finalize()
                    })
                    .collect::<Vec<_>>(),
            )
            .execute_mock()
            .await
            .unwrap();

        assert_eq!(res, vec![23u32.into(), 23u32.into(), 23u32.into()]);
    }

    #[tokio::test]
    async fn multiplication() {
        let inputs = [1, 2, 3u32];
        let res = Cluster::new(3)
            .with_args(
                (0..3)
                    .map(|id| {
                        let me = Id(id);
                        type E = Exp<Element32>;
                        let [a, b, c] = E::share_or_receive(inputs[id], me);
                        let sum = a * b * c;
                        let res = sum.open();
                        res.finalize()
                    })
                    .collect::<Vec<_>>(),
            )
            .execute_mock()
            .await
            .unwrap();

        assert_eq!(res, vec![6u32.into(), 6u32.into(), 6u32.into()]);
    }
}
