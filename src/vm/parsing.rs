/// Pseudo-parsing direct to an array-backed AST which just is a bytecode stack.
use std::{
    array,
    ops::{Add, Mul, Sub},
};

use crate::{
    algebra::math::Vector,
    net::Id,
    vm::{Const, Instruction, Script, Value},
};

#[derive(Debug)]
pub struct Exp<F> {
    constants: Vec<Value<F>>,
    instructions: Vec<Instruction>,
}

impl<F> Exp<F> {
    fn empty() -> Self {
        Self {
            constants: vec![],
            instructions: vec![],
        }
    }
    fn add_constant(&mut self, value: impl Into<Value<F>>) -> Const {
        self.constants.push(value.into());
        Const(self.constants.len() as u16 - 1)
    }

    fn constant_op(value: impl Into<Value<F>>, opcode: Instruction) -> Self {
        let constants = vec![value.into()];
        Self {
            instructions: vec![opcode],
            constants,
        }
    }

    pub fn share(secret: impl Into<F>) -> Self {
        Self::constant_op(secret.into(), Instruction::Share(Const(0)))
    }

    pub fn share_vec(secret: impl Into<Vector<F>>) -> Self {
        Self::constant_op(secret.into(), Instruction::Share(Const(0)))
    }

    pub fn receive_input(id: Id) -> Self {
        Self {
            constants: vec![],
            instructions: vec![Instruction::Recv(id)],
        }
    }

    pub fn share_or_receive<const N: usize>(input: impl Into<F>, me: Id) -> [Self; N] {
        let mut input: Option<F> = Some(input.into());
        array::from_fn(|i| {
            let id = Id(i);
            if id == me {
                let f = input.take().expect("We only do this once.");
                Self::share(f)
            } else {
                Self::receive_input(id)
            }
        })
    }
    pub fn open(mut self) -> Self {
        self.instructions.push(Instruction::Recombine);
        self
    }

    pub fn finalize(self) -> Script<F> {
        let Self {
            constants,
            instructions,
        } = self;
        Script {
            constants,
            instructions,
        }
    }

    fn append(&mut self, mut other: Self) {
        let n = self.constants.len() as u16;
        self.constants.append(&mut other.constants);
        for opcode in other.instructions.iter_mut() {
            match opcode {
                Instruction::Share(addr) => addr.0 += n,
                Instruction::SymShare(addr) => addr.0 += n,
                Instruction::Load(addr) => addr.0 += n,
                Instruction::MulCon(addr) => addr.0 += n,
                _ => (),
            }
        }
        self.instructions.append(&mut other.instructions);
    }
}

impl<F> Add for Exp<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.append(rhs);
        self.instructions.push(Instruction::Add);
        self
    }
}

impl<F> Mul for Exp<F> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.append(rhs);
        self.instructions.push(Instruction::Mul);
        self
    }
}

impl<F> Mul<F> for Exp<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        let addr = self.add_constant(rhs);
        self.instructions.push(Instruction::MulCon(addr));
        self
    }
}

impl<F> Sub for Exp<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.append(rhs);
        self.instructions.push(Instruction::Sub);
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
                        dbg!(&res);
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
