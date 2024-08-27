use itertools::Itertools;
/// Pseudo-parsing direct to an array-backed AST which just is a bytecode stack.
use std::{
    array,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{
    algebra::math::Vector,
    net::Id,
    vm::{Const, Instruction, Script, Value},
};

/// An expression stack
#[derive(Clone, Debug)]
pub struct Exp<F> {
    constants: Vec<Value<F>>,
    instructions: Vec<Instruction>,
}

// A dynamicly sized list of expressions.
#[derive(Clone, Debug)]
pub struct ExpList<F> {
    constant: Value<F>,
}

// An opened expression (last step)
#[derive(Clone, Debug)]
pub struct Opened<F>(Exp<F>);

impl<F> Exp<F> {
    pub fn from_parts(constants: Vec<Value<F>>, instructions: Vec<Instruction>) -> Self {
        Self {
            constants,
            instructions,
        }
    }

    fn empty() -> Self {
        Self {
            constants: vec![],
            instructions: vec![],
        }
    }

    fn append_constant(&mut self, value: impl Into<Value<F>>) -> Const {
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

    // This is slighty cursed.
    pub fn symmetric_share(secret: impl Into<F>) -> ExpList<F> {
        ExpList {
            constant: Value::Single(secret.into()),
        }
    }

    // This is slighty cursed.
    pub fn symmetric_share_vec(secret: impl Into<Vector<F>>) -> ExpList<F> {
        ExpList {
            constant: Value::Vector(secret.into()),
        }
    }

    /// Secret share into a field value
    ///
    /// * `secret`: value to secret share
    pub fn share(secret: impl Into<F>) -> Self {
        Self::constant_op(secret.into(), Instruction::Share(Const(0)))
    }

    /// Secret share a vector
    ///
    /// * `secret`: vector to secret share
    pub fn share_vec(secret: impl Into<Vector<F>>) -> Self {
        Self::constant_op(secret.into(), Instruction::Share(Const(0)))
    }

    /// Receive are share from a given party `id`
    ///
    /// * `id`: Id of party to receive from
    pub fn receive_input(id: Id) -> Self {
        Self {
            constants: vec![],
            instructions: vec![Instruction::Recv(id)],
        }
    }

    /// Share and receive based on your given Id
    ///
    /// * `input`: Your input to secret-share
    /// * `me`: Your Id
    pub fn share_and_receive<const N: usize>(input: impl Into<F>, me: Id) -> [Self; N] {
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

    /// Share and receive based on your given Id
    ///
    /// * `input`: Your input to secret-share
    /// * `me`: Your Id
    pub fn share_and_receive_n(input: impl Into<F>, me: Id, n: usize) -> Vec<Self> {
        let mut input: Option<F> = Some(input.into());
        (0..n)
            .map(|i| {
                let id = Id(i);
                if id == me {
                    let f = input.take().expect("We only do this once.");
                    Self::share(f)
                } else {
                    Self::receive_input(id)
                }
            })
            .collect()
    }

    /// Open the secret value
    pub fn open(mut self) -> Opened<F> {
        self.instructions.push(Instruction::Recombine);
        Opened(self)
    }

    /// Helper function to manage changes in addreses when combining expressions
    fn append(&mut self, mut other: Self) {
        let n = self.constants.len() as u16;
        self.constants.append(&mut other.constants);
        for opcode in other.instructions.iter_mut() {
            match opcode {
                Instruction::Share(addr) => addr.0 += n,
                Instruction::SymShare(addr) => addr.0 += n,
                Instruction::MulCon(addr) => addr.0 += n,
                Instruction::Recv(_) // Explicit do nothing here.
                | Instruction::RecvVec(_)
                | Instruction::Recombine
                | Instruction::Add
                | Instruction::Sum(_)
                | Instruction::Mul
                | Instruction::Sub => (),
            }
        }
        self.instructions.append(&mut other.instructions);
    }
}

impl<T> Opened<T> {
    pub fn finalize<F>(self) -> Script<F>
    where
        T: Into<F>,
    {
        let Exp {
            constants,
            instructions,
        } = self.0;
        let constants = constants.into_iter().map(|v| v.convert()).collect();
        Script {
            constants,
            instructions,
        }
    }

    pub fn try_finalize<F>(self) -> Result<Script<F>, T::Error>
    where
        T: TryInto<F>,
    {
        let Exp {
            constants,
            instructions,
        } = self.0;
        let constants = constants
            .into_iter()
            .map(|v| v.try_convert())
            .try_collect()?;
        Ok(Script {
            constants,
            instructions,
        })
    }
}

impl<F> ExpList<F> {
    /// Promise that the explist is `size` long
    ///
    /// This will then assume that there a `size` on the stack when executing.
    pub fn concrete(self, own: usize, size: usize) -> Vec<Exp<F>> {
        let mut me = Some(Exp {
            constants: vec![self.constant],
            instructions: vec![Instruction::SymShare(Const(0))],
        });
        (0..size)
            .map(|id| {
                if id == own {
                    me.take().unwrap()
                } else {
                    Exp::empty()
                }
            })
            .collect()
    }

    pub fn sum(self) -> Exp<F> {
        use Instruction as I;
        Exp {
            constants: vec![self.constant],
            instructions: vec![I::SymShare(Const(0)), I::Sum(0)],
        }
    }
}

impl<F> AddAssign for Exp<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.append(rhs);
        self.instructions.push(Instruction::Add);
    }
}

impl<F> SubAssign for Exp<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.append(rhs);
        self.instructions.push(Instruction::Sub);
    }
}

impl<F> MulAssign for Exp<F> {
    fn mul_assign(&mut self, rhs: Self) {
        self.append(rhs);
        self.instructions.push(Instruction::Mul);
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
        let addr = self.append_constant(rhs);
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

impl<F> Sum for Exp<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let (mut exp, size) = iter.fold((Exp::empty(), 0usize), |(mut acc, count), exp| {
            acc.append(exp);
            (acc, count + 1)
        });
        exp.instructions.push(Instruction::Sum(size));
        exp
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
                        let [a, b, c] = E::share_and_receive(inputs[id], me);
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
                        let [a, b, c] = E::share_and_receive(inputs[id], me);
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

    #[tokio::test]
    async fn mult_add() {
        let inputs = [1, 2, 3u32];
        let res = Cluster::new(3)
            .with_args(
                (0..3)
                    .map(|id| {
                        let me = Id(id);
                        type E = Exp<Element32>;
                        let [a, b, c] = E::share_and_receive(inputs[id], me);
                        let sum = a + b * c; // no need to implement precedence!
                        let res = sum.open();
                        res.finalize()
                    })
                    .collect::<Vec<_>>(),
            )
            .execute_mock()
            .await
            .unwrap();

        // 1 + 2 * 3 = 7
        assert_eq!(res, vec![7u32.into(), 7u32.into(), 7u32.into()]);
    }

    #[tokio::test]
    async fn explist() {
        let inputs = [1, 2, 3u32];
        let res = Cluster::new(3)
            .with_args(
                (0..3)
                    .map(|id| {
                        type E = Exp<Element32>;
                        let exp = E::symmetric_share(inputs[id]);
                        let [a, b, c]: [E; 3] = exp.concrete(id, 3).try_into().unwrap();
                        let sum = a + b * c; // no need to implement precedence!
                        let res = sum.open();
                        res.finalize()
                    })
                    .collect::<Vec<_>>(),
            )
            .execute_mock()
            .await
            .unwrap();

        // 1 + 2 * 3 = 7
        assert_eq!(res, vec![7u32.into(), 7u32.into(), 7u32.into()]);
    }

    #[tokio::test]
    async fn sum() {
        let inputs = [1, 2, 3u32];
        let res = Cluster::new(3)
            .with_args(
                (0..3)
                    .map(|id| {
                        let me = Id(id);
                        type E = Exp<Element32>;
                        let [a, b, c] = E::share_and_receive(inputs[id], me);
                        let sum: E = [a, b, c].into_iter().sum();
                        let res = sum.open();
                        res.finalize()
                    })
                    .collect::<Vec<_>>(),
            )
            .execute_mock()
            .await
            .unwrap();

        // 1 + 2 + 3 = 6
        assert_eq!(res, vec![6u32.into(), 6u32.into(), 6u32.into()]);
    }

    #[tokio::test]
    async fn sum_explist() {
        let inputs = [1, 2, 3u32];
        let res = Cluster::new(3)
            .with_args(
                (0..3)
                    .map(|id| {
                        type E = Exp<Element32>;
                        let exp = E::symmetric_share(inputs[id]);
                        let sum = exp.sum();
                        let res = sum.open();
                        res.finalize()
                    })
                    .collect::<Vec<_>>(),
            )
            .execute_mock()
            .await
            .unwrap();

        // 1 + 2 + 3 = 6
        assert_eq!(res, vec![6u32.into(), 6u32.into(), 6u32.into()]);
    }
}
