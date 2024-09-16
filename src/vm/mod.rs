mod ast;
pub mod parsing;

use std::{collections::BTreeMap, error::Error};

use ff::Field;
use itertools::{Either, Itertools};
use rand::RngCore;
use thiserror::Error;

use crate::{
    algebra::{math::Vector, Length},
    net::{agency::Broadcast, connection::TcpConnection, network::Network, Id, SplitChannel},
    protocols::beaver::{beaver_multiply, beaver_multiply_vector, BeaverTriple},
    schemes::interactive::InteractiveSharedMany,
};

#[derive(Debug, Clone)]
pub enum Value<F> {
    Single(F),
    Vector(Vector<F>),
}

impl<F> Value<F> {
    pub fn unwrap_single(self) -> F {
        match self {
            Value::Single(v) => v,
            _ => panic!("Was vector and not a single!"),
        }
    }

    pub fn unwrap_vector(self) -> Vector<F> {
        match self {
            Value::Vector(v) => v,
            _ => panic!("Was single and not a vector!"),
        }
    }

    pub fn map<U>(self, func: impl Fn(F) -> U) -> Value<U> {
        match self {
            Value::Single(a) => Value::Single(func(a)),
            Value::Vector(a) => Value::Vector(a.into_iter().map(func).collect()),
        }
    }

    pub fn to_vec(self) -> Vec<F> {
        match self {
            Value::Single(s) => vec![s],
            Value::Vector(v) => v.into(),
        }
    }
}

impl<F> From<F> for Value<F> {
    fn from(value: F) -> Self {
        Value::Single(value)
    }
}
impl<F> From<Vector<F>> for Value<F> {
    fn from(value: Vector<F>) -> Self {
        Value::Vector(value)
    }
}

impl<F> Value<F> {
    pub fn convert<B>(self) -> Value<B>
    where
        F: Into<B>,
    {
        match self {
            Value::Single(v) => Value::Single(v.into()),
            Value::Vector(v) => Value::Vector(v.into_iter().map(|v| v.into()).collect()),
        }
    }

    pub fn try_convert<B>(self) -> Result<Value<B>, F::Error>
    where
        F: TryInto<B>,
    {
        match self {
            Value::Single(v) => Ok(Value::Single(v.try_into()?)),
            Value::Vector(v) => {
                let bs: Vector<B> = v.into_iter().map(|v| v.try_into()).try_collect()?;
                Ok(Value::Vector(bs))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstRef(u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instruction {
    Share(ConstRef),
    SymShare(ConstRef),
    MulCon(ConstRef),
    Recv(Id),
    RecvVec(Id),
    Recombine,
    Add,
    Mul,
    Sub,
    Sum(usize),
}

#[derive(Clone, Debug)]
enum SharedValue<S: InteractiveSharedMany> {
    Single(S),
    Vector(S::VectorShare),
}

pub struct Script<F> {
    constants: Vec<Value<F>>,
    instructions: Vec<Instruction>,
}

impl<F> Script<F> {
    pub fn mults(&self, parties: usize) -> usize {
        self.mults_and_shared(parties)
            .0
            .into_iter()
            .map(|(size, amount)| size * amount)
            .sum()
    }

    pub fn shared(&self, parties: usize) -> usize {
        self.mults_and_shared(parties).1
    }

    fn get_constant(&self, addr: ConstRef) -> &Value<F> {
        self.constants.get(addr.0 as usize).unwrap()
    }

    fn mults_and_shared(&self, parties: usize) -> (BTreeMap<usize, usize>, usize) {
        let mut mults = BTreeMap::new();
        let mut shared = 0;
        let mut stack = Vec::new();
        for inst in &self.instructions {
            match inst {
                Instruction::Share(addr) => match self.get_constant(*addr) {
                    Value::Single(_) => {
                        stack.push(1usize);
                        shared += 1;
                    }
                    Value::Vector(v) => {
                        stack.push(v.len());
                        shared += v.len();
                    }
                },
                Instruction::SymShare(addr) => match self.get_constant(*addr) {
                    Value::Single(_) => {
                        stack.append(&mut vec![1usize; parties]);
                        shared += parties;
                    }
                    Value::Vector(v) => {
                        shared += v.len() * parties;
                        stack.append(&mut vec![v.len(); parties])
                    }
                },
                Instruction::MulCon(addr) => {
                    stack.pop();
                    match self.get_constant(*addr) {
                        Value::Single(_) => stack.push(1usize),
                        Value::Vector(v) => stack.push(v.len()),
                    }
                }
                Instruction::Recv(_) => {
                    stack.push(1);
                    shared += 1;
                }
                Instruction::RecvVec(_) => {
                    stack.push(0);
                    shared += 1;
                }
                Instruction::Recombine => {
                    stack.pop();
                }
                Instruction::Add | Instruction::Sub => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    if a == 0 {
                        stack.push(b)
                    } else {
                        stack.push(a)
                    }
                }
                Instruction::Mul => {
                    let size = stack.pop().unwrap();
                    mults
                        .entry(size)
                        .and_modify(|amount: &mut usize| *amount += 1);
                }
                Instruction::Sum(n) => {
                    let n = if *n == 0 { parties } else { *n };
                    for _ in 0..(n - 1) {
                        stack.pop();
                    }
                }
            }
        }
        (mults, shared)
    }
}

#[derive(Debug)]
pub struct Engine<C: SplitChannel, S: InteractiveSharedMany, R> {
    network: Network<C>,
    context: S::Context,
    fueltank: FuelTank<S>,
    rng: R,
}

struct Stack<S: InteractiveSharedMany> {
    stack: Vec<SharedValue<S>>,
}

impl<S: InteractiveSharedMany> Stack<S> {
    pub fn new() -> Self {
        Self { stack: vec![] }
    }

    pub fn push(&mut self, val: SharedValue<S>) {
        self.stack.push(val)
    }

    pub fn push_single(&mut self, single: S) {
        self.stack.push(SharedValue::Single(single))
    }

    pub fn push_vector(&mut self, vector: S::VectorShare) {
        self.stack.push(SharedValue::Vector(vector))
    }

    pub fn pop(&mut self) -> SharedValue<S> {
        self.stack.pop().expect("No value found")
    }

    pub fn pop_single(&mut self) -> S {
        match self.stack.pop() {
            Some(SharedValue::Single(single)) => single,
            _ => panic!("no valid value found"),
        }
    }

    pub fn pop_vector(&mut self) -> S::VectorShare {
        match self.stack.pop() {
            Some(SharedValue::Vector(vector)) => vector,
            _ => panic!("no valid value found"),
        }
    }

    pub fn take_singles(&mut self, n: usize) -> impl Iterator<Item = S> + '_ {
        self.stack.drain(0..n).map(|v| match v {
            SharedValue::Single(v) => v,
            _ => panic!(),
        })
    }

    pub fn take_vectors(&mut self, n: usize) -> impl Iterator<Item = S::VectorShare> + '_ {
        self.stack.drain(0..n).map(|v| match v {
            SharedValue::Vector(v) => v,
            _ => panic!(),
        })
    }

    pub fn take(
        &mut self,
        n: usize,
    ) -> Either<impl Iterator<Item = S> + '_, impl Iterator<Item = S::VectorShare> + '_> {
        match self.stack.last() {
            Some(SharedValue::Single(_)) => Either::Left(self.take_singles(n)),
            Some(SharedValue::Vector(_)) => Either::Right(self.take_vectors(n)),
            None => panic!(),
        }
    }
}

struct FuelTank<S> {
    pub mult_triples: Vec<BeaverTriple<S>>,
}

impl<S> FuelTank<S>
where
    S: InteractiveSharedMany,
{
    pub fn single_mult(&mut self) -> Option<BeaverTriple<S>> {
        self.mult_triples.pop()
    }

    pub fn vector_mult(&mut self, size: usize) -> Option<BeaverTriple<S::VectorShare>> {
        if self.mult_triples.len() < size {
            return None;
        }
        let iter = self.mult_triples.drain(..size);
        Some(BeaverTriple::vectorized(iter))
    }
}

impl<S> std::fmt::Debug for FuelTank<S>
where
    S: InteractiveSharedMany,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuelTank")
            .field("mult_triples", &self.mult_triples.len())
            .finish()
    }
}

impl<S> Default for FuelTank<S>
where
    S: InteractiveSharedMany,
{
    fn default() -> Self {
        Self {
            mult_triples: vec![],
        }
    }
}

#[derive(Debug, Error)]
#[error("Error during Execution: {reason}")]
pub struct ExecutionError {
    #[from]
    reason: Box<dyn Error + Send + 'static>,
}

impl ExecutionError {
    pub fn new<E>(error: E) -> Self
    where
        E: Error + Send + 'static,
    {
        Self {
            reason: Box::new(error),
        }
    }
}

impl<C, S, R, F> Engine<C, S, R>
where
    C: SplitChannel,
    S: InteractiveSharedMany<Value = F> + 'static,
    R: RngCore + Send,
    F: Field,
{
    pub fn new(context: S::Context, network: Network<C>, rng: R) -> Self {
        Self {
            network,
            context,
            fueltank: FuelTank::default(),
            rng,
        }
    }

    pub fn id(&self) -> Id {
        self.network.id()
    }

    pub fn peers(&self) -> Vec<Id> {
        self.network.peers()
    }

    pub fn add_fuel(&mut self, fuel: &mut Vec<BeaverTriple<S>>) {
        self.fueltank.mult_triples.append(fuel);
    }

    // TODO: Superscalar execution when awaiting.

    pub async fn execute(&mut self, script: &Script<F>) -> Result<Value<F>, ExecutionError> {
        let mut stack = Stack::new();
        let mut results: Vec<Value<_>> = vec![];
        let constants = &script.constants;

        let n = script.instructions.len();
        let m = constants.len();
        let i = self.network.id();
        tracing::info!("Starting execution of {n} instructions with {m} contants as player {i}");

        for opcode in script.instructions.iter() {
            tracing::trace!("Executing opcode: {opcode:?}");
            self.step(&mut stack, &mut results, constants, opcode)
                .await
                .map_err(ExecutionError::new)?;
        }

        // TODO: Handle missing output.
        // Handle multiple outputs
        Ok(results.pop().unwrap())
    }

    async fn step(
        &mut self,
        stack: &mut Stack<S>,
        results: &mut Vec<Value<F>>,
        constants: &[Value<F>],
        opcode: &Instruction,
    ) -> Result<(), S::Error> {
        let ctx = &mut self.context;
        let mut coms = &mut self.network;
        let mut rng = &mut self.rng;
        let fueltank = &mut self.fueltank;
        let get_constant = |addr: &ConstRef| &constants[addr.0 as usize];

        match opcode {
            Instruction::Share(addr) => {
                let f = get_constant(addr);
                match f {
                    Value::Single(f) => {
                        let share = S::share(ctx, *f, &mut rng, &mut coms).await?;
                        stack.push_single(share)
                    }
                    Value::Vector(fs) => {
                        let share = S::share_many(ctx, fs, &mut rng, &mut coms).await?;
                        stack.push_vector(share)
                    }
                }
            }
            Instruction::SymShare(addr) => {
                let f = get_constant(addr);
                match f {
                    Value::Single(f) => {
                        let shares = S::symmetric_share(ctx, *f, &mut rng, &mut coms).await?;
                        let shares = shares.into_iter().map(|s| SharedValue::Single(s));
                        stack.stack.extend(shares) // dirty hack
                    }
                    Value::Vector(f) => {
                        let shares = S::symmetric_share_many(ctx, f, &mut rng, &mut coms).await?;
                        let shares = shares.into_iter().map(|s| SharedValue::Vector(s));
                        stack.stack.extend(shares) // dirty hack
                    }
                }
            }
            Instruction::Recv(id) => {
                let share = S::receive_share(ctx, &mut coms, *id).await?;
                stack.push_single(share)
            }
            Instruction::RecvVec(id) => {
                let share = S::receive_share_many(ctx, &mut coms, *id).await?;
                stack.push_vector(share)
            }
            Instruction::Recombine => {
                match stack.pop() {
                    SharedValue::Single(share) => {
                        let f = S::recombine(ctx, share, &mut coms).await?;
                        results.push(Value::Single(f));
                    }
                    SharedValue::Vector(share) => {
                        let f = S::recombine_many(ctx, share, &mut coms).await?;
                        results.push(Value::Vector(f));
                    }
                };
            }
            Instruction::Add => {
                let a = stack.pop();
                let b = stack.pop();
                match (a, b) {
                    (SharedValue::Single(a), SharedValue::Single(b)) => stack.push_single(a + b),
                    (SharedValue::Vector(a), SharedValue::Vector(b)) => stack.push_vector(a + &b),
                    _ => panic!("Unsupported operation"),
                }
            }
            Instruction::Sub => {
                let a = stack.pop();
                let b = stack.pop();
                match (a, b) {
                    (SharedValue::Single(a), SharedValue::Single(b)) => stack.push_single(a - b),
                    (SharedValue::Vector(a), SharedValue::Vector(b)) => stack.push_vector(a - &b),
                    _ => panic!("Unsupported operation"),
                }
            }
            Instruction::MulCon(addr) => {
                let con = get_constant(addr);
                match (stack.pop(), con) {
                    (SharedValue::Single(a), Value::Single(con)) => stack.push_single(a * *con),
                    (SharedValue::Vector(_a), Value::Vector(_con)) => {
                        todo!("vector mult")
                        //stack.push_vector(a * &constant)
                    }
                    (SharedValue::Single(_), Value::Vector(_)) => todo!(),
                    (SharedValue::Vector(_), Value::Single(_)) => todo!(),
                }
            }
            Instruction::Mul => {
                let x = stack.pop();
                let y = stack.pop();
                match (x, y) {
                    (SharedValue::Single(x), SharedValue::Single(y)) => {
                        let triple = fueltank.single_mult().unwrap();
                        let z = beaver_multiply(ctx, x, y, triple, &mut coms).await?;
                        stack.push_single(z)
                    }
                    (SharedValue::Vector(x), SharedValue::Vector(y)) => {
                        let size = x.len();
                        let triple = fueltank.vector_mult(size).unwrap();
                        let z: S::VectorShare =
                            beaver_multiply_vector::<F, S>(ctx, &x, &y, triple, &mut coms).await?;
                        stack.push_vector(z);
                    }
                    (SharedValue::Vector(_), SharedValue::Single(_y)) => {
                        todo!()
                    }
                    (SharedValue::Single(_), SharedValue::Vector(_)) => todo!(),
                };
            }
            Instruction::Sum(size) => {
                // Zero is a sentinal value that represents the party size.
                let size = if *size == 0 {
                    self.network.size()
                } else {
                    *size
                };
                let res = match stack.take(size) {
                    Either::Left(iter) => {
                        let res = iter.reduce(|s, acc| acc + s).unwrap();
                        SharedValue::Single(res)
                    }
                    Either::Right(iter) => {
                        let res = iter.reduce(|s, acc| acc + &s).unwrap();
                        SharedValue::Vector(res)
                    }
                };
                stack.push(res)
            }
        }
        Ok(())
    }

    pub async fn raw<Func, Out>(&mut self, routine: Func) -> Out
    where
        Func: async Fn(&mut Network<C>, &mut S::Context, &mut R) -> Out,
    {
        // TODO: Add other resources.
        routine(&mut self.network, &mut self.context, &mut self.rng).await
    }
}

impl<S: InteractiveSharedMany, R> Engine<TcpConnection, S, R> {
    pub async fn shutdown(self) -> Result<(), std::io::Error> {
        self.network.shutdown().await
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use crate::{
        algebra::{self, element::Element32},
        net::{agency::Broadcast, connection::DuplexConnection, network::InMemoryNetwork, Id},
        testing::{self, mock},
        vm::{ConstRef, Engine, Instruction, Value},
    };

    pub fn dumb_engine(
        network: InMemoryNetwork,
    ) -> Engine<DuplexConnection, mock::Share<algebra::element::Element32>, rand::rngs::mock::StepRng>
    {
        let context = mock::Context {
            all_parties: network.size(),
            me: network.id(),
        };

        let rng = StepRng::new(42, 7);

        Engine {
            network,
            context,
            fueltank: super::FuelTank::default(),
            rng,
        }
    }

    #[tokio::test]
    async fn add_two() {
        let results = testing::Cluster::new(2)
            .with_args([47, 52u32])
            .run_with_args(|net, val| async move {
                let mut engine = dumb_engine(net);
                use Instruction::{Add, Recombine, SymShare};
                let val = Element32::from(val);
                let script = crate::vm::Script {
                    constants: vec![Value::Single(val)],
                    instructions: vec![
                        SymShare(ConstRef(0)), // add two to the stack.
                        Add,
                        Recombine,
                    ],
                };
                let res: u32 = engine
                    .execute(&script)
                    .await
                    .unwrap()
                    .unwrap_single()
                    .into();
                engine.network.shutdown().await.unwrap();
                res
            })
            .await
            .unwrap();

        assert_eq!(results, vec![99, 99]);
    }

    #[tokio::test]
    async fn add_two_again() {
        use Instruction::{Add, Recombine, Recv, Share};
        let a = Element32::from(42u32);
        let a = crate::vm::Script {
            constants: vec![Value::Single(a)],
            instructions: vec![
                Share(ConstRef(0)), // +1
                Recv(Id(1)),        // +1
                Add,                // -1
                Recombine,          // -1
            ],
        };
        let b = Element32::from(57u32);
        let b = crate::vm::Script {
            constants: vec![Value::Single(b)],
            instructions: vec![
                Recv(Id(0)),        // +1
                Share(ConstRef(0)), // +1
                Add,                // -1
                Recombine,          // -1
            ],
        };
        let results = testing::Cluster::new(2)
            .with_args([a, b])
            .run_with_args(|net, script| async move {
                let mut engine = dumb_engine(net);
                let res: u32 = engine
                    .execute(&script)
                    .await
                    .unwrap()
                    .unwrap_single()
                    .into();
                engine.network.shutdown().await.unwrap();
                res
            })
            .await
            .unwrap();

        assert_eq!(results, vec![99, 99]);
    }
}
