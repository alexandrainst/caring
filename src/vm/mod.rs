pub mod parsing;

use ff::Field;
use rand::RngCore;

use crate::{
    algebra::math::Vector,
    net::{network::Network, Id, SplitChannel},
    protocols::beaver::{beaver_multiply, BeaverTriple},
    schemes::{
        interactive::{InteractiveShared, InteractiveSharedMany},
        shamir,
    },
};

#[derive(Debug, Clone)]
pub enum Value<F> {
    Single(F),
    Vector(Vector<F>),
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

#[derive(Debug, Clone, Copy)]
pub struct Const(u16);

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    Share(Const),
    SymShare(Const),
    Recv(Id),
    RecvVec(Id),
    Recombine,
    Add,
    MulCon(Const),
    Mul,
    Sub,
    Load(Const),
}

// TODO: Handle vectorized shares

enum SharedValue<S: InteractiveSharedMany> {
    Single(S),
    Vector(S::VectorShare),
}

pub struct Script<F> {
    constants: Vec<Value<F>>,
    instructions: Vec<Instruction>,
}

pub struct Engine<C: SplitChannel, S: InteractiveShared, R> {
    network: Network<C>,
    context: S::Context,
    fueltank: Vec<BeaverTriple<S>>,
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
        self.stack.pop().unwrap()
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
            fueltank: vec![],
            rng,
        }
    }

    pub fn add_fuel(&mut self, fuel: &mut Vec<BeaverTriple<S>>) {
        self.fueltank.append(fuel);
    }

    // TODO: Superscalar execution when awaiting.

    pub async fn execute(&mut self, script: &Script<F>) -> F {
        let mut stack = Stack::new();
        let mut results = vec![];
        let constants = &script.constants;

        for opcode in script.instructions.iter() {
            self.step(&mut stack, &mut results, constants, opcode)
                .await
                .unwrap();
        }

        results.pop().unwrap()
    }

    async fn step(
        &mut self,
        stack: &mut Stack<S>,
        results: &mut Vec<F>,
        constants: &[Value<F>],
        opcode: &Instruction,
    ) -> Result<(), S::Error> {
        let ctx = &mut self.context;
        let mut coms = &mut self.network;
        let mut rng = &mut self.rng;
        let fueltank = &mut self.fueltank;
        match opcode {
            Instruction::Share(addr) => {
                let f = &constants[addr.0 as usize];
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
                let f = &constants[addr.0 as usize];
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
                let share = stack.pop_single();
                let f = S::recombine(ctx, share, &mut coms).await?;
                results.push(f);
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
                let constant = &constants[addr.0 as usize];
                match (stack.pop(), constant) {
                    (SharedValue::Single(a), Value::Single(constant)) => {
                        stack.push_single(a * *constant)
                    }
                    (SharedValue::Vector(a), Value::Vector(constant)) => {
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
                        let triple = fueltank.pop().unwrap();
                        let z = beaver_multiply(ctx, x, y, triple, &mut coms).await?;
                        stack.push_single(z)
                    }
                    (SharedValue::Vector(x), SharedValue::Vector(y)) => {
                        todo!()
                    }
                    (SharedValue::Vector(_), SharedValue::Single(y)) => {
                        todo!()
                    }
                    (SharedValue::Single(_), SharedValue::Vector(_)) => todo!(),
                };
            }
            _ => {
                todo!()
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use crate::{
        algebra::{self, element::Element32},
        net::{agency::Broadcast, connection::DuplexConnection, network::InMemoryNetwork, Id},
        testing::{self, mock},
        vm::{Const, Engine, Instruction, Value},
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
            fueltank: vec![],
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
                        SymShare(Const(0)), // add two to the stack.
                        Add,
                        Recombine,
                    ],
                };
                let res: u32 = engine.execute(&script).await.into();
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
                Share(Const(0)), // +1
                Recv(Id(1)),     // +1
                Add,             // -1
                Recombine,       // -1
            ],
        };
        let b = Element32::from(57u32);
        let b = crate::vm::Script {
            constants: vec![Value::Single(b)],
            instructions: vec![
                Recv(Id(0)),     // +1
                Share(Const(0)), // +1
                Add,             // -1
                Recombine,       // -1
            ],
        };
        let results = testing::Cluster::new(2)
            .with_args([a, b])
            .run_with_args(|net, script| async move {
                let mut engine = dumb_engine(net);
                let res: u32 = engine.execute(&script).await.into();
                engine.network.shutdown().await.unwrap();
                res
            })
            .await
            .unwrap();

        assert_eq!(results, vec![99, 99]);
    }
}
