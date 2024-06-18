pub mod parsing;

use rand::RngCore;

use crate::{
    algebra::{field::Field, math::Vector},
    net::{network::Network, SplitChannel},
    protocols::beaver::{beaver_multiply, BeaverTriple},
    schemes::interactive::{InteractiveShared, InteractiveSharedMany},
};

enum Instruction<F> {
    Share(F),
    Recombine,
    Add,
    MulCon(F),
    Mul,
    Sub,
}

// TODO: Handle vectorized shares

enum Value<S: InteractiveSharedMany> {
    Single(S),
    Vector(S::VectorShare),
}

pub struct Script<F>(Vec<Instruction<F>>);

pub struct Engine<C: SplitChannel, S: InteractiveShared, R> {
    network: Network<C>,
    context: S::Context,
    fueltank: Vec<BeaverTriple<S>>,
    rng: R,
}

impl<C, S, R, F> Engine<C, S, R>
where
    C: SplitChannel,
    S: InteractiveShared<Value = F> + std::ops::Mul<F, Output = S>,
    R: RngCore + Send,
    F: Field,
{
    // TODO: Superscalar execution when awaiting.

    pub async fn execute(&mut self, script: &Script<F>) -> F {
        let ctx = &mut self.context;
        let mut coms = &mut self.network;
        let mut rng = &mut self.rng;
        let fueltank = &mut self.fueltank;

        let mut stack = vec![];
        let mut results = vec![];

        for opcode in script.0.iter() {
            match opcode {
                Instruction::Share(f) => {
                    let share = S::share(ctx, *f, &mut rng, &mut coms).await.unwrap();
                    stack.push(share)
                }
                Instruction::Recombine => {
                    let share = stack.pop().unwrap();
                    let f = S::recombine(ctx, share, &mut coms).await.unwrap();
                    results.push(f);
                }
                Instruction::Add => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(a + b);
                }
                Instruction::Sub => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(a - b);
                }
                Instruction::MulCon(f) => {
                    let a = stack.pop().unwrap();
                    stack.push(a * *f);
                }
                Instruction::Mul => {
                    let x = stack.pop().unwrap();
                    let y = stack.pop().unwrap();
                    let triple = fueltank.pop().unwrap();
                    let z = beaver_multiply(ctx, x, y, triple, &mut coms).await.unwrap();
                    stack.push(z)
                }
            }
        }

        results.pop().unwrap()
    }
}
