pub mod parsing;

use rand::RngCore;

use crate::{
    algebra::field::Field,
    net::{network::Network, SplitChannel},
    protocols::beaver::{beaver_multiply, BeaverTriple},
    schemes::interactive::{InteractiveShared, InteractiveSharedMany},
};

pub enum Instruction<F> {
    Share(F),
    SymShare(F),
    Recv(usize),
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
        let mut stack = vec![];
        let mut results = vec![];

        for opcode in script.0.iter() {
            self.step(&mut stack, &mut results, opcode).await.unwrap();
        }

        results.pop().unwrap()
    }

    async fn step(
        &mut self,
        stack: &mut Vec<S>,
        results: &mut Vec<F>,
        opcode: &Instruction<F>,
    ) -> Result<(), S::Error> {
        let ctx = &mut self.context;
        let mut coms = &mut self.network;
        let mut rng = &mut self.rng;
        let fueltank = &mut self.fueltank;
        match opcode {
            Instruction::Share(f) => {
                let share = S::share(ctx, *f, &mut rng, &mut coms).await?;
                stack.push(share)
            }
            Instruction::SymShare(f) => {
                let mut shares = S::symmetric_share(ctx, *f, &mut rng, &mut coms).await?;
                stack.append(&mut shares);
            }
            Instruction::Recv(id) => {
                let share = S::receive_share(ctx, &mut coms, *id).await?;
                stack.push(share)
            }
            Instruction::Recombine => {
                let share = stack.pop().unwrap();
                let f = S::recombine(ctx, share, &mut coms).await?;
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
                let z = beaver_multiply(ctx, x, y, triple, &mut coms).await?;
                stack.push(z)
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
        net::{agency::Broadcast, connection::DuplexConnection, network::InMemoryNetwork},
        testing::{self, mock},
        vm::{Engine, Instruction},
    };

    pub fn dumb_engine(
        network: InMemoryNetwork,
    ) -> Engine<DuplexConnection, mock::Share<algebra::element::Element32>, rand::rngs::mock::StepRng>
    {
        let context = mock::Context {
            all_parties: network.size(),
            me: network.index,
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
                let script = crate::vm::Script(vec![
                    SymShare(val), // add two to the stack.
                    Add,
                    Recombine,
                ]);
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
        let a = crate::vm::Script(vec![
            Share(a),  // +1
            Recv(1),   // +1
            Add,       // -1
            Recombine, // -1
        ]);
        let b = Element32::from(57u32);
        let b = crate::vm::Script(vec![Recv(0), Share(b), Add, Recombine]);
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
