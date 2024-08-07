/// Pseudo-parsing direct to an array-backed AST which just is a bytecode stack.
use std::{
    array,
    ops::{Add, Mul, Sub},
};

use itertools::Itertools;

use crate::{
    net::Id,
    vm::{Instruction, Script},
};

struct Exp<F> {
    exp: Vec<Instruction<F>>,
}

impl<F> Exp<F> {
    pub fn share(secret: impl Into<F>) -> Self {
        Self {
            exp: vec![Instruction::Share(secret.into())],
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

    pub fn finalize(mut self) -> Script<F> {
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
                let f = input.take().unwrap();
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
        self.exp.push(Instruction::MulCon(rhs));
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
    use rand::SeedableRng;

    use crate::{
        algebra,
        net::{network::InMemoryNetwork, Id},
        protocols::beaver,
        testing::{self, Cluster},
        vm::{parsing::Exp, Engine},
    };

    type F = algebra::element::Element32;
    type Share = testing::mock::Share<F>;

    #[tokio::test]
    async fn addition() {
        async fn program(net: InMemoryNetwork, input: impl Into<F>) -> F {
            let me = net.id();

            let script = {
                let [a, b, c] = Exp::share_or_receive(input, me);

                let sum = a + b + c;
                let res = sum.open();
                res.finalize()
            };
            let ctx = Share::new_context(me, 3);
            let rng = rand::rngs::StdRng::from_entropy();

            let mut engine = Engine::<_, Share, _>::new(ctx, net, rng);
            engine.execute(&script).await
        }

        let res = Cluster::new(3)
            .with_args([3u32, 7, 13])
            .run_with_args(program)
            .await
            .unwrap();

        assert_eq!(res, vec![23u32.into(), 23u32.into(), 23u32.into()]);
    }

    #[tokio::test]
    async fn multiplication() {
        async fn program(net: InMemoryNetwork, input: impl Into<F>) -> F {
            let me = net.id();

            let script = {
                let [a, b, c] = Exp::share_or_receive(input, me);

                let sum = a * b * c;
                let res = sum.open();
                res.finalize()
            };
            let ctx = Share::new_context(me, 3);
            let rng = rand::rngs::StdRng::from_entropy();

            let net = net.tap(format!("Party {me:?}"));
            let mut engine = Engine::<_, Share, _>::new(ctx, net, rng);

            let seeded_rng = rand::rngs::StdRng::seed_from_u64(7);
            let mut fueltanks = beaver::BeaverTriple::fake_many(&ctx, seeded_rng, 2);
            engine.add_fuel(&mut fueltanks[me.0]);
            engine.execute(&script).await
        }

        let res = Cluster::new(3)
            .with_args([1u32, 2, 3])
            .run_with_args(program)
            .await
            .unwrap();

        assert_eq!(res, vec![6u32.into(), 6u32.into(), 6u32.into()]);
    }
}
