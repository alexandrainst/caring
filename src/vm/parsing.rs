use std::ops::{Add, Mul, Sub};

use crate::vm::{Instruction, Script};

struct Exp<F> {
    exp: Vec<Instruction<F>>,
}

impl<F> Exp<F> {
    pub fn share(secret: impl Into<F>) -> Self {
        Self {
            exp: vec![Instruction::Share(secret.into())],
        }
    }

    pub fn receive_input(id: usize) -> Self {
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
        net::network::InMemoryNetwork,
        protocols::beaver,
        testing::{self, Cluster},
        vm::{parsing::Exp, Engine},
    };

    type F = algebra::element::Element32;
    type Share = testing::mock::Share<F>;

    #[tokio::test]
    async fn addition() {
        async fn program(net: InMemoryNetwork, input: impl Into<F>) -> F {
            let id0 = net.index;
            let id1 = (1 + id0) % 3;
            let id2 = (2 + id0) % 3;

            let script = {
                let a = Exp::share(input);
                let b = Exp::receive_input(id1);
                let c = Exp::receive_input(id2);

                let sum = a + b + c;
                let res = sum.open();
                res.finalize()
            };
            let ctx = Share::new_context(id0, 3);
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
            let id0 = net.index;
            let id1 = (1 + id0) % 3;
            let id2 = (2 + id0) % 3;

            let script = {
                let a = Exp::share(input);
                let b = Exp::receive_input(id1);
                let c = Exp::receive_input(id2);

                let sum = a * b * c;
                let res = sum.open();
                res.finalize()
            };
            let ctx = Share::new_context(id0, 3);
            let rng = rand::rngs::StdRng::from_entropy();

            let net = net.tap(format!("Party {id0}"));
            let mut engine = Engine::<_, Share, _>::new(ctx, net, rng);

            let seeded_rng = rand::rngs::StdRng::seed_from_u64(7);
            let mut fueltanks = beaver::BeaverTriple::fake_many(&ctx, seeded_rng, 2);
            engine.add_fuel(&mut fueltanks[id0]);
            engine.execute(&script).await
        }

        let res = Cluster::new(3)
            .with_args([3u32, 7, 13])
            .run_with_args(program)
            .await
            .unwrap();

        assert_eq!(res, vec![273u32.into(), 273u32.into(), 273u32.into()]);
    }
}
