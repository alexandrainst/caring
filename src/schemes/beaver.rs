// TODO: Move to a 'common' protocol lib since this isn't a secret sharing scheme?
use std::marker::PhantomData;

use ff::Field;
use rand::RngCore;

use crate::{net::agency::Broadcast, schemes::Shared};

/// Beaver (Multiplication) Triple
#[derive(Clone)]
pub struct BeaverTriple<F, S: Shared<F>> {
    pub phantom: PhantomData<F>,
    pub shares: (S, S, S),
}

#[derive(Clone)]
pub struct BeaverSquare<F, S: Shared<F>> {
    phantom: PhantomData<F>,
    val: S,
    val_squared: S,
}


#[derive(Clone)]
pub struct BeaverPower<F, S: Shared<F>> {
    phantom: PhantomData<F>,
    val: S,
    powers: Vec<S>,
}

impl<F: Field, C, S: Shared<F, Context = C>> BeaverTriple<F, S> {
    /// Fake a set of beaver triples.
    ///
    /// This produces `n` shares corresponding to a shared beaver triple,
    /// however locally with the values known.
    ///
    /// * `ids`: ids to produce for
    /// * `threshold`: threshold to reconstruct
    /// * `rng`: rng to sample from
    pub fn fake(ctx: &C, mut rng: &mut impl RngCore) -> Vec<Self> {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c: F = a * b;
        // Share (preproccess)
        let a = S::share(ctx, a, rng);
        let b = S::share(ctx, b, rng);
        let c = S::share(ctx, c, rng);
        itertools::izip!(a, b, c)
            .map(|(a, b, c)| Self {
                shares: (a, b, c),
                phantom: PhantomData,
            })
            .collect()
    }
}

/// Perform multiplication using beaver triples
///
/// * `x`: first share to multiply
/// * `y`: second share to multiply
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn beaver_multiply<
    C,
    S: Shared<F, Context = C> + Copy + std::ops::Mul<F, Output = S>,
    F: Field + serde::Serialize + serde::de::DeserializeOwned,
>(
    ctx: &C,
    x: S,
    y: S,
    triple: BeaverTriple<F, S>,
    agent: &mut impl Broadcast,
) -> Option<S> {
    // TODO: Better error handling.
    let BeaverTriple {
        shares: (a, b, c),
        phantom: _,
    } = triple;
    let ax: S = a + x;
    let by: S = b + y;

    // Sending both at once it more efficient.
    let resp = agent.symmetric_broadcast::<_>((ax, by)).await.ok()?;
    let (ax, by): (Vec<_>, Vec<_>) = itertools::multiunzip(resp);

    let ax = S::recombine(ctx, &ax)?;
    let by = S::recombine(ctx, &by)?;

    Some(y * ax + a * (-by) + c)
}

#[cfg(test)]
mod test {

    use crate::{
        algebra::element::Element32,
        net::network::InMemoryNetwork,
        schemes::{
            beaver::{beaver_multiply, BeaverTriple},
            shamir::{self},
            ShamirParams,
        },
    };

    #[tokio::test]
    async fn beaver_mult() {
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        let ids: Vec<Element32> = (0..2u32).map(|i| i + 1).map(Element32::from).collect();

        let threshold: u64 = 2;

        // preproccessing
        let ctx = ShamirParams { threshold, ids };
        let triples = BeaverTriple::fake(&ctx, &mut rng);

        let mut taskset = tokio::task::JoinSet::new();
        // MPC
        async fn do_mpc(
            triple: BeaverTriple<Element32, shamir::Share<Element32>>,
            network: InMemoryNetwork,
            ctx: ShamirParams<Element32>,
        ) {
            let mut rng = rand::rngs::mock::StepRng::new(1, 7);
            let mut network = network;
            let v = Element32::from(5u32);
            let shares = shamir::share(v, &ctx.ids, ctx.threshold, &mut rng);
            let shares = network.symmetric_unicast(shares).await.unwrap();
            let res = beaver_multiply(&ctx, shares[0], shares[1], triple, &mut network)
                .await
                .unwrap();
            let res = network.symmetric_broadcast(res).await.unwrap();
            let res = shamir::reconstruct(&res);
            let res: u32 = res.into();
            assert_eq!(res, 25);
        }
        let cluster = InMemoryNetwork::in_memory(ctx.ids.len());
        for network in cluster {
            let triple = triples[network.index].clone();
            let ctx = ctx.clone();
            taskset.spawn(do_mpc(triple, network, ctx));
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
}
