use std::marker::PhantomData;

use ff::Field;
use rand::RngCore;

use crate::{
    agency::Broadcast, schemes::Shared,
};

#[derive(Clone)]
pub struct BeaverTriple<F, S: Shared<F>> {
    phantom: PhantomData<F>,
    shares: (S, S, S),
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
    pub fn fake(ctx: &mut C, mut rng: &mut impl RngCore) -> Vec<Self> {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c: F = a * b;
        // Share (preproccess)
        let a = S::share(ctx, a);
        let b = S::share(ctx, b);
        let c = S::share(ctx, c);
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
    S: Shared<F, Context=C> + Copy + std::ops::Mul<F, Output=S>,
    F: Field + serde::Serialize + serde::de::DeserializeOwned,
    E : std::fmt::Debug,
>(
    ctx: &mut C,
    x: S,
    y: S,
    triple: BeaverTriple<F, S>,
    network: &mut impl Broadcast<E>,
) -> Option<S> {
    let BeaverTriple {
        shares: (a, b, c),
        phantom: _,
    } = triple;
    let ax: S = a + x;
    let by: S = b + y;

    // Sending both at once it more efficient.
    let resp = network.symmetric_broadcast::<_>((ax, by)).await.unwrap();
    let (ax, by): (Vec<_>, Vec<_>) = itertools::multiunzip(resp);

    let ax = S::recombine(ctx, &ax)?;
    let by = S::recombine(ctx, &by)?;

    Some(y * ax + a * (-by) + c)
}

#[cfg(test)]
mod test {
    use rand::{thread_rng, RngCore};

    use crate::{element::Element32, schemes::{beaver::{BeaverTriple, beaver_multiply}, shamir::{share, self}, ShamirParams}, network::InMemoryNetwork};


    #[tokio::test]
    async fn beaver_mult() {
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        let ids: Vec<Element32> = (0..2u32).map(|i| i + 1).map(Element32::from).collect();

        let threshold: u64 = 2;


        // preproccessing
        let ctx = ShamirParams {threshold, ids, rng: Box::new(rng)};
        let triples = BeaverTriple::fake(&mut ctx, &mut rng);

        let mut taskset = tokio::task::JoinSet::new();
        // MPC
        async fn do_mpc(triple: BeaverTriple<Element32, shamir::Share<Element32>>, network: InMemoryNetwork, mut ctx: ShamirParams<Element32>) {
            let mut rng = rand::rngs::mock::StepRng::new(1, 7);
            let mut network = network;
            let v = Element32::from(5u32);
            let shares = shamir::share(v, &ctx.ids, ctx.threshold, &mut rng);
            let shares = network.symmetric_unicast(shares).await.unwrap();
            let res = beaver_multiply(&mut ctx, shares[0], shares[1], triple, &mut network)
                .await
                .unwrap();
            let res = network.symmetric_broadcast(res).await.unwrap();
            let res = shamir::reconstruct(&res);
            let res: u32 = res.into();
            assert_eq!(res, 25);

        }
        let cluster = InMemoryNetwork::in_memory(ids.len());
        for network in cluster {
            let ids = ids.clone();
            let triple = triples[network.index].clone();
            let ctx = ShamirParams {threshold, ids, rng: Box::new(rng)};
            taskset.spawn(do_mpc(triple, network, ctx));
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }

}
