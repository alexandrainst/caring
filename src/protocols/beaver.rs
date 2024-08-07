use std::iter;

use itertools::{izip, multiunzip};
use rand::RngCore;

use crate::{
    algebra::field::Field,
    net::{agency::Broadcast, Communicate},
    schemes::{interactive::InteractiveShared, Shared},
};

/// Beaver (Multiplication) Triple
#[derive(Clone)]
pub struct BeaverTriple<S> {
    pub shares: (S, S, S),
}

#[derive(Clone)]
pub struct BeaverPower<S: Shared> {
    val: S,
    powers: Vec<S>,
}

impl<F: Field, C: Clone, S: Shared<Value = F, Context = C>> BeaverTriple<S> {
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
        let a = S::share(ctx, a, &mut rng);
        let b = S::share(ctx, b, &mut rng);
        let c = S::share(ctx, c, &mut rng);
        itertools::izip!(a, b, c)
            .map(|(a, b, c)| Self { shares: (a, b, c) })
            .collect()
    }

    pub fn fake_many(ctx: &C, mut rng: impl RngCore, count: usize) -> Vec<Vec<Self>> {
        let zipped = iter::from_fn(|| {
            let a = F::random(&mut rng);
            let b = F::random(&mut rng);
            let c = a * b;
            Some((a, b, c))
        })
        .take(count);
        let (a, b, c): (Vec<_>, Vec<_>, Vec<_>) = multiunzip(zipped);
        // Share (preproccess)
        let a = S::share_many_naive(ctx, &a, &mut rng);
        let b = S::share_many_naive(ctx, &b, &mut rng);
        let c = S::share_many_naive(ctx, &c, &mut rng);

        itertools::izip!(a, b, c)
            .map(|(a, b, c)| {
                let mut triples = Vec::new();
                for shares in izip!(a, b, c) {
                    triples.push(Self { shares })
                }
                triples
            })
            .collect()
    }

    /// Construct a beaver triple from shares.
    ///
    /// The shares must hold the invariant that `a * b = c`,
    /// for the underlying field `F`, otherwise they are
    /// considered malformed.
    pub fn from_foreign(a: S, b: S, c: S) -> Self {
        Self { shares: (a, b, c) }
    }
}

/// Perform multiplication using beaver triples
///
/// * `ctx`: context for secret sharing scheme
/// * `x`: first share to multiply
/// * `y`: second share to multiply
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn beaver_multiply<
    F: Field,
    S: InteractiveShared<Value = F> + std::ops::Mul<S::Value, Output = S>,
>(
    ctx: &mut S::Context,
    x: S,
    y: S,
    triple: BeaverTriple<S>,
    mut coms: impl Communicate,
) -> Result<S, S::Error> {
    let BeaverTriple { shares: (a, b, c) } = triple;
    let ax: S = x + a.clone();
    let by: S = y.clone() + b;

    // TODO: concurrency
    let ax = S::recombine(ctx, ax, &mut coms).await?;
    let by = S::recombine(ctx, by, &mut coms).await?;

    Ok(y * ax + a * (-by) + c)
}

pub async fn beaver_multiply_many<
    C,
    F: Field,
    S: Shared<Value = F, Context = C> + Copy + std::ops::Mul<S::Value, Output = S>,
>(
    ctx: &C,
    xs: &[S],
    ys: &[S],
    triples: &[BeaverTriple<S>],
    agent: &mut impl Broadcast,
) -> Option<Vec<S>> {
    let mut zs = Vec::new();
    // TODO: Better error handling.
    for (triple, &x, &y) in izip!(triples, xs, ys) {
        let BeaverTriple { shares: (a, b, c) } = triple;
        let ax: S = *a + x;
        let by: S = *b + y;
        // Very sad. Very inefficient.
        let resp = agent.symmetric_broadcast::<_>((ax, by)).await.ok()?;
        let (ax, by): (Vec<_>, Vec<_>) = itertools::multiunzip(resp);

        let ax = S::recombine(ctx, &ax)?;
        let by = S::recombine(ctx, &by)?;
        let z = y * ax + *a * (-by) + *c;
        zs.push(z);
    }
    Some(zs)
}

//pub async fn beaver_multiply_many2<
//    C,
//    F: Field,
//    S: Shared<Value = F, Context = C> + Copy + std::ops::Mul<S::Value, Output = S>,
//>(
//    ctx: &C,
//    xs: &[S],
//    ys: &[S],
//    triples: &[BeaverTriple<S>],
//    agent: Network<impl SplitChannel + Send + 'static>,
//) -> Option<Vec<S>> {
//    let n = xs.len();
//    let (gateway, mut muxes) = NetworkGateway::multiplex(agent, n);
//    let iter = multizip((xs, ys, triples, muxes.iter_mut())).map(|(x, y, triple, net)| {
//        beaver_multiply(ctx, *x, *y, triple.clone(), net)
//    });
//    let zs : Option<Vec<S>> = join_all(iter).await.into_iter().collect();
//    let _ = gateway.takedown().await;
//    zs
//}

#[derive(Clone)]
pub struct BeaverSquare<S: Shared> {
    val: S,
    val_squared: S,
}

impl<F: Field, C, S: Shared<Value = F, Context = C>> BeaverSquare<S> {
    pub fn fake(ctx: &C, mut rng: impl RngCore) -> Vec<Self> {
        let a = F::random(&mut rng);
        let c: F = a * a;
        // Share (preproccess)
        let a = S::share(ctx, a, &mut rng);
        let c = S::share(ctx, c, &mut rng);
        itertools::izip!(a, c)
            .map(|(a, c)| Self {
                val: a,
                val_squared: c,
            })
            .collect()
    }

    /// Construct a beaver triple from shares.
    ///
    /// The shares must hold the invariant that `a * a = c`,
    /// for the underlying field `F`, otherwise they are
    /// considered malformed.
    pub fn from_foreign(val: S, val_squared: S) -> Self {
        Self { val, val_squared }
    }
}

/// Perform squaring using beaver's trick
///
/// * `ctx` context for secret sharing scheme
/// * `x`: first share to square
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn beaver_square<
    C,
    S: Shared<Value = F, Context = C> + Copy + std::ops::Mul<F, Output = S>,
    F: Field + serde::Serialize + serde::de::DeserializeOwned,
>(
    ctx: &C,
    x: S,
    triple: BeaverSquare<S>,
    agent: &mut impl Broadcast,
) -> Option<S> {
    // TODO: Better error handling.
    let BeaverSquare { val, val_squared } = triple;
    let ax: S = val + x;

    // Sending both at once it more efficient.
    let ax = agent.symmetric_broadcast::<_>(ax).await.ok()?;
    let ax = S::recombine(ctx, &ax)?;

    Some((x - val) * ax + val_squared)
}

#[cfg(test)]
mod test {

    use itertools::Itertools;

    use super::*;
    use crate::{
        algebra::element::Element32,
        net::{network::InMemoryNetwork, Id},
        schemes::shamir::{self, ShamirParams},
        testing::mock,
    };

    #[test]
    fn fake_shares() {
        type Share = mock::Share<Element32>;
        let ctx = Share::new_context(Id(1), 3);
        let mut rng = rand::rngs::mock::StepRng::new(7, 32);
        let triples_set = BeaverTriple::<Share>::fake_many(&ctx, &mut rng, 7);
        for (i, triples) in triples_set.into_iter().enumerate() {
            for triple in triples {
                let i = Id(i);
                assert_eq!(triple.shares.0.issued_to, i);
                assert_eq!(triple.shares.1.issued_to, i);
                assert_eq!(triple.shares.2.issued_to, i);
            }
        }
    }

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
            triple: BeaverTriple<shamir::Share<Element32>>,
            network: InMemoryNetwork,
            mut ctx: ShamirParams<Element32>,
        ) {
            let mut rng = rand::rngs::mock::StepRng::new(1, 7);
            let mut network = network;
            let v = Element32::from(5u32);
            let shares = shamir::share(v, &ctx.ids, ctx.threshold, &mut rng);
            let shares = network.symmetric_unicast(shares).await.unwrap();
            let res = beaver_multiply(&mut ctx, shares[0], shares[1], triple, &mut network)
                .await
                .unwrap();
            let res = network.symmetric_broadcast(res).await.unwrap();
            let res = shamir::reconstruct(&ctx, &res);
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

    #[tokio::test]
    async fn beaver_mult_vec() {
        type F = Element32;
        type S = shamir::Share<F>;
        let mut rng = rand::rngs::mock::StepRng::new(0, 7);
        let threshold = 2;
        let ids: Vec<Element32> = (1..=3u32).map(Element32::from).collect();
        let ctx = ShamirParams { threshold, ids };
        let triples = BeaverTriple::<S>::fake_many(&ctx, &mut rng, 2);

        let (t1, t2, t3) = triples.into_iter().collect_tuple().unwrap();

        crate::testing::Cluster::new(3)
            .with_args([([5, 2], t1), ([7, 3], t2), ([0, 0u32], t3)])
            .run_with_args(|mut network, (arg, triple)| async move {
                let ids: Vec<Element32> = (1..=3u32).map(Element32::from).collect();
                //let ctx = mock::Context{all_parties: ids.len(), me: network.index};
                let ctx = ShamirParams { threshold, ids };
                let mut rng = rand::rngs::mock::StepRng::new(1, 7);
                let x: Vec<_> = arg.into_iter().map(F::from).collect();
                let shares = S::share_many_naive(&ctx, &x, &mut rng);

                let shares: Vec<_> = network.symmetric_unicast(shares).await.unwrap();

                let (a, b, _) = shares.into_iter().collect_tuple().unwrap();

                let c = beaver_multiply_many(&ctx, &a, &b, &triple, &mut network)
                    .await
                    .unwrap();

                let shares: Vec<_> = network.symmetric_broadcast(c).await.unwrap();
                let res: Option<_> = S::recombine_many_naive(&ctx, &shares).into_iter().collect();
                let res: Vec<_> = res.unwrap();
                let res: Vec<_> = res.into_iter().map(u32::from).collect();
                assert_eq!(res, [5 * 7, 2 * 3]);
            })
            .await
            .unwrap();
    }
}
