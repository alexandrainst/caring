//! Replicated Secret Sharing over Three Parites (Rep3)
//!
//! For an example see
//! https://medium.com/partisia-blockchain/mpc-techniques-series-part-1-secret-sharing-d8f98324674a
//!

use ff::Field;
use rand::RngCore;

use derive_more::{Add, Sub};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    net::{Id, Tuneable},
    schemes::interactive::InteractiveMult,
};

#[derive(Debug, Clone, Copy, Add, Sub, Serialize, Deserialize)]
pub struct Share<F: Field>(F, F);

pub fn share<F: Field>(secret: F, mut rng: impl RngCore) -> [Share<F>; 3] {
    let mut shares = [0; 3].map(|_| F::random(&mut rng));
    let sum = shares[0] + shares[1] + shares[2];
    shares[0] += secret - sum;

    [(0, 1), (1, 2), (2, 0)].map(|(i, j)| Share(shares[i], shares[j]))
}

pub fn recombine<F: Field>(shares: &[Share<F>; 3]) -> F {
    shares[0].0 + shares[1].0 + shares[2].0
}

impl<F: Field + Serialize + DeserializeOwned> super::Shared for Share<F> {
    type Context = ();
    type Value = F;

    fn share(_ctx: &Self::Context, secret: F, mut rng: impl RngCore) -> Vec<Self> {
        share(secret, &mut rng).to_vec()
    }

    fn recombine(_ctx: &Self::Context, shares: &[Self]) -> Option<F> {
        let shares: &[_; 3] = shares.try_into().unwrap();
        Some(recombine(shares))
    }
}

pub async fn multiplication<F: Field + Serialize + DeserializeOwned>(
    a: Share<F>,
    b: Share<F>,
    cx: &mut impl Tuneable,
) -> Share<F> {
    let (prev_id, next_id) = match cx.id().0 {
        0 => (2, 1),
        1 => (0, 2),
        2 => (1, 0),
        _ => panic!("ID higher than 3"),
    };
    let next_id = Id(next_id);
    let prev_id = Id(prev_id);

    let c0 = (a.0 * b.0) + (a.0 * b.1) + (a.1 * b.0);

    // TODO: randomization (currently we lose privacy);

    cx.send_to(next_id, &c0).await.unwrap();
    let c1 = cx.recv_from(prev_id).await.unwrap();
    dbg!(cx.id(), c1);

    Share(c0, c1)
}

impl<F: Field + serde::Serialize + serde::de::DeserializeOwned> InteractiveMult for Share<F> {
    async fn interactive_mult<U: Tuneable>(
        _ctx: &Self::Context,
        net: &mut U,
        a: Self,
        b: Self,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(multiplication(a, b, net).await)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::algebra::element::Element32;
    use itertools::{izip, Itertools};

    #[test]
    fn identity() {
        type F = Element32;
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);
        let a = 42;
        let b: u32 = {
            let shares = share(F::from(a), &mut rng);
            recombine(&shares).into()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn add() {
        type F = crate::algebra::element::Element32;
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);
        let (a, b): (u32, u32) = (3, 3);
        let c: u32 = {
            let a = share(F::from(a), &mut rng);
            let b = share(F::from(b), &mut rng);
            let c: [_; 3] = izip!(a, b)
                .map(|(a, b)| a + b)
                .collect_vec()
                .try_into()
                .unwrap();
            recombine(&c).into()
        };
        assert_eq!(c, a + b);
    }

    #[tokio::test]
    async fn mult() {
        type F = Element32;
        let cluster = crate::testing::Cluster::new(3);
        cluster
            .with_args([4u32, 3, 0])
            .run_with_args(|mut network, input| async move {
                let mut rng = rand::rngs::mock::StepRng::new(42, 7);

                // secret-sharing
                let shares = share::<F>(input.into(), &mut rng);
                let shares: Vec<Share<F>> =
                    network.symmetric_unicast(shares.to_vec()).await.unwrap();
                let [a, b, ..] = shares[..] else {
                    // drop the third share.
                    panic!("Can't multiply with only one share")
                };

                // mpc
                let c = multiplication(a, b, &mut network).await;

                // opening
                let shares = network.symmetric_broadcast(c).await.unwrap();
                let c: u32 = recombine(&shares.try_into().unwrap()).into();
                assert_eq!(c, 4 * 3);
            })
            .await
            .unwrap();
    }

    #[ignore = "Known bug. The output becomes malformed after the second mult"]
    #[tokio::test]
    async fn mult_twice() {
        type F = Element32;
        let cluster = crate::testing::Cluster::new(3);
        cluster
            .with_args([4u32, 3, 2])
            .run_with_args(|mut network, input| async move {
                let mut rng = rand::rngs::mock::StepRng::new(42, 7);

                // secret-sharing
                let shares = share::<F>(input.into(), &mut rng);
                let shares: Vec<Share<F>> =
                    network.symmetric_unicast(shares.to_vec()).await.unwrap();
                let [a, b, c] = shares[..] else {
                    // drop the third share.
                    panic!("Can't multiply with only one share")
                };

                {
                    // debugging
                    let shares = network.symmetric_broadcast(c).await.unwrap();
                    let c: u32 = recombine(&shares.try_into().unwrap()).into();
                    dbg!(c);
                }

                let d = multiplication(a, b, &mut network).await; // first multiplication

                {
                    // debugging
                    let shares = network.symmetric_broadcast(d).await.unwrap();
                    let d: u32 = recombine(&shares.try_into().unwrap()).into();
                    dbg!(d);
                }

                let e = multiplication(d, c, &mut network).await; // second multiplication

                // opening
                let shares = network.symmetric_broadcast(e).await.unwrap();
                let e: u32 = recombine(&shares.try_into().unwrap()).into();
                assert_eq!(e, 4 * 3 * 2);
            })
            .await
            .unwrap();
    }
}
