//! Very much WIP.
use ff::Field;
use itertools::{Either, Itertools};
use rand::RngCore;

use crate::{net::Channel, schemes::Shared};

// TODO: Abstract this better to both 2 and N parties
pub async fn choose<Ctx, F: Field, S: Shared<Value = F, Context = Ctx>>(
    ctx: &Ctx,
    rng: &mut impl RngCore,
    cx: &mut impl Channel,
    payload: Vec<S>,
    m: usize,
    verifier: impl Fn(&[F]) -> bool,
) -> Result<Vec<S>, &'static str> {
    let n = payload.len();

    // randomly select sacrificed payloads

    use rand::seq::IteratorRandom;
    let to_check = (0..n).choose_multiple(rng, m);

    dbg!(&to_check);
    let (sacrificed, spared): (Vec<_>, Vec<_>) =
        payload.into_iter().enumerate().partition_map(|(i, s)| {
            if to_check.contains(&i) {
                Either::Left(s)
            } else {
                Either::Right(s)
            }
        });

    assert_eq!(sacrificed.len(), m);
    cx.send(&to_check).await.unwrap();
    let others_sacrificed = cx.recv().await.unwrap();

    let sacrificed = [others_sacrificed, sacrificed]; // TODO: This order matters,
                                                      // and needs to derived.

    let sacrificed: Vec<F> = S::recombine_many(ctx, &sacrificed)
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .unwrap();

    assert_eq!(sacrificed.len(), m);
    if verifier(&sacrificed) {
        cx.send(&true).await.unwrap();
        Ok(spared)
    } else {
        cx.send(&false).await.unwrap();
        Err("verification failed")
    }
}

pub async fn cut<Ctx, F: Field, S: Shared<Value = F, Context = Ctx> + Sync>(
    _ctx: &Ctx,
    cx: &mut impl Channel,
    payload: Vec<S>,
    _m: usize,
) -> Result<Vec<S>, ()> {
    // TODO: Send payload?

    // receive the list of sacrificed payloads
    let to_check: Vec<usize> = cx.recv().await.unwrap();
    let (sacrificed, spared): (Vec<_>, Vec<_>) =
        payload.into_iter().enumerate().partition_map(|(i, s)| {
            if to_check.contains(&i) {
                Either::Left(s)
            } else {
                Either::Right(s)
            }
        });

    // let _ = cx.symmetric_broadcast(sacrificed).await.unwrap();
    cx.send(&sacrificed).await.unwrap();
    let accepted = cx.recv().await.unwrap();
    if accepted {
        Ok(spared)
    } else {
        Err(())
    }
}

#[cfg(test)]
mod test {
    use ff::{Field, PrimeField};
    use rand::rngs::mock::StepRng;

    use crate::{
        algebra::element::Element32,
        net::{
            agency::Broadcast,
            connection::{self, DuplexConnection},
            Channel, SplitChannel,
        },
        protocols::cutnchoose::{choose, cut},
        schemes::{
            shamir::{self, ShamirParams},
            Shared,
        },
    };

    struct SingleBroadcast {
        inner: DuplexConnection,
        is_first: bool,
    }

    impl Broadcast for SingleBroadcast {
        type BroadcastError = <DuplexConnection as Channel>::Error;

        fn broadcast(
            &mut self,
            msg: &(impl serde::Serialize + Sync),
        ) -> impl std::future::Future<Output = Result<(), Self::BroadcastError>> {
            self.inner.send(msg)
        }

        async fn symmetric_broadcast<T>(&mut self, msg: T) -> Result<Vec<T>, Self::BroadcastError>
        where
            T: serde::Serialize + serde::de::DeserializeOwned + Sync,
        {
            let (cx, rx) = self.inner.split();
            let (recv, send): (Result<T, _>, _) = futures::join!(rx.recv(), cx.send(&msg));
            send?;
            let other = recv?;
            if self.is_first {
                Ok(vec![msg, other])
            } else {
                Ok(vec![other, msg])
            }
        }

        fn recv_from<T: serde::de::DeserializeOwned>(
            &mut self,
            _idx: usize,
        ) -> impl futures::prelude::Future<Output = Result<T, Self::BroadcastError>> {
            self.inner.recv()
        }

        fn size(&self) -> usize {
            2
        }
    }

    impl Channel for SingleBroadcast {
        type Error = <DuplexConnection as Channel>::Error;

        fn send<T: serde::Serialize + Sync>(
            &mut self,
            msg: &T,
        ) -> impl futures::prelude::Future<Output = Result<(), Self::Error>> {
            self.inner.send(msg)
        }

        fn recv<T: serde::de::DeserializeOwned>(
            &mut self,
        ) -> impl futures::prelude::Future<Output = Result<T, Self::Error>> {
            self.inner.recv()
        }
    }

    #[tokio::test]
    async fn sunshine() {
        type F = Element32;
        type S = shamir::Share<Element32>;
        let n = 32;
        let (ch1, ch2) = connection::DuplexConnection::in_memory();
        let t1 = async move {
            let ids = vec![F::from_u128(1), F::from_u128(2)];
            let ctx = ShamirParams { threshold: 2, ids };
            let mut rng = StepRng::new(32, 7);
            let mut ch = SingleBroadcast {
                inner: ch1,
                is_first: true,
            };

            let (z1, z2): (Vec<_>, Vec<_>) = (0..n)
                .map(|_| S::share(&ctx, F::ZERO, &mut rng))
                .map(|shares| (shares[0], shares[1]))
                .unzip();

            assert!(z2.len() == 32);
            ch.send(&z2).await.unwrap();

            cut(&ctx, &mut ch, z1, 16).await.unwrap();
        };
        let t2 = async move {
            let ids = vec![F::from_u128(1), F::from_u128(2)];
            let ctx = ShamirParams { threshold: 2, ids };
            let mut rng = StepRng::new(32, 7);
            let mut ch = SingleBroadcast {
                inner: ch2,
                is_first: false,
            };

            let z2: Vec<S> = ch.recv().await.unwrap();
            assert!(z2.len() == 32);
            choose(&ctx, &mut rng, &mut ch, z2, 16, |opened: &[F]| {
                dbg!(&opened);
                opened.iter().all(|s| s.is_zero_vartime())
            })
            .await
            .unwrap();
        };

        futures::join!(t1, t2);
    }
}
