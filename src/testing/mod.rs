//! This module documents various tools which can be used to test or benchmark schemes.
pub mod mock;

use crate::{
    algebra::element::Element32,
    net::{agency::Broadcast, network::InMemoryNetwork},
    protocols::beaver,
    vm::{Engine, Script},
};
use futures::TryFutureExt;
use rand::rngs::mock::StepRng;
use std::{error::Error, future::Future};
use tokio::task::JoinError;

pub struct Cluster<Arg = (), Net = InMemoryNetwork> {
    players: Vec<Net>, //players: tokio::task::JoinSet<InMemoryNetwork>,
    args: Vec<Arg>,
}
impl Cluster {
    /// Create a new cluster with `n` parties
    pub fn new(n: usize) -> Self {
        let players = InMemoryNetwork::in_memory(n);
        Self {
            args: vec![(); n],
            players,
        }
    }
}

impl<C> Cluster<(), C> {
    /// Provide arguments of type `A` to the cluster as a list with each networked party.
    /// The arguments are to be provided as a list of size eqaul to the clister
    ///
    /// # Example
    /// ```rust
    /// use caring::testing::*;             //      party 1      party 2
    ///                                     //    v--------v   v--------v
    /// let cluster = Cluster::new(2).with_args([(1, "alice"), (2, "bob")])
    /// cluster.run_with_args(|com, (id, name)| async move {
    ///     println!("[{id}] hello from {name}");
    /// }).await.unwrap();
    ///
    /// ```
    pub fn with_args<A>(self, args: impl Into<Vec<A>>) -> Cluster<A, C> {
        let args: Vec<A> = args.into();
        let m = args.len();
        let n = self.players.len();
        assert_eq!(m, n, "Cluster of size {n} requies {n} arguments, got {m}");
        Cluster::<A, C> {
            args,
            players: self.players,
        }
    }

    /// Run a cluster
    ///
    /// # Returns
    /// A vector of results from the parties
    ///
    /// # Errors
    /// Returns a `[tokio::task::JoinError]` upon a any panic'ed party.
    pub async fn run<T, P, F>(self, prg: P) -> Result<Vec<T>, JoinError>
    where
        T: Send + 'static,
        P: Fn(C) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        self.run_with_args(|p, _| prg(p)).await
    }
}

impl<Arg, C> Cluster<Arg, C> {
    pub fn more_args<B>(self, args: impl Into<Vec<B>>) -> Cluster<(Arg, B), C> {
        let args: Vec<_> = self.args.into_iter().zip(args.into()).collect();
        Cluster {
            players: self.players,
            args,
        }
    }

    /// Run a cluster with arguments
    ///
    /// # Example
    /// ```rust
    /// use caring::testing::*;             //      party 1      party 2
    ///                                     //    v--------v   v--------v
    /// let cluster = Cluster::new(2).with_args([(1, "alice"), (2, "bob")])
    /// cluster.run_with_args(|net, (id, name)| async move {
    ///     println!("[{id}] hello from {name}");
    /// }).await.unwrap();
    ///
    /// ```
    pub async fn run_with_args<T, P, F>(self, prg: P) -> Result<Vec<T>, JoinError>
    where
        T: Send + 'static,
        P: Fn(C, Arg) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        let n = self.players.len();
        tracing::info!("Starting cluster with {n} players");
        let futures: Vec<_> = self
            .players
            .into_iter()
            .zip(self.args.into_iter())
            .enumerate()
            .map(|(id, (p, arg))| {
                let fut = prg(p, arg);
                use tracing::Instrument;
                let span = tracing::info_span!("Player", id = id);
                tokio::spawn(fut.instrument(span)).inspect_err(move |e| {
                    let reason = e.source();
                    if e.is_panic() {
                        tracing::error!("Player {id} panic'ed: {e}, reason: {reason:#?}");
                    } else if e.is_cancelled() {
                        tracing::error!("Player {id} was cancelled: {e}, reason: {reason:#?}");
                    } else {
                        tracing::error!("Player {id} returned an error: {e}, reason: {reason:#?}");
                    }
                })
            })
            .collect();

        futures::future::join_all(futures)
            .await
            .into_iter()
            .collect()
    }
}

impl Cluster<Script<Element32>> {
    pub async fn execute_mock(self) -> Result<Vec<Element32>, JoinError> {
        self.run_with_args(|network, script| async move {
            type S = mock::Share<Element32>;
            let context = mock::Context {
                all_parties: network.size(),
                me: network.id(),
            };
            let private_rng = StepRng::new(42, 7 + network.id().0 as u64);
            let shared_rng = StepRng::new(3, 14);
            let mut fueltanks = beaver::BeaverTriple::fake_many(&context, shared_rng, 2000);
            let mut engine = Engine::<_, S, _>::new(context, network, private_rng);
            engine.add_fuel(&mut fueltanks[context.me.0]);
            engine.execute(&script).await.unwrap_single()
        })
        .await
    }
}

#[cfg(test)]
mod test {
    use crate::testing::Cluster;

    #[tokio::test]
    async fn hello() {
        let c: u32 = Cluster::new(32)
            .run(|mut network| async move {
                let msg = "Joy to the world!".to_owned();
                network.broadcast(&msg).await.unwrap();
                let post: Vec<String> = network.receive_all().await.unwrap();
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
                1 // to check that we actually run
            })
            .await
            .unwrap()
            .iter()
            .sum();
        assert_eq!(c, 32);
    }
}
