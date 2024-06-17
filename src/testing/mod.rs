//! This module documents various tools which can be used to test or benchmark schemes.
pub mod mock;

use crate::net::network::InMemoryNetwork;
use std::future::Future;
use tokio::task::JoinError;

pub struct Cluster<Arg = ()> {
    players: Vec<InMemoryNetwork>, //players: tokio::task::JoinSet<InMemoryNetwork>,
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

    /// Provide arguments of type `A` to the cluster as a list with each networked party.
    /// The arguments are to be provided as a list of size eqaul to the clister
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
    pub fn with_args<A>(self, args: impl Into<Vec<A>>) -> Cluster<A> {
        let args: Vec<A> = args.into();
        let m = args.len();
        let n = self.players.len();
        assert_eq!(m, n, "Cluster of size {n} requies {n} arguments, got {m}");
        Cluster {
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
        P: Fn(InMemoryNetwork) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        self.run_with_args(|p, _| prg(p)).await
    }
}

impl<Arg: Clone> Cluster<Arg> {
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
        P: Fn(InMemoryNetwork, Arg) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        let futures: Vec<_> = self
            .players
            .into_iter()
            .zip(self.args.into_iter())
            .map(|(p, arg)| {
                let fut = prg(p, arg);
                tokio::spawn(fut)
            })
            .collect();

        futures::future::join_all(futures)
            .await
            .into_iter()
            .collect()
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
