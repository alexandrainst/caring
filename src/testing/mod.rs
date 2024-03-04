//! This module documents various tools which can be used to test or benchmark schemes.
use std::future::Future;

use tokio::task::JoinError;

use crate::net::network::InMemoryNetwork;

pub struct Cluster<Arg = ()> {
    //players: Vec<Arc<RefCell<InMemoryNetwork>>>
    players: Vec<InMemoryNetwork>, //players: tokio::task::JoinSet<InMemoryNetwork>,
    args: Option<Vec<Arg>>,
}
impl Cluster {
    pub fn new(size: usize) -> Self {
        let players = InMemoryNetwork::in_memory(size);
        Self {
            players,
            args: None,
        }
    }

    pub fn with_args<A>(self, args: impl Into<Vec<A>>) -> Cluster<A> {
        Cluster {
            args: Some(args.into()),
            players: self.players,
        }
    }

    pub async fn run<T, P, F>(self, prg: P) -> Result<Vec<T>, JoinError>
    where
        T: Send + 'static,
        P: Fn(InMemoryNetwork) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        self.run_with_args(|p, _| prg(p)).await
    }
}

impl<Arg> Cluster<Arg> {
    pub async fn run_with_args<T, P, F>(self, prg: P) -> Result<Vec<T>, JoinError>
    where
        T: Send + 'static,
        P: Fn(InMemoryNetwork, Arg) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        let futures: Vec<_> = self
            .players
            .into_iter()
            .zip(self.args.into_iter().flatten())
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
        // Yes this is a problem.
        // We really need scoped async tasks, but those don't really exist.

        Cluster::new(32)
            .run(|mut network| async move {
                let msg = "Joy to the world!".to_owned();
                network.broadcast(&msg).await.unwrap();
                let post: Vec<String> = network.receive_all().await.unwrap();
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
            })
            .await
            .unwrap();
    }
}
