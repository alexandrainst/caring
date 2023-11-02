//! This module documents various tools which can be used to test or benchmark schemes.

use std::future::Future;

use tokio::task::JoinError;

use crate::network::InMemoryNetwork;

pub struct Cluster {
    //players: Vec<Arc<RefCell<InMemoryNetwork>>>
    players: Vec<InMemoryNetwork>, //players: tokio::task::JoinSet<InMemoryNetwork>,
}

impl Cluster {
    pub fn new(size: usize) -> Self {
        let players = InMemoryNetwork::in_memory(size);
        // let players = players.into_iter()
        //     .map(RefCell::new)
        //     .map(Arc::new).collect();
        Self { players }
    }

    // pub fn spawn(players: usize) -> Self {
    //     let networks =  connection::InMemoryNetwork::in_memory(players);

    //     let channel = tokio::sync::broadcast::channel(1);
    //     let players = tokio::task::JoinSet::new();
    //     for network in networks {
    //         let channel =  channel.1;
    //         players.spawn(async {
    //             while let Ok(job) = channel.recv().await {
    //                 job(network).await;
    //             }
    //             network
    //         });
    //     }
    //     Self {players}
    // }

    pub async fn run<'a, T, P, F>(&'a mut self, prg: P) -> Result<Vec<T>, JoinError>
    where
        T: Send + 'static,
        P: Fn(&'a mut InMemoryNetwork) -> F,
        F: Future<Output = T> + Send + 'static,
    {
        let futures: Vec<_> = self
            .players
            .iter_mut()
            .map(|p| {
                let fut = prg(p);
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
        let cluster = Cluster::new(32);
        let cluster = Box::new(cluster);
        let cluster = Box::leak(cluster);

        cluster
            .run(|network| async move {
                let msg = "Joy to the world!".to_owned();
                network.broadcast(&msg);
                let post: Vec<String> = network.receive_all().await.unwrap();
                for package in post {
                    assert_eq!(package, "Joy to the world!");
                }
            })
            .await
            .unwrap();
    }
}
