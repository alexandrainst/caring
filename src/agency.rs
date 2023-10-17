use tokio::io::{AsyncWrite, AsyncRead};

use crate::connection::Network;

pub trait Broadcast {
    fn broadcast(&mut self, msg: &impl serde::Serialize);

    // TODO: Reconsider this
    #[allow(async_fn_in_trait)]
    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned;

}

pub trait Unicast {
    fn unicast(&mut self, msgs: &[impl serde::Serialize]);

    // TODO: Reconsider this
    #[allow(async_fn_in_trait)]
    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned;
}



impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Unicast for Network<R,W> {
    fn unicast(&mut self, msgs: &[impl serde::Serialize]) {
        self.unicast(msgs)
    }

    async fn symmetric_unicast<T>(&mut self, msgs: Vec<T>) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned {
        self.symmetric_unicast(msgs).await
    }
}

impl<R: AsyncRead + Unpin, W: AsyncWrite + Unpin> Broadcast for Network<R,W> {
    fn broadcast(&mut self, msg: &impl serde::Serialize) {
        self.broadcast(msg)
    }

    async fn symmetric_broadcast<T>(&mut self, msg: T) -> Vec<T>
        where T: serde::Serialize + serde::de::DeserializeOwned {
            self.symmetric_broadcast(msg).await
    }
}
