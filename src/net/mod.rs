use futures::Future;

use crate::net::{agency::Broadcast, connection::{ConnectionError, Connection}};

pub mod agency;
pub mod connection;
pub mod network;



// TODO: Serde trait bounds on `T`
// TODO: Properly use this trait for other things (Connection/Agency etc.)
pub trait Channel {
    type Error;

    fn send<T: serde::Serialize>(&self, msg: &T) -> impl Future<Output = Result<(), Self::Error>>;

    fn recv<T: serde::de::DeserializeOwned>(&mut self) -> impl Future<Output = Result<T, Self::Error>>;
}


impl<R: tokio::io::AsyncRead + std::marker::Unpin, W: tokio::io::AsyncWrite + std::marker::Unpin> Channel for Connection<R,W> {
    type Error = ConnectionError;

    fn send<T: serde::Serialize>(&self, msg: &T) -> impl Future<Output = Result<(), Self::Error>> {
        async {todo!()}
    }

    fn recv<T: serde::de::DeserializeOwned>(&mut self) -> impl Future<Output = Result<T, Self::Error>> {
        Connection::recv(self)
    }
}


pub trait ChannelStation : Broadcast + Unicast {
    type Error = ConnectionError;
    type SubChannel : Channel<Error=Self::Error>;
    type Idx;

    fn tune_mut(&mut self, idx: Self::Idx) -> &mut Channel;

    fn tune(&self, idx: Idx) -> &Channel;
}
