use std::{
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    time::Duration,
};

use itertools::Itertools;
use turmoil::{
    self,
    net::{TcpListener, TcpStream},
    ToIpAddrs,
};

fn setup(others: impl ToIpAddrs) -> (SocketAddr, Vec<SocketAddr>) {
    let port = 0;
    let me = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port));

    let others = turmoil::lookup_many(others)
        .into_iter()
        .map(|addr| SocketAddr::new(addr, port))
        .collect_vec();

    (me, others)
}

#[test]
fn hello_network() {
    use caring::net::network::TcpNetwork;
    let mut sim = turmoil::Builder::new().build();

    sim.host("party0", || async {
        let (me, peers) = setup("party1, party2");
        let mut network = TcpNetwork::connect(me, &peers).await?;
        let res = network.symmetric_broadcast("hello".to_string()).await?;
        assert_eq!(res, vec!["hello1", "hello2", "hello3"]);
        Ok(())
    });

    sim.host("party1", || async {
        let (me, peers) = setup("party0, party2");
        let mut network = TcpNetwork::connect(me, &peers).await?;
        let res = network.symmetric_broadcast("hello".to_string()).await?;
        assert_eq!(res, vec!["hello1", "hello2", "hello3"]);
        Ok(())
    });

    sim.host("party2", || async {
        let (me, peers) = setup("party0, party1");
        let mut network = TcpNetwork::connect(me, &peers).await?;
        let res = network.symmetric_broadcast("hello".to_string()).await?;
        assert_eq!(res, vec!["hello1", "hello2", "hello3"]);
        Ok(())
    });

    sim.run().unwrap();
}

#[test]
fn connection() {
    use caring::net::connection::TcpConnection;
    let mut sim = turmoil::Builder::new().build();
    const PORT: u16 = 1792;

    sim.host("party0", || async {
        let (stream, _) = TcpListener::bind(format!("0.0.0.0:{PORT}"))
            .await?
            .accept()
            .await?;
        let mut con = TcpConnection::from_tcp_stream(stream);
        let msg: String = con.recv().await?;
        assert_eq!(msg, "hello, party0");
        con.send(&"hello, party1".to_string()).await?;
        Ok(())
    });

    sim.host("party1", || async {
        let addr = turmoil::lookup("party1");
        let stream = TcpStream::connect((addr, PORT)).await?;
        let mut con = TcpConnection::from_tcp_stream(stream);
        con.send(&"hello, party0".to_string()).await?;
        let msg: String = con.recv().await?;
        assert_eq!(msg, "hello, party1");
        Ok(())
    });

    sim.run().unwrap();
}
