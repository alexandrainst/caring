// #![allow(unused)]
//pub mod theater;

use std::{env, net::SocketAddr};

use caring::{network::TcpNetwork, schemes::feldman};
use rand::Rng;

#[tokio::main]
async fn main() {
    // Argument parsing
    let mut args = env::args();
    args.next();
    let me: String = args.next().unwrap();
    let me: SocketAddr = me.parse().unwrap();
    let peers: Vec<SocketAddr> = args.map(|s| s.parse().unwrap()).collect();

    // Setup TcpNetwork.
    let mut network: TcpNetwork = TcpNetwork::connect(me, &peers).await;
    println!("My id is {}", network.index);

    // Just sending some messages.
    println!("Now I am going to talk with my friends!");
    network.broadcast(&"Hello!");
    let res: Vec<Box<str>> = network.receive_all().await;
    println!("I got a message!");
    for (i, s) in res.iter().enumerate() {
        println!("I got something: {s} from {i}");
    }

    println!("Let's try some MPC");
    let start = std::time::Instant::now();
    let mut rng = rand::thread_rng();
    let num: u32 = rng.gen_range(1..100);
    println!("My number for today is: {num}");

    let num = curve25519_dalek::Scalar::from(num);
    let parties: Vec<_> = network
        .participants()
        .map(|id| (id + 1))
        .map(curve25519_dalek::Scalar::from)
        .collect();
    let shares = feldman::share::<curve25519_dalek::Scalar, curve25519_dalek::RistrettoPoint>(
        num, &parties, 2, &mut rng,
    );

    // broadcast my shares.
    println!("Sharing shares...");
    let shares = network.symmetric_unicast(shares).await;

    for s in shares.iter() {
        assert!(s.verify(), "Somebody is cheating!");
    }

    // compute
    println!("Computing...");
    let my_result = shares.into_iter().sum();
    // Should be the same for every party!
    let open_shares = network.symmetric_broadcast(my_result).await;

    println!("Reconstructing...");
    let res = feldman::reconstruct(&open_shares).expect("Bad");

    println!("Extractring u32...");
    let res: [u8; 4] = res.as_bytes()[0..4].try_into().unwrap();
    let res = u32::from_le_bytes(res);
    println!("We got {res}!");
    println!("Took {:#?}", start.elapsed());
}
