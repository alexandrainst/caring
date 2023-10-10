// #![allow(unused)]
mod element;
pub mod shamir;
pub mod vss;
pub mod engine;
mod connection;

use std::{env, net::SocketAddr};



use curve25519_dalek::Scalar;
use engine::Engine;
use rand::Rng;

use crate::connection::TcpNetwork;

#[tokio::main]
async fn main() {

    // #[cfg(tokio_unstable)]
    // console_subscriber::init();

    // Argument parsing
    let mut args = env::args();
    args.next();
    let me: String = args.next().unwrap();
    let me : SocketAddr = me.parse().unwrap();
    let peers : Vec<SocketAddr> = args.map(|s| s.parse().unwrap()).collect();
    // Construct the engine
    let mut network: TcpNetwork = TcpNetwork::connect(me, &peers).await;
    let mut rng = rand::thread_rng();
    let num : u32 = rng.gen_range(1..100);
    println!("My id is {}", network.index);
    println!("My number for today is: {num}");

    let start = std::time::Instant::now();
    println!("Now I am going to talk with my friends!");
    network.broadcast(&"Hello!");
    let res : Vec<Box<str>> = network.receive_all().await;
    println!("I got a message!");
    for (i,s) in res.iter().enumerate() {
        println!("I got something: {s} from {i}");
    }

    println!("Let's try some MPC");
    let num = curve25519_dalek::Scalar::from(num);
    let parties : Vec<_> = network.participants()
        .map(|id| (id + 1))
        .map(curve25519_dalek::Scalar::from).collect();
    let shares = shamir::share::<curve25519_dalek::Scalar>(num, &parties, 2, &mut rng);
    
    // broadcast my shares.
    println!("Sharing shares...");
    let shares = network.symmetric_unicast(shares).await;

    // compute
    println!("Computing...");
    let my_result = shares.into_iter().sum();
    let open_shares = network.symmetric_broadcast(my_result).await;

    println!("Reconstructing...");
    let res = shamir::reconstruct(&open_shares);

    println!("Extractring u32...");
    let res : [u8; 4] = res.as_bytes()[0..4].try_into().unwrap();
    let res = u32::from_le_bytes(res);
    println!("We got {res}!");
    println!("Took {:#?}", start.elapsed());


}
