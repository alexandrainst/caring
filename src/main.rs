// #![allow(unused)]
mod element;
mod float;
pub mod field;
pub mod shamir;
mod vss;
pub mod engine;

use std::{env, net::SocketAddr};



use curve25519_dalek::Scalar;
use engine::Engine;
use rand::Rng;

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
    let mut engine: Engine = Engine::connect(me, &peers).await;
    let mut rng = rand::thread_rng();
    let num : u32 = rng.gen_range(1..100);
    println!("My number for today is: {num}");

    println!("Now I am going to talk with my friends!");
    engine.broadcast(b"Hello!");
    let res : Vec<[u8; 6]> = engine.recv_from_all().await;
    println!("I got a message!");
    for r in res {
        let s = String::from_utf8_lossy(&r);
        println!("I got something: {s}");
    }
    // Decide on some IDs
    let my_id : u32 = rng.gen();
    engine.broadcast(&my_id.to_le_bytes());
    let their_id = engine.recv_from_all().await;
    let their_id = u32::from_le_bytes(their_id[0]);
    let mut ids = [my_id, their_id];
    ids.sort();
    let (my_id, their_id) : (u32, u32) = if ids[0] == my_id {(1, 2)} else {(2, 1)};
    println!("I am {my_id}, and my friend is {their_id}");
    let mut ids = [my_id, their_id];
    ids.sort();



    println!("Let's try some MPC");
    let num = curve25519_dalek::Scalar::from(num);
    let parties : Vec<_> = ids.into_iter().map(curve25519_dalek::Scalar::from).collect();
    let shares = shamir::share::<curve25519_dalek::Scalar>(num, &parties, 2, &mut rng);
    
    // broadcast my shares.
    println!("Sharing shares...");
    dbg!(&shares);
    let share1 = shares[(my_id - 1) as usize];
    let msg = shares[(their_id - 1) as usize].y.as_bytes();
    engine.broadcast(msg);
    
    // receive my part.
    let msg = engine.recv_from_all().await[0];
    let msg = Scalar::from_canonical_bytes(msg).unwrap();
    let share2 = shamir::Share{x: my_id.into(), y: msg};

    // compute
    println!("Computing...");
    let my_result = share1 + share2;
    engine.broadcast(my_result.y.as_bytes());
    let msg = engine.recv_from_all().await[0];
    let msg = Scalar::from_canonical_bytes(msg).unwrap();
    let their_result = shamir::Share{x: their_id.into(), y: msg};

    dbg!(&my_result);
    dbg!(&their_result);
    println!("Reconstructing...");
    let res = shamir::reconstruct(&[my_result, their_result]);
    dbg!(&res);
    let res : [u8; 4] = res.as_bytes()[0..4].try_into().unwrap();
    let res = u32::from_le_bytes(res);
    println!("We got {res}!");


}
