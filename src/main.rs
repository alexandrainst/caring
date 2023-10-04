// #![allow(unused)]
mod element;
mod float;
pub mod field;
pub mod shamir;
mod vss;
pub mod engine;

use std::{env, net::SocketAddr};



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
    let num = rng.gen_range(1..100);
    println!("My number for today is: {num}");

    println!("Now I am going to talk with my friends!");
    engine.broadcast(b"Hello!");
    let res : Vec<[u8; 4]> = engine.recv_from_all().await;
    println!("I got a message!");
    for r in res {
        let s = String::from_utf8_lossy(&r);
        println!("I got something: {s}");
    }
}
