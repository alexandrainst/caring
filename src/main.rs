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

fn main() {
    let mut args = env::args();
    args.next();
    let my_addr: String = args.next().unwrap();
    dbg!(&my_addr);
    let me : SocketAddr = my_addr.parse().unwrap();
    let peers : Vec<SocketAddr> = args.map(|s| s.parse().unwrap()).collect();
    let mut engine: Engine = Engine::connect(me, &peers).unwrap();
    
    let mut rng = rand::thread_rng();
    let num = rng.gen_range(1..100);
    println!("My number for today is: {num}");

    engine.execute(|engine: &mut Engine| async {
        engine.commit().await;
        println!("Something cool");
    });


}
