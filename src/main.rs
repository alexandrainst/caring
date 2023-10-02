// #![allow(unused)]
mod element;
mod float;
pub mod field;
pub mod shamir;
mod vss;
pub mod engine;

use std::{env, net::SocketAddr};

use curve25519_dalek::Scalar;
use nalgebra::{Matrix, U4, SMatrix};
use nalgebra::Dyn;

type Weird = SMatrix<shamir::Share<Scalar>, 2, 2>;

use engine::Engine;

fn main() {
    let mut args = env::args();
    args.next();
    let my_addr: String = args.next().unwrap();
    dbg!(&my_addr);
    let my_addr : SocketAddr = my_addr.parse().unwrap();
    let peers : Vec<_> = args.map(|s| s.parse().unwrap()).collect();
    let _engine: Option<Engine> = Engine::connect(my_addr, &peers);

    let mut b: i32 = 2;
    

    b += 3;
}
