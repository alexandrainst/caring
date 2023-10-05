// #![allow(unused)]
mod element;
pub mod shamir;
pub mod vss;
pub mod engine;

use std::{env, net::SocketAddr};



use curve25519_dalek::Scalar;
use engine::Engine;
use rand::Rng;

use crate::engine::PartyID;

fn helper<T>(mut vec: Vec<(PartyID, T)>) -> Vec<T> {
    vec.sort_unstable_by_key(|p| p.0);
    vec.into_iter().map(|(_, t)| t).collect()
}

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

    let start = std::time::Instant::now();
    println!("Now I am going to talk with my friends!");
    engine.broadcast(b"Hello!");
    let res : Vec<(_, [u8; 6])> = engine.recv_from_all().await;
    println!("I got a message!");
    for r in res {
        let s = String::from_utf8_lossy(&r.1);
        let friend = r.0.0;
        println!("I got something: {s} from {friend}");
    }

    let my_id = engine.id;
    let binding = engine.participants();
    let their_id = binding.iter().find(|&id| my_id != *id).unwrap();
    println!("I am {}, they are {}", my_id.0, their_id.0);

    println!("Let's try some MPC");
    let num = curve25519_dalek::Scalar::from(num);
    let parties : Vec<_> = engine.participants().into_iter()
        .map(|id| id.0 + 1)
        .map(curve25519_dalek::Scalar::from).collect();
    let shares = shamir::share::<curve25519_dalek::Scalar>(num, &parties, 2, &mut rng);
    
    // broadcast my shares.
    println!("Sharing shares...");
    dbg!(&shares);
    let share1 = shares[my_id.idx()];
    let msg = shares[their_id.idx()].y.as_bytes();
    let msg = helper(engine.symmetric_broadcast(*msg).await)[their_id.idx()];
    let msg = Scalar::from_canonical_bytes(msg).unwrap();
    let share2 = shamir::Share{x: (my_id.0 + 1).into(), y: msg};
    dbg!(share2);

    // compute
    println!("Computing...");
    let my_result = share1 + share2;
    let msg = my_result.y.as_bytes();
    let msg = helper(engine.symmetric_broadcast(*msg).await)[their_id.idx()];
    let msg = Scalar::from_canonical_bytes(msg).unwrap();
    let their_result = shamir::Share{x: (their_id.0 + 1).into(), y: msg};

    dbg!(&my_result);
    dbg!(&their_result);
    println!("Reconstructing...");
    let res = shamir::reconstruct(&[my_result, their_result]);
    dbg!(&res);
    let res : [u8; 4] = res.as_bytes()[0..4].try_into().unwrap();
    let res = u32::from_le_bytes(res);
    println!("We got {res}!");
    println!("Took {:#?}", start.elapsed());


}
