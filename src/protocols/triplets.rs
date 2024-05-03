//// Lets start by doing the simplest version possible.
//// We vil start even without the HE, just to make the main operations work first.
//// We also ignore mac values and other checks.
//// We need the triplets mainly for SPDZ. Therefor we will pause the efforts for now in order to look at SPDZ.

//use ff::PrimeField;
//use rand::RngCore;

//use crate::{
    //net::agency::Broadcast,
    ////schemes::spdz::{self, SpdzContext}
    //schemes::spdz,
//};

///// Multiplication Triple
//#[derive(Clone)]
//pub struct MultiplicationTriple<F: PrimeField> {
    //pub shares: (spdz::Share<F>, spdz::Share<F>, spdz::Share<F>),
//}

//// Note: We need to be able to make shares directly from field elements. There is not methood to do that right now.
//// Therefor we will first specialize it, to only work for SPDZ, and make a share from the struct there.
//// Later we might consider generalizeing it.
//impl<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned> MultiplicationTriple<F> {
    ///// This is based om beaver.rs fake beaver triples.
    ///// I it suppose to produces 3n shares corresponding to a beaver triple shared amoung n parties,
    /////

    //// We change here from using the context, to only askring for the specific information that we need.
    //pub async fn make_triplet(
        //mut rng: &mut impl RngCore,
        //agent: &mut impl Broadcast,
        //is_chosen_party: bool,
    //) -> Self {
        //let ai = spdz::Share {
            //val: F::random(&mut rng),
            //mac: F::random(&mut rng),
        //};
        //let bi = spdz::Share {
            //val: F::random(&mut rng),
            //mac: F::random(&mut rng),
        //};
        //let mut ci = spdz::Share {
            //val: F::random(&mut rng),
            //mac: F::random(&mut rng),
        //};

        //// TODO: Now the values are suppoed to be encrypted using a 1-D HE
        //let ai_e = ai;
        //let bi_e = bi;
        //let ci_e = ci;

        //// Then each party do a broadcast of the encrypted elements.
        //// TODO-later: send all three at the same time, more efficient.
        //let a_vec = agent.symmetric_broadcast(ai_e).await.unwrap();
        //let b_vec = agent.symmetric_broadcast(bi_e).await.unwrap();
        //let c_vec = agent.symmetric_broadcast(ci_e).await.unwrap();
        //// Then the encripted version of a, b and c_ is summed up
        //let a: F = a_vec.iter().map(|s| s.val).sum();
        //let b: F = b_vec.iter().map(|s| s.val).sum();
        //let c: F = c_vec.iter().map(|s| s.val).sum();

        //// TODO: Then encrypted Delta is found as [a]*[b] - [c_]
        //let delta = a * b - c;

        //// TODO: Then there is done a distributed decryption of [Delta] - which results in everybody knowing Delta
        //// TODO: All parties except party 1 saves the random elemetes they picked
        //// TODO: Party one saves a1 b1 and c_1 + Delta

        //if is_chosen_party {
            //ci.val = ci.val - delta;
        //}
        //let ci = ci;

        //Self {
            //shares: (ai, bi, ci),
        //}
    //}
//}
