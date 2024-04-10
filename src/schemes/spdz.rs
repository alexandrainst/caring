//! (Some versirn of) SPDZ
//!
//! The idea is to implement the bare-bones version of spdz, probably a more recent one.
//! We will probably need some homomorphic encryption or oblivious transfer to enable this.
//!
//! Further more stands the issue of sharing the input in a nice way, since a lot of
//! preprocessing is needed in spdz beforehand.
//!
//! The spdz(2k) scheme is actually not that difficult in the online phase,
//! however it requres some heavy machinery in the offline phase.
//!
 

// TODO: make costum errors.
use ff::PrimeField;

use derive_more::{Add, AddAssign, Sub, SubAssign};
use rand::{thread_rng, RngCore, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use tracing_subscriber::field::debug;
use std::io;
use crate::{net::{agency::Broadcast, network::{self, InMemoryNetwork}}, protocols::{cointoss::CoinToss, preprocessing::{self, RandomKnownToMe, RandomKnownToPi}, commitments}};

// Should we allow Field or use PrimeField?
#[derive(Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign, serde::Serialize, serde::Deserialize)]
pub struct Share<F: PrimeField> {
    // This field is nice and I like it
    pub val: F,
    // This field is scary and I don't know how it should be handled
    pub mac: F,
}

impl<F: PrimeField> Share<F> {
    pub fn add_public(self, val: F, is_chosen_party: bool, mac_key_share: F) -> Self {
        let val_val = if is_chosen_party { val } else { F::ZERO };
        Share {
            val: self.val + val_val,
            mac: self.mac + val * mac_key_share,
        }
    }

    pub fn sub_public(self, val: F, chosen_one: bool, mac_key_share: F) -> Self {
        let val_val = if chosen_one { val } else { F::ZERO };
        Share {
            val: self.val - val_val,
            mac: self.mac - val * mac_key_share,
        }
    }
}

pub async fn secret_mult<F>(
    s1: Share<F>, 
    s2: Share<F>, 
    context: &mut SpdzContext<F>, 
    network: &mut impl Broadcast 
) -> Result<Share<F>,()>
where F: PrimeField + serde::Serialize + serde::de::DeserializeOwned 
{
    // TODO: Do something meaningfull with the errors
    let triplet_option = context.preprocessed_values.triplets.pop();
    let triplet = match triplet_option {
        Some(t) => t,
        None => {
            return Err(());
        }
    };
    let is_chosen_party = context.params.who_am_i == 0;
    let mac_key_share = context.params.mac_key_share;

    let e = s1 - triplet.a;
    let d = s2 - triplet.b;
    // here we need to make a broadcast to partially open the elements 
    let e = partial_opening_2(e.val, network).await;
    let d = partial_opening_2(d.val, network).await;
    let res = (triplet.c + triplet.b*e + triplet.a*d).add_public(e*d, is_chosen_party, mac_key_share);
    Ok(res)
}

// Mutliplication between a share and a public value
// This operation is symmetric
impl<F: PrimeField> std::ops::Mul<F> for Share<F> {
    type Output = Share<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Share {
            val: self.val * rhs,
            mac: self.mac * rhs,
        }
    }
}
// The same can not be done for addition, unless we can give it access to the context some how. 
// TODO: verify with Mikkel that that is not a possiblility.


// TODO: Write share and resive_share_from together to one function 
    // Need to find out How to broadcast frome one specific party to all others - and how to await for that.
//pub async fn share<F: PrimeField>(op_val: Option<F>, rand_known_to_me: &mut Vec<(Share<F>, F)>, rand_known_to_i: &mut Vec<Vec<Share<F>>>, who_am_i: usize, who_is_sending: usize) -> Share<F> {
    //let is_chosen_one = who_am_i == who_is_sending;
    //if is_chosen_one{
        //let val = op_val.unwrap(); // Not a pretty solution
        //let (share, correction) = send_share(val, rand_known_to_me);
        //share
        //// TODO: broadcast correction value
    //} else {
        //// Resive correction value
        //recive_share_from(correction, rand_known_to_i, who_is_sending)
        
    //}
//}

pub fn send_share<F: PrimeField>(
    val: F, 
    rand_known_to_me: &mut RandomKnownToMe<F>, 
    mac_key_share: F
) -> Result<(Share<F>, F),()> {
    // ToDo: Throw a nice error if there is no more elements. Then we need more preprocessing.
    let (rand_share, r) = match rand_known_to_me.shares_and_vals.pop(){
        Some((r_share, r_elm)) => {
            println!("send share went fine");
            (r_share, r_elm)},
        None => {
            println!("The sharing is failing at send");
            return Err(())},
    };
    let correction = val - r;
    let share = rand_share.add_public(correction, true, mac_key_share);
    Ok((share, correction))
}

pub fn send_share_v2<F: PrimeField>(
    val: F, 
    rand_known_to_me: &mut RandomKnownToMe<F>, 
    rand_known_to_i: &mut RandomKnownToPi<F>,
    who_am_i: usize,
    mac_key_share: F
) -> Result<(Share<F>, F),()> {
    // ToDo: Throw a nice error if there is no more elements. Then we need more preprocessing.
    //let (rand_share, r) = match rand_known_to_me.shares_and_vals.pop(){
        //Some((r_share, r_elm)) => {
            //println!("send share went fine");
            //(r_share, r_elm)},
        //None => {
            //println!("The sharing is failing at send");
            //return Err(())},
    //};
    let r = match rand_known_to_me.shares_and_vals.pop(){
        Some((_, r)) => {
            println!("send share went fine");
            r},
        None => {
            println!("The sharing is failing at send");
            return Err(())},
    };
    let (r_share) = match rand_known_to_i.shares[who_am_i].pop(){
        Some(r_share) => {
            println!("send share went fine");
            r_share},
        None => {
            println!("The sharing is failing at send");
            return Err(())},
    };
    let correction = val - r;
    let share = r_share.add_public(correction, true, mac_key_share);
    Ok((share, correction))
}

// When resiving a share, the party resiving it needs to know who send it.
pub fn recive_share_from<F: PrimeField>(
    correction: F, 
    rand_known_to_i: &mut RandomKnownToPi<F>, 
    who:usize, 
    mac_key_share: F
) -> Result<Share<F>,()> {
    // ToDo: Throw an error if there is no more elements. Then we need more preprocessing.
    let rand_share = match rand_known_to_i.shares[who].pop(){
        Some(s) => {
            println!("resive share went fine");
            s},
        None => {
            println!("The sharing is failing at resive");
            return Err(())}
    }; 
    let share = rand_share.add_public(correction, false, mac_key_share);
    Ok(share)
}

// The parties need to send the elemets to each other, and then use this partiel opening to combine them.
pub fn partial_opening<F: PrimeField>(context: &mut SpdzContext<F>,candidate_vals: Vec<F>) -> F{
    let s: F = candidate_vals.iter().sum();
    context.opened_values.push(s);
    s
}
// partiel opening that handles the broadcasting but not the pushing to opended values. We will get back to that if/when it becomes nessesary.
pub async fn partial_opening_2<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(candidate_val: F, network: &mut impl Broadcast) -> F{
    let candidate_vals = network.symmetric_broadcast(candidate_val).await.unwrap();
    candidate_vals.iter().sum()
}


// Parms fields are public, so they can be used from preprocessing, 
// but preprocessing is specified to SPDZ anyway and should therefor probably be a module here instead.
// TODO: Therefor move preprocessing to be a module under SPDZ and make less stuff public :) 
#[derive(Debug)]
pub struct SpdzParams<F: PrimeField> {
    pub mac_key_share: F,
    pub who_am_i: usize,
}

// Context fields are public, so they can be used from preprocessing, 
// but preprocessing is specified to SPDZ anyway and should therefor probably be a module here instead.
// TODO: Therefor move preprocessing to be a module under SPDZ and make less stuff public :) 
#[derive(Debug)]
pub struct SpdzContext<F: PrimeField> {
    pub opened_values: Vec<F>, 
    pub closed_values: Vec<Share<F>>,
    // dbgr supplier (det. random bit generator)
    pub params: SpdzParams<F>,
    pub preprocessed_values: preprocessing::PreprocessedValues<F>,
}


//TODO: change this to return an option instead, such that it does not return anything if it fails.
pub async fn open_elm<F>(
    share_to_open: Share<F>, 
    network: &mut impl Broadcast, 
    mac_key_share: &F
) -> F
where F: PrimeField + serde::Serialize + serde::de::DeserializeOwned + std::convert::Into<u64> 
{
    let opened_val = partial_opening_2(share_to_open.val, network).await;
    let this_went_well = check_one_elment(opened_val, network, &share_to_open.mac, mac_key_share).await;
    if this_went_well {
        println!("yes this went well");
        opened_val
    } else {
        // Here we need to cast some err.
        println!("The check did not go though!");
        share_to_open.val
    }

}

pub async fn check_one_elment<F>(
    val_to_check: F, 
    network: &mut impl Broadcast, 
    share_of_mac_to_val: &F, 
    mac_key_share: &F,
) -> bool
where F: PrimeField + serde::Serialize + serde::de::DeserializeOwned + std::convert::Into<u64>
{
    let d = *mac_key_share * val_to_check - share_of_mac_to_val;
    let (c,s) = commitments::commit(d);
    let cs = network.symmetric_broadcast((c,s)).await.unwrap();
    let ds = network.symmetric_broadcast(d).await.unwrap();
    let dcs = ds.iter().zip(cs.iter());
    let mut this_went_well = true;
    for (d,(c,s)) in dcs{
        let t = commitments::verify_commit(d, c, s);
        if !t {
            this_went_well = false;
        }
    }

    let ds_sum:F = ds.iter().sum();
    (ds_sum == 0.into()) && this_went_well
}


#[cfg(test)]
mod test {

    use ff::Field;
    //use rayon::vec;
    //use tokio_util::context;

    use crate::{algebra::element::Element32, net, protocols::preprocessing::{self, RandomKnownToMe, RandomKnownToPi, SecretValues}};

    use super::*;

    // All these tests use dealer preprosessing 
    // setup
    fn dummie_prepross() -> (SpdzContext<Element32>, SpdzContext<Element32>, preprocessing::SecretValues<Element32>){
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);
        let known_to_each = vec![1,1];
        let number_of_triplets = 1;
        let number_of_parties = 2;
        let (mut contexts, secret_values) = preprocessing::dealer_prepross(rng, known_to_each, number_of_triplets, number_of_parties);
        let c1 = contexts.pop().unwrap();
        let c0 = contexts.pop().unwrap();
        (c0,c1, secret_values)
    }
    #[test]
    fn test_dummi_prepros(){
        let (p1_context, p2_context, secret_values) = dummie_prepross();

        // unpacking
        let p1_params = p1_context.params;
        let p2_params = p2_context.params;
        let p1_preprocessed = p1_context.preprocessed_values;
        let p2_preprocessed = p2_context.preprocessed_values;
        let p1_known_to_pi = p1_preprocessed.rand_known_to_i.shares;
        let p2_known_to_pi = p2_preprocessed.rand_known_to_i.shares;
        let p1_known_to_me = p1_preprocessed.rand_known_to_me.shares_and_vals;
        let p2_known_to_me = p2_preprocessed.rand_known_to_me.shares_and_vals;
        let p1_triplets = p1_preprocessed.triplets;
        let p2_triplets = p2_preprocessed.triplets;
        let mac = secret_values.mac_key;
        let a = p1_triplets[0].a + p2_triplets[0].a;
        let b = p1_triplets[0].b + p2_triplets[0].b;
        let c = p1_triplets[0].c + p2_triplets[0].c;
        let p1_who_am_i = p1_params.who_am_i;
        let p2_who_am_i = p2_params.who_am_i;
        
        // testing
        assert!(a.val * b.val == c.val);
        assert!(a.mac == a.val*secret_values.mac_key);
        assert!(b.mac == b.val*secret_values.mac_key);
        assert!(c.mac == c.val*secret_values.mac_key);
        assert!(p1_who_am_i == 0);
        assert!(p2_who_am_i == 1);
        let r_val = p1_known_to_pi[0][0].val+p2_known_to_pi[0][0].val;
        let r_mac = p1_known_to_pi[0][0].mac+p2_known_to_pi[0][0].mac;
        let r2: Element32 = p1_known_to_me[0].1;
        assert!(r_val == r2);
        assert!(r_mac == r2*mac);
        assert!(p1_known_to_me[0].0.val == p1_known_to_pi[0][0].val);
        assert!(p1_known_to_me[0].0.mac == p1_known_to_pi[0][0].mac);
        let s_val = p1_known_to_pi[1][0].val+p2_known_to_pi[1][0].val;
        let s_mac = p1_known_to_pi[1][0].mac+p2_known_to_pi[1][0].mac;
        let s2 = p2_known_to_me[0].1;
        assert!(s_val == s2);
        assert!(s_mac == s2*mac);
        assert!(p2_known_to_me[0].0.val == p2_known_to_pi[1][0].val);
        assert!(p2_known_to_me[0].0.mac == p2_known_to_pi[1][0].mac);
    }
    #[test]
    fn test_dealer(){
        let rng = rand::rngs::mock::StepRng::new(42, 7);
        let known_to_each = vec![1,2];
        let number_of_triplets = 2;
        let number_of_parties = 2;
        let (mut contexts, secret_values) = preprocessing::dealer_prepross(rng, known_to_each, number_of_triplets, number_of_parties);

        // unpacking 
        let mac:Element32 = secret_values.mac_key;
        let p2_context = contexts.pop().unwrap(); 
        let p1_context = contexts.pop().unwrap();
        let p1_params = p1_context.params;
        let p2_params = p2_context.params;
        let p1_preprocessed = p1_context.preprocessed_values;
        let p2_preprocessed = p2_context.preprocessed_values;
        let p1_known_to_pi = p1_preprocessed.rand_known_to_i.shares;
        let p2_known_to_pi = p2_preprocessed.rand_known_to_i.shares;
        let p1_known_to_me = p1_preprocessed.rand_known_to_me.shares_and_vals;
        let p2_known_to_me = p2_preprocessed.rand_known_to_me.shares_and_vals;
        let p1_triplets = p1_preprocessed.triplets;
        let p2_triplets = p2_preprocessed.triplets;

        // Testing
        assert!(p1_params.mac_key_share + p2_params.mac_key_share == mac);

        let r = p1_known_to_pi[0][0]+p2_known_to_pi[0][0];
        let r2 = p1_known_to_me[0].1;
        assert!(r.val == r2);
        assert!(r.mac == r2*mac);

        let s = p1_known_to_pi[1][0]+p2_known_to_pi[1][0];
        let s2 = p2_known_to_me[0].1;
        assert!(s.val == s2);
        assert!(s.mac == s2*mac);

        let s1 = p1_known_to_pi[1][1]+p2_known_to_pi[1][1];
        let s3 = p2_known_to_me[1].1;
        assert!(s1.val == s3);
        assert!(s1.mac == s3*mac);

        let a1 = p1_triplets[0].a + p2_triplets[0].a;
        let b1 = p1_triplets[0].b + p2_triplets[0].b;
        let c1 = p1_triplets[0].c + p2_triplets[0].c;

        assert!(a1.val * b1.val == c1.val);
        assert!(a1.val * mac == a1.mac);
        assert!(b1.val * mac == b1.mac);
        assert!(c1.val * mac == c1.mac);

        let a2 = p1_triplets[1].a + p2_triplets[1].a;
        let b2 = p1_triplets[1].b + p2_triplets[1].b;
        let c2 = p1_triplets[1].c + p2_triplets[1].c;

        assert!(a2.val * b2.val == c2.val);
        assert!(a2.val * mac == a2.mac);
        assert!(b2.val * mac == b2.mac);
        assert!(c2.val * mac == c2.mac);
    }
    #[test]
    fn test_sharing() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        assert!(p1_context.params.who_am_i == 0);
        assert!(p2_context.params.who_am_i == 1);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share_v2(elm1, &mut (p1_prepros.rand_known_to_me), &mut p1_prepros.rand_known_to_i, p1_context.params.who_am_i, p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!((elm1_1.mac + elm1_2.mac) == (elm1*secret_values.mac_key));
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm2_1 = match recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);
    }
    #[test]
    fn test_partial_opening() {
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_context.preprocessed_values.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm1_2 = match recive_share_from(correction, &mut (p2_context.preprocessed_values.rand_known_to_i), 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(elm2, &mut (p2_context.preprocessed_values.rand_known_to_me), p2_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm2_1 = match recive_share_from(correction, &mut p1_context.preprocessed_values.rand_known_to_i, 1, p1_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);

        // P1 opens elm1
        let val1_v2 = partial_opening(&mut p1_context, vec![elm1_1.val, elm1_2.val]); 
        partial_opening(&mut p2_context, vec![elm1_1.val, elm1_2.val]);
        assert!(val1_v2 == elm1);

        // P2 opens elm2
        let val2_v2 = partial_opening(&mut p2_context, vec![elm2_1.val, elm2_2.val]); 
        partial_opening(&mut p1_context, vec![elm2_1.val, elm2_2.val]);
        assert!(val2_v2 == elm2);

        assert!(p1_context.opened_values == p2_context.opened_values);
    }

    #[tokio::test]
    async fn test_partial_opening_2() {
        use crate::net::network::InMemoryNetwork;
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_context.preprocessed_values.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        //let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut (p2_context.preprocessed_values.rand_known_to_i), 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(elm2, &mut (p2_context.preprocessed_values.rand_known_to_me), p2_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm2_1 = match recive_share_from(correction, &mut p1_context.preprocessed_values.rand_known_to_i, 1, p1_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);

        // Parties opens elm1
        async fn do_mpc(
            network: InMemoryNetwork,
             elm: F,
             val1: F
            ){
            let mut network = network;
            let val1_guess = partial_opening_2( elm, &mut network).await;
            assert!(val1_guess == val1);
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(2); //asuming two players
        let elm1_v = vec![elm1_1, elm1_2];
        let mut i = 0;
        for network in cluster {
            taskset.spawn(do_mpc(network, elm1_v[i].val, elm1));
            i += 1;
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }

    }

    #[test]
    fn test_addition() { 
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm2_1 = match recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);

        // Adding ss-elements 
        let elm3_1 = elm1_1 + elm2_1;

        let elm3_2 = elm1_2 + elm2_2;
        
        assert!(elm1 + elm2 == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1+elm2)*secret_values.mac_key);
    }
    
    #[test]
    fn test_subtracting() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm2_1 = match recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);

        // Subtracting ss-elements 
        let elm3_1 = elm1_1 - elm2_1;

        let elm3_2 = elm1_2 - elm2_2;
        
        assert!(elm1 - elm2 == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1-elm2)*secret_values.mac_key);
    }

    #[test]
    fn test_multiplication_with_pub_constant() { 
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // Multiplying with pub_constant 
        let elm3_1 = elm1_1 * pub_constant;

        let elm3_2 = elm1_2 * pub_constant;
        
        assert!(elm1 * pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1*pub_constant)*secret_values.mac_key);
    }
    #[tokio::test]
    async fn test_secret_shared_multipllication() {
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let p1_prepros = &mut p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let p2_prepros = &mut p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let elm2_1 = match recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);

        let expected_res = Share{val: elm1*elm2, mac: elm1*elm2*secret_values.mac_key};
        // Multiplicate
        async fn do_mpc(
            network: InMemoryNetwork,
             s1: Share<F>,
             s2: Share<F>,
             context: SpdzContext<F>,
             expected_res: Share<F>,
            ){
            let mut context = context;
            let mut network = network;
            let res_share_result = secret_mult(s1, s2, &mut context, &mut network).await;
            let res_share = match res_share_result {
                Ok(share) => share,
                Err(error) => {
                    assert!(false); // TODO: do we want it to panic? - we do want it to run the rest of the tests, even when this one fails.
                    Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
                }
            };
            let res = partial_opening_2(res_share.val, &mut network).await;
            assert!(expected_res.val == res);

            let res = open_elm(res_share, &mut network, &context.params.mac_key_share).await;
            assert!(expected_res.val == res);
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(2); //asuming two players
        let elm1_v = vec![elm1_1, elm1_2];
        let elm2_v = vec![elm2_1, elm2_2];
        let mut context = vec![p1_context, p2_context];
        let mut i = 0;
        for network in cluster {
            let context_here = context.pop().unwrap();
            taskset.spawn(do_mpc(network, elm1_v[i], elm2_v[i], context_here, expected_res));
            i += 1;
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
        
    }
    #[test]
    fn test_add_with_public_constant() { 
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // Adding with pub_constant 
        let elm3_1 = elm1_1.add_public(pub_constant, 0==p1_context.params.who_am_i, p1_context.params.mac_key_share);

        let elm3_2 = elm1_2.add_public(pub_constant, 0==p2_context.params.who_am_i, p2_context.params.mac_key_share);
        
        assert!(elm1 + pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1+pub_constant)*secret_values.mac_key);
    }
    #[test]
    fn test_sub_with_public_constant() { 
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share){
            Ok((e,c)) => (e,c),
            Err(e) => {
                assert!(false);
                (Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}, F::from_u128(0u128))
            }
        };
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share){
            Ok(s) => s,
            Err(e) => {
                assert!(false);
                Share{val:F::from_u128(0u128), mac:F::from_u128(0u128)}
            }
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // Adding with pub_constant 
        let elm3_1 = elm1_1.sub_public(pub_constant, 0==p1_context.params.who_am_i, p1_context.params.mac_key_share);

        let elm3_2 = elm1_2.sub_public(pub_constant, 0==p2_context.params.who_am_i, p2_context.params.mac_key_share);
        
        assert!(elm1 - pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1-pub_constant)*secret_values.mac_key);
    }
    // TODO: test checking
    // TODO: test errors

}
