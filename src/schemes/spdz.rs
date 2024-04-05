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
 
use ff::PrimeField;

use derive_more::{Add, AddAssign, Sub, SubAssign};
use rand::{thread_rng, RngCore, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use tracing_subscriber::field::debug;
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

// TODO: consider also making this retun an option or result, such that it can return nothing if there are to few preprocessed values
pub async fn secret_mult<F>(
    s1: Share<F>, 
    s2: Share<F>, 
    context: &mut SpdzContext<F>, 
    network: &mut impl Broadcast 
) -> Share<F>
where F: PrimeField + serde::Serialize + serde::de::DeserializeOwned 
{
    // TODO: need to verify that there are atleast one priplets pair left, otherwise cast an error. 
    let triplet = context.preprocessed_values.triplets.pop().unwrap();
    let is_chosen_party = context.params.who_am_i == 0;
    let mac_key_share = context.params.mac_key_share;

    let e = s1 - triplet.a;
    let d = s2 - triplet.b;
    // here we need to make a broadcast to partially open the elements 
    let e = partial_opening_2(e.val, network).await;
    let d = partial_opening_2(d.val, network).await;
    (triplet.c + triplet.b*e + triplet.a*d).add_public(e*d, is_chosen_party, mac_key_share)
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

pub fn send_share<F: PrimeField>(val: F, rand_known_to_me: &mut RandomKnownToMe<F>, mac_key_share: F) -> (Share<F>, F) {
    // ToDo: Throw an error if there is no more elements. Then we need more preprocessing.
    let (rand_share, r) = rand_known_to_me.shares_and_vals.pop().unwrap(); 
    let correction = val - r;
    let share = rand_share.add_public(correction, true, mac_key_share);
    (share, correction)
}

// When resiving a share, the party resiving it needs to know who send it.
pub fn recive_share_from<F: PrimeField>(correction: F, rand_known_to_i: &mut RandomKnownToPi<F>, who:usize, mac_key_share: F) -> Share<F> {
    // ToDo: Throw an error if there is no more elements. Then we need more preprocessing.
    let rand_share = rand_known_to_i.shares[who].pop().unwrap(); 
    let share = rand_share.add_public(correction, false, mac_key_share);
    share
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


#[derive(Debug)]
struct SpdzParams<F: PrimeField> {
    mac_key_share: F,
    who_am_i: usize,
}

#[derive(Debug)]
pub struct SpdzContext<F: PrimeField> {
    opened_values: Vec<F>, 
    closed_values: Vec<Share<F>>,
    // dbgr supplier (det. random bit generator)
    params: SpdzParams<F>,
    preprocessed_values: preprocessing::PreprocessedValues<F>,
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


// TODO: Change to SPDZ 2 style like the rest 
    // TODO: Start by being able to check one element - this is done in the open_elm function. 
        // Just make a version that does not do the opening, but takes a partially opened elm.
    // TODO: Then check multiple elements 
pub async fn mac_check<Rng: SeedableRng + RngCore, F: PrimeField + Serialize + DeserializeOwned>(
    ctx: &mut SpdzContext<F>,
    cx: &mut impl Broadcast,
) -> Result<(), ()> {
    // TODO-later: More specific errors
    // This should all be done way nicer.
    let mut cointoss = CoinToss::new(thread_rng());
    let seed = Rng::Seed::default(); // I hate this
    let _coin: [u8; 32] = cointoss.toss(cx).await.unwrap();
    let mut rng = Rng::from_seed(seed);

    // This could probably be a lot nicer if opened and closed values were one list.
    // They probably should be since they should have the same length I think.
    let n = ctx.opened_values.len();
    let rs: Vec<_> = (0..n).map(|_| F::random(&mut rng)).collect();
    let a: F = ctx
        .opened_values
        .iter()
        .zip(rs.iter())
        .map(|(&b, r)| b * r)
        .sum();
    let gamma: F = ctx
        .closed_values
        .iter()
        .zip(rs.iter())
        .map(|(v, r)| v.mac * r)
        .sum();
    let delta = gamma - ctx.params.mac_key_share * a;

    let deltas = cx.symmetric_broadcast(delta).await.unwrap(); // (commitment)
    let delta_sum: F = deltas.iter().sum();

    if delta_sum.is_zero_vartime() {
        ctx.opened_values.clear();
        ctx.closed_values.clear();
        // great success!
        Ok(())
    } else {
        // bad! someone is corrupted
        Err(())
    }
}

#[cfg(test)]
mod test {

    use ff::Field;
    //use rayon::vec;
    //use tokio_util::context;

    use crate::{algebra::element::Element32, net, protocols::preprocessing::{self, RandomKnownToMe, RandomKnownToPi}};

    use super::*;

    // All these tests use bogus preprosessing 
    struct SecretValues<F> {
        mac_key: F,
        secret_shared_elements: Vec<F>,
    }
    // TODO: make a dealer function in the preprocessing module, that takes the 
    // setup
    fn dummie_prepross() -> (SpdzContext<Element32>, SpdzContext<Element32>, SecretValues<Element32>){
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);


        // Secret values
        type F = Element32;
        let mac_key = F::random(&mut rng);

        let r = F::random(&mut rng);
        let r_mac = r * mac_key;
        let s = F::random(&mut rng);
        let s_mac = s * mac_key;

        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c = a*b;

        let a_mac = a * mac_key;
        let b_mac = b * mac_key;
        let c_mac = c * mac_key;

        let secret_values = SecretValues{
            mac_key,
            secret_shared_elements: vec![r,s,a,a_mac,b,b_mac,c,c_mac],
        };

        // P1
        let mac_key_share = F::random(&mut rng);
        
        let r1 = F::random(&mut rng);
        let r_mac_1 = F::random(&mut rng);
        let r1_share = Share{val:r1, mac:r_mac_1};

        let s1 = F::random(&mut rng);
        let s_mac_1 = F::random(&mut rng);
        let s1_share = Share{val:s1, mac:s_mac_1};
        
        let a1 = F::random(&mut rng);
        let b1: Element32 = F::random(&mut rng);
        let c1 = F::random(&mut rng);

        let a_mac_1 = F::random(&mut rng);
        let b_mac_1 = F::random(&mut rng);
        let c_mac_1 = F::random(&mut rng);

        let a1_share = Share{val:a1, mac:a_mac_1};
        let b1_share = Share{val:b1, mac:b_mac_1};
        let c1_share = Share{val:c1, mac:c_mac_1};

        let a_triplet = preprocessing::make_multiplicationtriplet(a1_share, b1_share, c1_share);
        let triplets = vec![a_triplet];
        
        let rand_known_to_i = RandomKnownToPi{shares: vec![vec![r1_share],vec![s1_share]]};
        let rand_known_to_me= RandomKnownToMe{shares_and_vals: vec![(r1_share, r)]};
        let p1_preprosvals = preprocessing::PreprocessedValues{
            triplets,
            rand_known_to_i,
            rand_known_to_me,
        };
        let p1_context = SpdzContext{
            opened_values: vec![],
            closed_values: vec![],
            params: SpdzParams{mac_key_share, who_am_i: 0},
            //mac_key_share: mac_key_share_1,
            //who_am_i: 0,
            preprocessed_values: p1_preprosvals,
        };

        // P2
        let mac_key_share = mac_key - mac_key_share;

        let r2 = r - r1;
        let r_mac_2 = r_mac - r_mac_1;
        let r2_share = Share{val:r2, mac:r_mac_2};
        
        let s2 = s - s1;
        let s_mac_2 = s_mac - s_mac_1;
        let s2_share = Share{val:s2, mac:s_mac_2};

        let a2 = a-a1;
        let b2: Element32 = b-b1;
        let c2 = c-c1;

        let a_mac_2 = a_mac - a_mac_1;
        let b_mac_2 = b_mac - b_mac_1;
        let c_mac_2 = c_mac - c_mac_1;

        let a2_share = Share{val:a2, mac:a_mac_2};
        let b2_share = Share{val:b2, mac:b_mac_2};
        let c2_share = Share{val:c2, mac:c_mac_2};

        let a_triplet = preprocessing::make_multiplicationtriplet(a2_share, b2_share, c2_share);
        let triplets = vec![a_triplet];
        let rand_known_to_i = RandomKnownToPi{shares: vec![vec![r2_share], vec![s2_share]]};
        let rand_known_to_me = RandomKnownToMe{shares_and_vals: vec![(s2_share, s)]}; 
        let p2_preprosvals = preprocessing::PreprocessedValues{
            triplets,
            rand_known_to_i,
            rand_known_to_me,
        };
        let p2_context = SpdzContext{
            opened_values: vec![],
            closed_values: vec![],
            params: SpdzParams{mac_key_share, who_am_i: 1},
            preprocessed_values: p2_preprosvals,
        };
        (p1_context, p2_context, secret_values)
    }
    #[test]
    fn test_dummi_prepros(){
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        let a = p1_context.preprocessed_values.triplets[0].a + p2_context.preprocessed_values.triplets[0].a;
        let b = p1_context.preprocessed_values.triplets[0].b + p2_context.preprocessed_values.triplets[0].b;
        let c = p1_context.preprocessed_values.triplets[0].c + p2_context.preprocessed_values.triplets[0].c;
        assert!(a.val * b.val == c.val);
        assert!(a.mac == a.val*secret_values.mac_key);
        assert!(b.mac == b.val*secret_values.mac_key);
        assert!(c.mac == c.val*secret_values.mac_key);
    }
    #[test]
    fn test_sharing() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share);
        
        let elm2_1 = recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share);
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);
    }
    #[test]
    fn test_partial_opening() {
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        //let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = send_share(elm1, &mut (p1_context.preprocessed_values.rand_known_to_me), p1_context.params.mac_key_share);
        
        //let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut (p2_context.preprocessed_values.rand_known_to_i), 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(elm2, &mut (p2_context.preprocessed_values.rand_known_to_me), p2_context.params.mac_key_share);
        
        let elm2_1 = recive_share_from(correction, &mut p1_context.preprocessed_values.rand_known_to_i, 1, p1_context.params.mac_key_share);
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2*secret_values.mac_key);

        // P1 opens elm1
        // ToDo: Deside whether P1 broadcasts the values from all shares to all parties or all parties broadcast their value from their share of the element.
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
        //let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = send_share(elm1, &mut (p1_context.preprocessed_values.rand_known_to_me), p1_context.params.mac_key_share);
        
        //let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut (p2_context.preprocessed_values.rand_known_to_i), 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(elm2, &mut (p2_context.preprocessed_values.rand_known_to_me), p2_context.params.mac_key_share);
        
        let elm2_1 = recive_share_from(correction, &mut p1_context.preprocessed_values.rand_known_to_i, 1, p1_context.params.mac_key_share);
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
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share);
        
        let elm2_1 = recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share);
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
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share);
        
        let elm2_1 = recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share);
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
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
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
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let p2_prepros = &mut p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(elm2, &mut (p2_prepros.rand_known_to_me), p2_context.params.mac_key_share);
        
        let elm2_1 = recive_share_from(correction, &mut p1_prepros.rand_known_to_i, 1, p1_context.params.mac_key_share);
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
            let res_share = secret_mult(s1, s2, &mut context, &mut network).await;
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
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
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
        let (elm1_1, correction) = send_share(elm1, &mut (p1_prepros.rand_known_to_me), p1_context.params.mac_key_share);
        
        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = recive_share_from(correction, &mut p2_prepros.rand_known_to_i, 0, p2_context.params.mac_key_share);
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1*secret_values.mac_key);
        
        // Adding with pub_constant 
        let elm3_1 = elm1_1.sub_public(pub_constant, 0==p1_context.params.who_am_i, p1_context.params.mac_key_share);

        let elm3_2 = elm1_2.sub_public(pub_constant, 0==p2_context.params.who_am_i, p2_context.params.mac_key_share);
        
        assert!(elm1 - pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1-pub_constant)*secret_values.mac_key);
    }
    // TODO: test checking

}
