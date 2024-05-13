//! (Some version of) SPDZ
//! This SPDZ implementation is primarely based on the following lecture by Ivan Damgård:
//! (part one:) https://www.youtube.com/watch?v=N80DV3Brds0 (and part two:) https://www.youtube.com/watch?v=Ce45hp24b2E
//!
//! We will need some homomorphic encryption or oblivious transfer to enable preprocessing.
//! But for now that is handled by a dealer.
//!


// TODO: make costum errors.
use crate::{net::Communicate, schemes::interactive::InteractiveShared};
use ff::PrimeField;
use async_trait::async_trait;
use rand::RngCore;
use crate::{
    net::agency::Broadcast,
    protocols::commitments::{commit, verify_commit, commit_many, verify_many},
};
use derive_more::{Add, AddAssign, Sub, SubAssign};

use self::preprocessing::ForSharing;

pub mod preprocessing;
use std::path::Path;

// Should we allow Field or use PrimeField?
#[derive(
    Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign, serde::Serialize, serde::Deserialize,
)]
pub struct Share<F: PrimeField> {
    val: F,
    mac: F,
}

impl<F: PrimeField> Share<F> {
    pub fn add_public(self, val: F, is_chosen_party: bool, params: &SpdzParams<F>) -> Self {
        let mac_key_share = params.mac_key_share;
        let val_val = if is_chosen_party { val } else { F::ZERO };
        Share {
            val: self.val + val_val,
            mac: self.mac + val * mac_key_share,
        }
    }

    pub fn sub_public(self, val: F, chosen_one: bool, params: &SpdzParams<F>) -> Self {
        let mac_key_share = params.mac_key_share;
        let val_val = if chosen_one { val } else { F::ZERO };
        Share {
            val: self.val - val_val,
            mac: self.mac - val * mac_key_share,
        }
    }
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
// TODO: Do we need the hole context? - In any case, we do not need the hole context to be mutable.
pub async fn secret_mult<F>(
    s1: Share<F>,
    s2: Share<F>,
    triplets: &mut preprocessing::Triplets<F>,
    params: &SpdzParams<F>,
    opened_values: &mut Vec<F>,
    network: &mut impl Broadcast,
) -> Result<Share<F>, ()>
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    // TODO: Could be meaningfull to have a custom error, for when there is not enough preprocessed values.
    // And then return an error instead of panicing
    let triplet = triplets
        .get_triplet();
    let is_chosen_party = params.who_am_i == 0;
    //let mac_key_share = params.mac_key_share;

    let e = s1 - triplet.a;
    let d = s2 - triplet.b;
    let e = partial_opening(
        &e,
        &params,
        network,
        opened_values,
    )
    .await;
    let d = partial_opening(
        &d,
        &params,
        network,
        opened_values,
    )
    .await;
    let res = (triplet.c + triplet.b * e + triplet.a * d).add_public(
        e * d,
        is_chosen_party,
        params,
    );
    Ok(res)
}

impl<'ctx, F> InteractiveShared<'ctx> for Share<F>
where F: PrimeField + serde::Serialize + serde::de::DeserializeOwned + std::convert::Into<u64>,
{
    type Context = &'ctx mut SpdzContext<F>;
    type Value = F;
    type Error = ();

    async fn share(
            ctx: Self::Context,
            secret: Self::Value,
            rng: impl RngCore + Send,
            coms: impl Communicate,
        ) -> Result<Self, ()> {
        share(
            Some(secret), 
            &mut ctx.preprocessed_values.for_sharing,
            todo!("who_is_sending"), 
            &ctx.params, 
            &mut coms
        ).await
    }


    async fn symmetric_share(
        ctx: Self::Context,
        secret: Self::Value,
        rng: impl RngCore + Send,
        coms: impl Communicate,
    ) -> Result<Vec<Self>, Self::Error> {
        todo!()
    }
    
    async fn receive_share(
            ctx: Self::Context,
            coms: impl Communicate,
            from: usize,
        ) -> Result<Self, ()> {
        share(
            None, 
            &mut ctx.preprocessed_values.for_sharing, 
            todo!("who_is_sending"), 
            &ctx.params, 
            &mut coms,
        ).await
    }

    // This might not be the propper way to do recombine - it depends on what exanctly recombine is suppose to mean :)
    async fn recombine(ctx: Self::Context, share: Self, mut network: impl Communicate) -> Result<F, ()> {
        Ok(open_res(share, &mut network, &ctx.params, todo!("opened values")).await)
    }
}

// For now just the most naive impl.
// TODO fix this
pub async fn share_many<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    op_vals: Vec<Option<F>>, 
    for_sharing: &mut ForSharing<F>,
    params: &SpdzParams<F>,
    who_is_sending: usize,
    network: &mut impl Broadcast,
) -> Vec<Share<F>> {
    //let mut res = op_vals.iter().map(|&op_val| async{share(op_val, for_sharing, who_is_sending, params, network).await.expect("...")}).collect()
    let mut res = vec![];
    for op_val in op_vals {
        res.push(share(op_val, for_sharing, who_is_sending, params, network).await.expect("something went wrong in sharing"));
    }
    res
}

pub async fn share<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    op_val: Option<F>,
    for_sharing: &mut ForSharing<F>,
    who_is_sending: usize,
    params: &SpdzParams<F>,
    network: &mut impl Broadcast,
) -> Result<Share<F>, ()> {
    let is_chosen_one = params.who_am_i == who_is_sending;
    if is_chosen_one {
        // TODO: Consider returning an error instead.
        let val = op_val.expect("The sender needs to enter the value to be send");
        let res = send_share(
            val,
            for_sharing,
            params,
        );
        let (share, correction) = match res {
            Ok((s, c)) => (s, c),
            Err(e) => return Err(e),
        };
        // TODO: return the error instead
        network
            .broadcast(&correction)
            .await
            .expect("Broadcasting went wrong");
        Ok(share)
    } else {
        let correction = network
            .recv_from(who_is_sending)
            .await
            .expect("all resivers should resive the correction");
        receive_share_from(
            correction,
            for_sharing,
            who_is_sending,
            params,
        )
    }
}
fn send_share<F: PrimeField>(
    val: F,
    for_sharing: &mut ForSharing<F>,
    params: &SpdzParams<F>, 
) -> Result<(Share<F>, F), ()> {
    let r = for_sharing.rand_known_to_me
        .vals
        .pop()
        .expect("To few preprocessed values");
    // TODO: return an error - preferably a costum not enough preprocessing error - lack of shared random elementsknown to party who_am_i
    let r_share = match for_sharing.rand_known_to_i.shares[params.who_am_i].pop() {
        Some(r_share) => r_share,
        None => {
            return Err(());
        }
    };
    let correction = val - r;
    let share = r_share.add_public(correction, params.who_am_i == 0, params);
    Ok((share, correction))
}

// When receiving a share, the party receiving it needs to know who send it.
fn receive_share_from<F: PrimeField>(
    correction: F,
    for_sharing: &mut ForSharing<F>,
    who_is_sending: usize,
    params: &SpdzParams<F>,
) -> Result<Share<F>, ()> {
    // ToDo: Throw an error if there is no more elements. Then we need more preprocessing. - see todo in send_share function
    let rand_share = match for_sharing.rand_known_to_i.shares[who_is_sending].pop() {
        Some(s) => s,
        None => {
            return Err(()); // TODO: return costum error, not enough preprocessing
        }
    };
    let share = rand_share.add_public(correction, params.who_am_i == 0, params);
    Ok(share)
}

// partiel opening that handles the broadcasting and pushing to opended values. All that is partially opened needs to be checked later.
async fn partial_opening<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    candidate_share: &Share<F>,
    params: &SpdzParams<F>,
    network: &mut impl Broadcast,
    partially_opened_vals: &mut Vec<F>,
) -> F {
    let candidate_vals = network.symmetric_broadcast(candidate_share.val).await.unwrap();
    let candidate_val = candidate_vals.iter().sum();
    partially_opened_vals.push(candidate_val * params.mac_key_share - candidate_share.mac);
    candidate_val
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpdzParams<F: PrimeField> {
    mac_key_share: F,
    who_am_i: usize,
}

// The SPDZ context needs to be public atleast to some degree, as it is needed for many operations that we would like to call publicly.
// If we do not want the context to be public, we probably need some getter functions - and some alter functions. (TODO)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpdzContext<F: PrimeField> {
    opened_values: Vec<F>,
    params: SpdzParams<F>,
    preprocessed_values: preprocessing::PreprocessedValues<F>,
}

// TODO: return an option or a result instead. - result probably makes most sense, but then we might want the costom errors first
pub async fn open_res<F>(
    share_to_open: Share<F>,
    network: &mut impl Broadcast,
    params: &SpdzParams<F>,
    opened_values: &Vec<F>,
) -> F
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned + std::convert::Into<u64>,
{   if !opened_values.is_empty() {
        panic!("dont open if there are unchecked open values") // TODO rerun error instead 
        // TODO: consider just calling check all - needs a random element though - either a generator or an element. 
        //check_all_d(opened_values, network, random_element)
    }
    // TODO: it might be meaningfull to verify that open_values are empty - and cast an error otherwise. 
    // As one are not allowed to open the result if not all partially opened values have been checked. 
    let opened_shares = network
        .symmetric_broadcast(share_to_open.val)
        .await
        .unwrap();
    let opened_val: F = opened_shares.iter().sum();
    let d = opened_val * params.mac_key_share - share_to_open.mac;
    let this_went_well = check_one_d(d, network).await;
    if this_went_well {
        opened_val
    } else {
        panic!("The check did not go though");
    }
}

// An element and its mac are accepted, if the sum of the corresponding d from each party is zero.
async fn check_one_d<F>(d: F, network: &mut impl Broadcast) -> bool
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned + std::convert::Into<u64>,
{
    let (c, s) = commit(&d);
    let cs = network.symmetric_broadcast((c, s)).await.unwrap();
    let ds = network.symmetric_broadcast(d).await.unwrap();
    let dcs = ds.iter().zip(cs.iter());
    let verify_commitments = dcs.fold(true, |acc, (d, (c, s))| verify_commit(d, c, s) && acc);
    let ds_sum: F = ds.iter().sum();
    (ds_sum == 0.into()) && verify_commitments
}

// An element is accepted, if the sum of the corresponding d from each party is zero.
// To test many elements at a time, the sums are made as a simle linear combination.
// In order to minimize broadcasts we use only one random element, which is then taken to the power of 1,2,...
// Now the random element is commited to and bradcasted together with the d's, instead of after. 
// - This is not problematic as the d's can't be altered after they are committed to, so they still can't depend on the random element
pub async fn check_all_d<F>(
    partially_opened_vals: &mut Vec<F>,
    network: &mut impl Broadcast,
    random_element: F,
) -> bool
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned + std::convert::Into<u64>,
{
    // TODO: make nice.
    let number_of_ds = partially_opened_vals.len();
    let mut partially_opened_vals_copy = partially_opened_vals.to_vec();
    let new_vec = &mut vec![];
    new_vec.append(partially_opened_vals);
    new_vec.append(&mut vec![random_element]);
    partially_opened_vals_copy.append(&mut vec![random_element]);
    let (c, s) = commit_many(new_vec);
    let cs = network.symmetric_broadcast((c, s)).await.unwrap();
    let mut dss = network
        .symmetric_broadcast(partially_opened_vals_copy)
        .await
        .unwrap();
    let csdss = cs.iter().zip(dss.iter());
    let this_went_well = csdss.fold(true, |acc, ((c,s),ds)| acc && verify_many(ds, c, s));
    
    // From each of the vectors we resived from the other parties, we pop of the random element in the end and use it to construct a shared random element.
    let r_elm = (&mut dss).iter_mut().fold(F::from_u128(0), |acc, ds| acc + ds.pop().expect("there is atleast one elm"));
    // From one shared random element we construct a number of elements that are hard to predict:
    let r_elms: Vec<F> = (1..number_of_ds+1).map(|i| power(r_elm.clone(), i)).collect();

    let mut sum = F::from_u128(0);
    
    for ds in dss {
        let rds = r_elms.iter().zip(ds.clone());
        sum = rds.fold(sum, |acc, (r, d)| acc + d*r);
    }
    this_went_well && (sum == F::from_u128(0))
}

// TODO: find a more efficent way to to take the power of an element.
fn power<F: std::ops::MulAssign + Clone + std::ops::Mul<Output = F>>(base:F, exp:usize)->F{
    assert!(exp>0);
    (0..exp-1).fold(base.clone(), |acc, _| acc*base.clone() )
}

#[cfg(test)]
mod test {
    use ff::Field;
    use rand::thread_rng;
    use rand::SeedableRng;

    use crate::{algebra::element::Element32, net::network::InMemoryNetwork};

    use super::*;

    // All these tests use dealer preprosessing
    fn dummie_prepross() -> (
        SpdzContext<Element32>,
        SpdzContext<Element32>,
        preprocessing::SecretValues<Element32>,
    ) {
        let rng = rand::rngs::mock::StepRng::new(42, 7);
        //let rng = thread_rng();
        let known_to_each = vec![1, 1];
        let number_of_triplets = 1;
        let number_of_parties = 2;
        let (mut contexts, secret_values) = preprocessing::dealer_prepross(
            rng,
            known_to_each,
            number_of_triplets,
            number_of_parties,
        );
        let c1 = contexts.pop().unwrap();
        let c0 = contexts.pop().unwrap();
        (c0, c1, secret_values)
    }
    #[test]
    fn test_dealer() {
        //let rng = rand::rngs::mock::StepRng::new(42, 7);
        let rng = thread_rng();
        let known_to_each = vec![1, 2];
        let number_of_triplets = 2;
        let number_of_parties = 2;
        let (mut contexts, secret_values) = preprocessing::dealer_prepross(
            rng,
            known_to_each,
            number_of_triplets,
            number_of_parties,
        );

        // unpacking
        let mac: Element32 = secret_values.mac_key;
        let p2_context = contexts.pop().unwrap();
        let p1_context = contexts.pop().unwrap();
        let p1_params = p1_context.params;
        let p2_params = p2_context.params;
        let mut p1_preprocessed = p1_context.preprocessed_values;
        let mut p2_preprocessed = p2_context.preprocessed_values;
        let p1_known_to_pi = p1_preprocessed.for_sharing.rand_known_to_i.shares;
        let p2_known_to_pi = p2_preprocessed.for_sharing.rand_known_to_i.shares;
        let p1_known_to_me = p1_preprocessed.for_sharing.rand_known_to_me.vals;
        let p2_known_to_me = p2_preprocessed.for_sharing.rand_known_to_me.vals;
        let p1_triplet_1 = p1_preprocessed.triplets.get_triplet();
        let p2_triplet_1 = p2_preprocessed.triplets.get_triplet();
        let p1_triplet_2 = p1_preprocessed.triplets.get_triplet();
        let p2_triplet_2 = p2_preprocessed.triplets.get_triplet();


        // Testing
        assert!(p1_params.mac_key_share + p2_params.mac_key_share == mac);

        let r = p1_known_to_pi[0][0] + p2_known_to_pi[0][0];
        let r2 = p1_known_to_me[0];
        assert!(r.val == r2);
        assert!(r.mac == r2 * mac);

        let s = p1_known_to_pi[1][0] + p2_known_to_pi[1][0];
        let s2 = p2_known_to_me[0];
        assert!(s.val == s2);
        assert!(s.mac == s2 * mac);

        let s1 = p1_known_to_pi[1][1] + p2_known_to_pi[1][1];
        let s3 = p2_known_to_me[1];
        assert!(s1.val == s3);
        assert!(s1.mac == s3 * mac);

        let a1 = p1_triplet_1.a + p2_triplet_1.a;
        let b1 = p1_triplet_1.b + p2_triplet_1.b;
        let c1 = p1_triplet_1.c + p2_triplet_1.c;

        assert!(a1.val * b1.val == c1.val);
        assert!(a1.val * mac == a1.mac);
        assert!(b1.val * mac == b1.mac);
        assert!(c1.val * mac == c1.mac);

        let a2 = p1_triplet_2.a + p2_triplet_2.a;
        let b2 = p1_triplet_2.b + p2_triplet_2.b;
        let c2 = p1_triplet_2.c + p2_triplet_2.c;

        assert!(a2.val * b2.val == c2.val);
        assert!(a2.val * mac == a2.mac);
        assert!(b2.val * mac == b2.mac);
        assert!(c2.val * mac == c2.mac);
    }
    #[test]
    fn test_sharing_using_send_and_resive() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        assert!(p1_context.params.who_am_i == 0);
        assert!(p2_context.params.who_am_i == 1);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong while P1 was sending the share.");

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        )
        .expect("Something went wrong while P2 was receiving the share.");
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!((elm1_1.mac + elm1_2.mac) == (elm1 * secret_values.mac_key));

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(
            elm2,
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the share");

        let elm2_1 = receive_share_from(
            correction,
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went wrong while P1 was receiving the share");
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);
    }
    #[tokio::test]
    async fn test_sharing_using_share() {
        type F = Element32;
        use crate::net::network::InMemoryNetwork;
        //let rng = rand::rngs::mock::StepRng::new(42, 7);
        let rng = thread_rng();
        let known_to_each = vec![1, 0];
        let number_of_triplets = 0;
        let number_of_parties = 2;
        let (mut contexts, _) = preprocessing::dealer_prepross(
            rng,
            known_to_each,
            number_of_triplets,
            number_of_parties,
        );
        let value = F::from_u128(91u128);
        let elms = [Some(value), None];
        let values = [value, value];
        async fn do_mpc(
            mut network: InMemoryNetwork,
            elm: Option<F>,
            val: F,
            mut context: SpdzContext<F>,
        ) {
            let element = share(
                elm,
                &mut context.preprocessed_values.for_sharing,
                0,
                &context.params,
                &mut network,
            )
            .await;
            let elm = element.expect("Something went wrong in sharing");
            let res = open_res(
                elm,
                &mut network,
                &context.params,
                &context.opened_values,
            )
            .await;
            assert!(val == res);
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        let mut i = 0;

        contexts.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(network, elms[i], values[i], contexts.pop().unwrap()));
            i += 1;
        }

        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
    #[tokio::test]
    async fn test_partial_opening_and_check_all_partial_values_one_by_one() {
        use crate::net::network::InMemoryNetwork;
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = send_share(
            elm1,
            &mut p1_context.preprocessed_values.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was sending the element.");

        let elm1_2 = receive_share_from(
            correction,
            &mut (p2_context.preprocessed_values.for_sharing),
            0,
            &p2_context.params,
        )
        .expect("Something went worng when P2 was receiving the element");
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(
            elm2,
            &mut p2_context.preprocessed_values.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the element.");

        let elm2_1 = receive_share_from(
            correction,
            &mut p1_context.preprocessed_values.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went worng when P1 was receiving the element");
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        // Parties opens elm1
        let p1_partially_opened_vals = p1_context.opened_values;
        let p2_partially_opened_vals = p2_context.opened_values;
        let mut partially_opened_vals = vec![p1_partially_opened_vals, p2_partially_opened_vals];
        let params = vec![
            p1_context.params,
            p2_context.params,
        ];
        async fn do_mpc(
            network: InMemoryNetwork,
            elm: Share<F>,
            val1: F,
            partially_opened_vals: Vec<F>,
            params: SpdzParams<F>,
        ) {
            //let rng_test = thread_rng();
            let mut network = network;
            let mut partially_opened_vals = partially_opened_vals;
            let val1_guess = partial_opening(
                &elm,
                &params,
                &mut network,
                &mut partially_opened_vals,
            )
            .await;
            assert!(val1_guess == val1);
            for d in partially_opened_vals {
                if !check_one_d(d, &mut network).await {
                    // TODO: check that it actually fails, if there is a wrong value somewhere.
                    panic!("Someone cheated")
                }
            }
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(2); //asuming two players
        let elm1_v = vec![elm1_1, elm1_2];
        let mut i = 0;
        partially_opened_vals.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(
                network,
                elm1_v[i],
                elm1,
                partially_opened_vals.pop().unwrap(),
                params[i].clone(),
            ));
            i += 1;
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
    #[tokio::test]
    async fn test_partial_opening_and_check_all_partial_values_using_check_all() {
        use crate::net::network::InMemoryNetwork;
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = send_share(
            elm1,
            &mut p1_context.preprocessed_values.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was sending the element.");

        let elm1_2 = receive_share_from(
            correction,
            &mut (p2_context.preprocessed_values.for_sharing),
            0,
            &p2_context.params,
        )
        .expect("Something went worng when P2 was receiving the element");
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(
            elm2,
            &mut p2_context.preprocessed_values.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the element.");

        let elm2_1 = receive_share_from(
            correction,
            &mut p1_context.preprocessed_values.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went worng when P1 was receiving the element");
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        // Parties opens elm1
        let p1_partially_opened_vals = p1_context.opened_values;
        let p2_partially_opened_vals = p2_context.opened_values;
        let mut partially_opened_vals = vec![p1_partially_opened_vals, p2_partially_opened_vals];
        let params = vec![
            p1_context.params,
            p2_context.params,
        ];
        async fn do_mpc(
            network: InMemoryNetwork,
            elm: Share<F>,
            val1: F,
            partially_opened_vals: Vec<F>,
            params: SpdzParams<F>,
        ) {
            let mut rng = rand::rngs::mock::StepRng::new(42, 7);
            let random_element = F::random(&mut rng);
            let mut network = network;
            let mut partially_opened_vals = partially_opened_vals;
            let val1_guess = partial_opening(
                &elm,
                &params,
                &mut network,
                &mut partially_opened_vals,
            )
            .await;
            assert!(val1_guess == val1);
            //if !check_all_d(&partially_opened_vals, &mut network).await {
            if !check_all_d(&mut partially_opened_vals, &mut network, random_element).await {
                panic!("Someone cheated")
            }
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(2); //asuming two players
        let elm1_v = vec![elm1_1, elm1_2];
        let mut i = 0;
        partially_opened_vals.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(
                network,
                elm1_v[i],
                elm1,
                partially_opened_vals.pop().unwrap(),
                params[i].clone(),
            ));
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
        let (elm1_1, correction) = send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was sending the element.");

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was receiving the element.");
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = send_share(
            elm2,
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the element.");

        let elm2_1 = receive_share_from(
            correction,
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was receiving the element.");
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        // Adding ss-elements
        let elm3_1 = elm1_1 + elm2_1;

        let elm3_2 = elm1_2 + elm2_2;

        assert!(elm1 + elm2 == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1 + elm2) * secret_values.mac_key);
    }
    #[test]
    fn test_subtracting() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(
            elm2,
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let elm2_1 = match receive_share_from(
            correction,
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        // Subtracting ss-elements
        let elm3_1 = elm1_1 - elm2_1;

        let elm3_2 = elm1_2 - elm2_2;

        assert!(elm1 - elm2 == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1 - elm2) * secret_values.mac_key);
    }
    #[test]
    fn test_multiplication_with_pub_constant() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // Multiplying with pub_constant
        let elm3_1 = elm1_1 * pub_constant;

        let elm3_2 = elm1_2 * pub_constant;

        assert!(elm1 * pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1 * pub_constant) * secret_values.mac_key);
    }
    #[tokio::test]
    async fn test_secret_shared_multipllication() {
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_prepross();

        // P1 shares an element
        let p1_prepros = &mut p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let p2_prepros = &mut p2_context.preprocessed_values;
        let elm1_2 = match receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_share(
            elm2,
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let elm2_1 = match receive_share_from(
            correction,
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        let expected_res = Share {
            val: elm1 * elm2,
            mac: elm1 * elm2 * secret_values.mac_key,
        };
        // Multiplicate
        async fn do_mpc(
            network: InMemoryNetwork,
            s1: Share<F>,
            s2: Share<F>,
            context: SpdzContext<F>,
            expected_res: Share<F>,
        ) {
            let mut context = context;
            let mut network = network;
            let res_share_result = secret_mult(s1, s2, &mut context.preprocessed_values.triplets, &context.params, &mut context.opened_values, &mut network).await;
            let res_share = match res_share_result {
                Ok(share) => share,
                Err(_) => panic!(),
            };
            let res = partial_opening(
                &res_share,
                &context.params,
                &mut network,
                &mut context.opened_values,
            )
            .await;
            assert!(expected_res.val == res);
            let rng = rand::rngs::mock::StepRng::new(42, 7);
            assert!(check_all_d(&mut context.opened_values, &mut network, F::random(rng)).await);

            let res = open_res(
                res_share,
                &mut network,
                &context.params,
                &context.opened_values,
            )
            .await;
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
            taskset.spawn(do_mpc(
                network,
                elm1_v[i],
                elm2_v[i],
                context_here,
                expected_res,
            ));
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
        let (elm1_1, correction) = match send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // Adding with pub_constant
        let elm3_1 = elm1_1.add_public(
            pub_constant,
            0 == p1_context.params.who_am_i,
            &p1_context.params,
        );

        let elm3_2 = elm1_2.add_public(
            pub_constant,
            0 == p2_context.params.who_am_i,
            &p2_context.params,
        );

        assert!(elm1 + pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1 + pub_constant) * secret_values.mac_key);
    }
    #[test]
    fn test_sub_with_public_constant() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_prepross();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed_values;
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match send_share(
            elm1,
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e, c),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_share_from(
            correction,
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s,
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // Adding with pub_constant
        let elm3_1 = elm1_1.sub_public(
            pub_constant,
            0 == p1_context.params.who_am_i,
            &p1_context.params,
        );

        let elm3_2 = elm1_2.sub_public(
            pub_constant,
            0 == p2_context.params.who_am_i,
            &p2_context.params,
        );

        assert!(elm1 - pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1 - pub_constant) * secret_values.mac_key);
    }
    #[tokio::test]
    async fn test_all_with_dealer() {
        // p1 supplies val_p1_1 and val_p1_2
        // p2 supplies val_p2
        // Both parties knows constant const_1
        // They calculate:
        // val_3 = val_p1_1 * val_p2
        // val_4 = val_3 + val_p1_2
        // val_5 = val_4 + const_1
        // open val_5

        // preprosessing by dealer
        type F = Element32;
        //let rng = rand::rngs::mock::StepRng::new(42, 7);
        let rng = thread_rng();
        let known_to_each = vec![2, 1];
        let number_of_triplets = 1;
        let number_of_parties = 2;
        let (mut contexts, _secret_values) = preprocessing::dealer_prepross(
            rng,
            known_to_each,
            number_of_triplets,
            number_of_parties,
        );
        //let mac: F = secret_values.mac_key;
        //let p2_context = contexts.pop().unwrap();
        //let p1_context = contexts.pop().unwrap();

        let val_p1_1 = F::from_u128(2u128);
        let val_p1_2 = F::from_u128(3u128);
        let val_p2 = F::from_u128(5u128);
        let const_1 = F::from_u128(7u128);
        let mut values_both = vec![
            vec![Some(val_p1_1), None, Some(val_p1_2)],
            vec![None, Some(val_p2), None],
        ];
        let constants = [const_1, const_1];
        async fn do_mpc(
            mut network: InMemoryNetwork,
            mut context: SpdzContext<F>,
            values: Vec<Option<F>>,
            constant: F,
        ) {
            //context.rng = Some(thread_rng());
            //let mut local_rng = thread_rng();
            // P1 sharing a value: val_p1_1
            let val_p1_1_res = share(
                values[0],
                &mut context.preprocessed_values.for_sharing,
                0,
                &context.params,
                &mut network,
            )
            .await;
            let val_p1_1 = val_p1_1_res.expect("Something went wrong in sharing elm_p1_1");

            // P2 sharing a value: val_p2
            let val_p2_res = share(
                values[1],
                &mut context.preprocessed_values.for_sharing,
                1,
                &context.params,
                &mut network,
            )
            .await;
            let val_p2 = val_p2_res.expect("Something went wrong in sharing elm_p2");

            // multiplying val_p1_1 and val_p2: val_3
            let val_3_res = secret_mult(val_p1_1, val_p2, &mut context.preprocessed_values.triplets, &context.params, &mut context.opened_values, &mut network).await;
            let val_3 = val_3_res.expect("Something went wrong in multiplication");
            assert!(context.opened_values.len() == 2); // Each multiplication needs partial opening of two elements.

            // P1 sharing a value: val_p1_2
            let val_p1_2_res = share(
                values[2],
                &mut context.preprocessed_values.for_sharing,
                0,
                &context.params,
                &mut network,
            )
            .await;
            let val_p1_2 = val_p1_2_res.expect("Something went wrong in sharing elm_p1_2");

            // Adding val_3 and val_p1_2: val_4
            let val_4 = val_3 + val_p1_2;

            // Adding val_4 with public constant const_1: val_5
            let const_1 = constant;
            let val_5 = val_4.add_public(
                const_1,
                context.params.who_am_i == 0,
                &context.params,
            );
            // Checking all partially opened values
            let mut rng = rand_chacha::ChaCha20Rng::from_entropy();
            let random_element = F::random(&mut rng);
            assert!(check_all_d(&mut context.opened_values, &mut network, random_element).await);

            // opening(and checking) val_5
            let res = open_res(val_5, &mut network, &context.params, &context.opened_values).await;
            assert!(res == F::from_u128(20u128));
        }
        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        let mut i = 0;
        contexts.reverse();
        values_both.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(
                network,
                contexts.pop().unwrap(),
                values_both.pop().unwrap(),
                constants[i],
            ));
            i += 1;
        }

        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
    #[tokio::test]
    async fn test_share_many() {
        // TODO: write a propper test!
        type F = Element32;
        let rng = rand::rngs::mock::StepRng::new(42, 7);
        let known_to_each = vec![2, 0];
        let number_of_triplets = 0;
        let number_of_parties = 2;
        let (mut contexts, _secret_values) = preprocessing::dealer_prepross(
            rng,
            known_to_each,
            number_of_triplets,
            number_of_parties,
        );

        let val_p1_1 = F::from_u128(2u128);
        let val_p1_2 = F::from_u128(3u128);
        let mut values_both = vec![
            vec![Some(val_p1_1), Some(val_p1_2)],
            vec![None, None],
        ];
        async fn do_mpc(
            mut network: InMemoryNetwork,
            mut context: SpdzContext<F>,
            values: Vec<Option<F>>,
        ) {
            //context.rng = Some(thread_rng());
            //let mut local_rng = thread_rng();
            // P1 sharing a value: val_p1_1
            let val_p1_res = share_many(
                values,
                &mut context.preprocessed_values.for_sharing,
                & context.params,
                0,
                &mut network,
            )
            .await;

        }
        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        let mut i = 0;
        contexts.reverse();
        values_both.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(
                network,
                contexts.pop().unwrap(),
                values_both.pop().unwrap(),
            ));
            i += 1;
        }

        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
    #[test]
    fn test_power(){
        type F = Element32;
        let a:F = F::from_u128(12u128);
        assert!(a == power(a, 1)); 
        let aa = a*a;
        assert!(aa == power(a, 2)); 
        let aaa = a*a*a;
        assert!(aaa == power(a, 3)); 
        let aaaa = a*a*a*a;
        assert!(aaaa == power(a, 4)); 
        let aaaaa = a*a*a*a*a;
        assert!(aaaaa == power(a, 5)); 
        let aaaaaa = a*a*a*a*a*a;
        assert!(aaaaaa == power(a, 6)); 

    }
    #[test]
    fn test_dealer_writing_to_file(){
        
        // preprosessing by dealer
        type F = Element32;
        let file_names = vec![Path::new("src/schemes/spdz/context2.bin"), Path::new("src/schemes/spdz/context1.bin")] ;
        let known_to_each = vec![1, 2];
        let number_of_triplets = 2;
        preprocessing::write_preproc_to_file(
            file_names.clone(),
            known_to_each,
            number_of_triplets,
            F::from_u128(0u128),
        ).unwrap();
        let p1_context: SpdzContext<F> = preprocessing::read_preproc_from_file(
            file_names[0],
        );
        let p2_context: SpdzContext<F> = preprocessing::read_preproc_from_file(
            file_names[1],
        );
        // unpacking
        let p1_params = p1_context.params;
        let p2_params = p2_context.params;
        let mac = p1_params.mac_key_share + p2_params.mac_key_share;
        let mut p1_preprocessed = p1_context.preprocessed_values;
        let mut p2_preprocessed = p2_context.preprocessed_values;
        let p1_known_to_pi = p1_preprocessed.for_sharing.rand_known_to_i.shares;
        let p2_known_to_pi = p2_preprocessed.for_sharing.rand_known_to_i.shares;
        let p1_known_to_me = p1_preprocessed.for_sharing.rand_known_to_me.vals;
        let p2_known_to_me = p2_preprocessed.for_sharing.rand_known_to_me.vals;
        let p1_triplet_1 = p1_preprocessed.triplets.get_triplet();
        let p2_triplet_1 = p2_preprocessed.triplets.get_triplet();
        let p1_triplet_2 = p1_preprocessed.triplets.get_triplet();
        let p2_triplet_2 = p2_preprocessed.triplets.get_triplet();


        // Testing
        let r = p1_known_to_pi[0][0] + p2_known_to_pi[0][0];
        let r2 = p1_known_to_me[0];
        assert!(r.val == r2);
        assert!(r.mac == r2 * mac);

        let s = p1_known_to_pi[1][0] + p2_known_to_pi[1][0];
        let s2 = p2_known_to_me[0];
        assert!(s.val == s2);
        assert!(s.mac == s2 * mac);

        let s1 = p1_known_to_pi[1][1] + p2_known_to_pi[1][1];
        let s3 = p2_known_to_me[1];
        assert!(s1.val == s3);
        assert!(s1.mac == s3 * mac);

        let a1 = p1_triplet_1.a + p2_triplet_1.a;
        let b1 = p1_triplet_1.b + p2_triplet_1.b;
        let c1 = p1_triplet_1.c + p2_triplet_1.c;

        assert!(a1.val * b1.val == c1.val);
        assert!(a1.val * mac == a1.mac);
        assert!(b1.val * mac == b1.mac);
        assert!(c1.val * mac == c1.mac);

        let a2 = p1_triplet_2.a + p2_triplet_2.a;
        let b2 = p1_triplet_2.b + p2_triplet_2.b;
        let c2 = p1_triplet_2.c + p2_triplet_2.c;

        assert!(a2.val * b2.val == c2.val);
        assert!(a2.val * mac == a2.mac);
        assert!(b2.val * mac == b2.mac);
        assert!(c2.val * mac == c2.mac);
    }
    // TODO: test checking - tjek that it can fail.
    // TODO: test errors - in general test that stuff fails when it has to.
}
