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
use rand::RngCore;
use crate::{
    net::agency::Broadcast, protocols::commitments::{commit, commit_many, verify_commit, verify_many}
};
use derive_more::{Add, AddAssign, Sub, SubAssign};

use self::preprocessing::ForSharing;

pub mod preprocessing;
use std::{error, path::Path};

// Should we allow Field or use PrimeField?
#[derive(
    Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign, serde::Serialize, serde::Deserialize, PartialEq,
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
pub async fn secret_mult<F>(
    s1: Share<F>,
    s2: Share<F>,
    triplets: &mut preprocessing::Triplets<F>,
    params: &SpdzParams<F>,
    opened_values: &mut Vec<F>,
    network: &mut impl Broadcast,
) -> Result<Share<F>, preprocessing::MissingPreProcError>
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    let triplet = match triplets
        .get_triplet(){
            Ok(tri) => tri,
            Err(e) => return Err(e),
        };
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
        todo!()
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
        todo!()
    }

    // This might not be the propper way to do recombine - it depends on what exanctly recombine is suppose to mean :)
    async fn recombine(ctx: Self::Context, share: Self, mut network: impl Communicate) -> Result<F, ()> {
        Ok(open_res(share, &mut network, &ctx.params, todo!("opened values")).await)
    }
}

// This impl of share_many only works if the party is either sending or resiving all the elements. 
// This is by design, as only the sender needs to broadcast, and that is where there are something to be won by doing it in bulk. 
pub async fn share<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    op_vals: Option<Vec<F>>, 
    for_sharing: &mut ForSharing<F>,
    params: &SpdzParams<F>,
    who_is_sending: usize,
    network: &mut impl Broadcast,
) -> Result<Vec<Share<F>>, Box<dyn error::Error>> {
    let is_chosen_one = params.who_am_i == who_is_sending; 
    if is_chosen_one {
        let values = op_vals.expect("The sender needs to enter the value to be send"); 
        // TODO: return some error 
        let res = send_shares(
            values,
            for_sharing,
            params,
        );
        let (shares, corrections) = match res {
            Ok((shares, corrections)) => (shares, corrections),
            Err(e) => return Err(e.into()),
        };
        match network
            .broadcast(&corrections)
            .await {
                Ok(_) => (),
                Err(e) => return Err(e.into())
            };
        Ok(shares)
    } else {
        let corrections = match network.recv_from(who_is_sending).await{
            Ok(vec) => vec,
            Err(e) => return Err(e.into()),
        };
        Ok(receive_shares_from(
            corrections,
            for_sharing,
            who_is_sending,
            params,
        )?)
    }    
}

fn send_shares<F: PrimeField>(
    vals: Vec<F>,
    for_sharing: &mut ForSharing<F>,
    params: &SpdzParams<F>, 
) -> Result<(Vec<Share<F>>, Vec<F>), preprocessing::MissingPreProcError> {
    let mut res_share: Vec<Share<F>> = vec![];
    let mut res_correction: Vec<F> = vec![];
    for val in vals {
        let r = match for_sharing.rand_known_to_me.vals.pop() {
            Some(elm) => elm,
            None => return Err(preprocessing::MissingPreProcError{e_type: preprocessing::MissingPreProcErrorType::MissingForSharingElement}),
        };
        let r_share = match for_sharing.rand_known_to_i.shares[params.who_am_i].pop() {
            Some(r_share) => r_share,
            None => {
                return Err(preprocessing::MissingPreProcError{e_type: preprocessing::MissingPreProcErrorType::MissingForSharingElement});
            }
        };
        let correction = val - r;
        let share = r_share.add_public(correction, params.who_am_i == 0, params);
        res_share.push(share);
        res_correction.push(correction);
    }
    Ok((res_share, res_correction))
}

// When receiving a share, the party receiving it needs to know who send it.
fn receive_shares_from<F: PrimeField>(
    corrections: Vec<F>,
    for_sharing: &mut ForSharing<F>,
    who_is_sending: usize,
    params: &SpdzParams<F>,
) -> Result<Vec<Share<F>>, preprocessing::MissingPreProcError> {
    let prep_rand_len = for_sharing.rand_known_to_i.shares[who_is_sending].len();
    let n = corrections.len();
    if n > for_sharing.rand_known_to_i.shares[who_is_sending].len(){
        return Err(preprocessing::MissingPreProcError{e_type: preprocessing::MissingPreProcErrorType::MissingForSharingElement}); 
    }
    let mut randoms = for_sharing.rand_known_to_i.shares[who_is_sending].split_off(prep_rand_len - n);
    // TODO consider changing send_shares to also use split_off instead of pop, so we don't need to reverse. 
    randoms.reverse(); 
    let rc = randoms.iter().zip(corrections);
    let shares: Vec<Share<F>> = rc.map(|(r,c)| r.add_public(c, params.who_am_i() == 0, params)).collect();
    Ok(shares)
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
    pub who_am_i: usize,
}

impl<F: PrimeField> SpdzParams<F> {
    pub fn who_am_i(&self) -> usize{
        self.who_am_i
    }
}

// The SPDZ context needs to be public atleast to some degree, as it is needed for many operations that we would like to call publicly.
// If we do not want the context to be public, we probably need some getter functions - and some alter functions. (TODO)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpdzContext<F: PrimeField> {
    pub opened_values: Vec<F>,
    pub params: SpdzParams<F>,
    pub preprocessed_values: preprocessing::PreprocessedValues<F>,
}

// Consider keeping both open_res and open_res_many, as open_res_many needs a random element to be picked, which is a non negligible overhead when only one element is verified.
pub async fn open_res_many<F>(
    shares_to_open: Vec<Share<F>>,
    network: &mut impl Broadcast,
    params: &SpdzParams<F>,
    prev_opened_values: &Vec<F>,
    random_element: F,
)-> Vec<F>
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    if !prev_opened_values.is_empty() {
        panic!("don't open if there are unchecked open values") // TODO rerun error instead 
        // TODO: consider just calling check all - needs a random element though - either a generator or an element. 
        //check_all_d(opened_values, network, random_element)
    }
    let n = shares_to_open.len();
    let (vals_to_open, macs_to_shares) :(Vec<F>, Vec<F>) = shares_to_open.iter().map(|share| (share.val, share.mac)).collect();
    let mut opened_vals:Vec<Vec<F>> = network
        .symmetric_broadcast(vals_to_open)
        .await
        .unwrap();
    // TODO: find a nicer sollution to adding up the diffrent vectors to one:
    let mut opened_vals_sum:Vec<F> = opened_vals.pop().expect("Atleast one element must be opened");
    while opened_vals.len() > 0 {
        let ov = opened_vals.pop().expect("we just verified that there are elements left");
        for i in 0..n{
            opened_vals_sum[i] += ov[i];
        }
    }

    let opened_shares = opened_vals_sum.iter().zip(macs_to_shares);
    let mut ds: Vec<F> = opened_shares.map(|(v,m)| params.mac_key_share * v - m).collect();
    let this_went_well = check_all_d(&mut ds, network, random_element).await; 
    if this_went_well {
        opened_vals_sum
    } else {
        panic!("The check did not go though");
    }
}
// TODO: return an option or a result instead. - result probably makes most sense, but then we might want the costom errors first
pub async fn open_res<F>(
    share_to_open: Share<F>,
    network: &mut impl Broadcast,
    params: &SpdzParams<F>,
    opened_values: &Vec<F>,
) -> F
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{   
    if !opened_values.is_empty() {
        panic!("don't open if there are unchecked open values") // TODO rerun error instead 
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
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
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
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
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
    use std::os::unix::net;

    use ff::Field;
    use rand::thread_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

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
        let p1_triplet_1 = p1_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");
        let p2_triplet_1 = p2_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");
        let p1_triplet_2 = p1_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");
        let p2_triplet_2 = p2_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");


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
        let (elm1_1_v, correction_v) = send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong while P1 was sending the share.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]); 

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        )
        .expect("Something went wrong while P2 was receiving the share.")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!((elm1_1.mac + elm1_2.mac) == (elm1 * secret_values.mac_key));

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2_v, correction_v) = send_shares(
            vec![elm2],
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the share");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]); 

        let elm2_1 = receive_shares_from(
            vec![correction],
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went wrong while P1 was receiving the share")[0];
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);
    }
    #[tokio::test]
    async fn test_sharing_using_share_many() {
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
        let elms = [Some(vec![value]), None];
        let values = [value, value];
        async fn do_mpc(
            mut network: InMemoryNetwork,
            elm: Option<Vec<F>>,
            val: F,
            mut context: SpdzContext<F>,
        ) {
            let element = share(
                elm,
                &mut context.preprocessed_values.for_sharing,
                &context.params,
                0,
                &mut network,
            )
            .await;
            let elm = element.expect("Something went wrong in sharing");
            let res = open_res(
                elm[0],
                &mut network,
                &context.params,
                &context.opened_values,
                //F::from_u128(8u128),
            )
            .await;
            //.await.pop().expect("There should be a result");
            assert!(val == res);
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        let mut i = 0;

        contexts.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(network, elms[i].clone(), values[i], contexts.pop().unwrap()));
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
        let (elm1_1_v, correction_v) = send_shares(
            vec![elm1],
            &mut p1_context.preprocessed_values.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was sending the element.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]); 

        let elm1_2 = receive_shares_from(
            vec![correction],
            &mut (p2_context.preprocessed_values.for_sharing),
            0,
            &p2_context.params,
        )
        .expect("Something went worng when P2 was receiving the element")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2_v, correction_v) = send_shares(
            vec![elm2],
            &mut p2_context.preprocessed_values.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the element.");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]); 

        let elm2_1 = receive_shares_from(
            vec![correction],
            &mut p1_context.preprocessed_values.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went worng when P1 was receiving the element")[0];
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
        let (elm1_1_v, correction_v) = send_shares(
            vec![elm1],
            &mut p1_context.preprocessed_values.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was sending the element.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]); 

        let elm1_2 = receive_shares_from(
            vec![correction],
            &mut (p2_context.preprocessed_values.for_sharing),
            0,
            &p2_context.params,
        )
        .expect("Something went worng when P2 was receiving the element")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2_v, correction_v) = send_shares(
            vec![elm2],
            &mut p2_context.preprocessed_values.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the element.");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]); 

        let elm2_1 = receive_shares_from(
            vec![correction],
            &mut p1_context.preprocessed_values.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went worng when P1 was receiving the element")[0];
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
        let (elm1_1_v, correction_v) = send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was sending the element.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]); 

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was receiving the element.")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2_v, correction_v) = send_shares(
            vec![elm2],
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was sending the element.");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]); 

        let elm2_1 = receive_shares_from(
            vec![correction],
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        )
        .expect("Something went wrong when P1 was receiving the element.")[0];
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
        let (elm1_1, correction) = match send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s[0],
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_shares(
            vec![elm2],
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let elm2_1 = match receive_shares_from(
            vec![correction],
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        ) {
            Ok(s) => s[0],
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
        let (elm1_1, correction) = match send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(e) => {println!("Error: {}", e); panic!()},
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s[0],
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
        let (elm1_1, correction) = match send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let p2_prepros = &mut p2_context.preprocessed_values;
        let elm1_2 = match receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s[0],
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (elm2_2, correction) = match send_shares(
            vec![elm2],
            &mut p2_prepros.for_sharing,
            &p2_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let elm2_1 = match receive_shares_from(
            vec![correction],
            &mut p1_prepros.for_sharing,
            1,
            &p1_context.params,
        ) {
            Ok(s) => s[0],
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
        let (elm1_1, correction) = match send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s[0],
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
        let (elm1_1, correction) = match send_shares(
            vec![elm1],
            &mut p1_prepros.for_sharing,
            &p1_context.params,
        ) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed_values;
        let elm1_2 = match receive_shares_from(
            vec![correction],
            &mut p2_prepros.for_sharing,
            0,
            &p2_context.params,
        ) {
            Ok(s) => s[0],
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
            vec![Some(vec![val_p1_1]), None, Some(vec![val_p1_2])],
            vec![None, Some(vec![val_p2]), None],
        ];
        let constants = [const_1, const_1];
        async fn do_mpc(
            mut network: InMemoryNetwork,
            mut context: SpdzContext<F>,
            values: Vec<Option<Vec<F>>>,
            constant: F,
        ) {
            // P1 sharing a value: val_p1_1
            let val_p1_1_res = share(
                values[0].clone(),
                &mut context.preprocessed_values.for_sharing,
                &context.params,
                0,
                &mut network,
            )
            .await;
            let val_p1_1 = val_p1_1_res.expect("Something went wrong in sharing elm_p1_1")[0];

            // P2 sharing a value: val_p2
            let val_p2_res = share(
                values[1].clone(),
                &mut context.preprocessed_values.for_sharing,
                &context.params,
                1,
                &mut network,
            )
            .await;
            let val_p2 = val_p2_res.expect("Something went wrong in sharing elm_p2")[0];

            // multiplying val_p1_1 and val_p2: val_3
            let val_3_res = secret_mult(val_p1_1, val_p2, &mut context.preprocessed_values.triplets, &context.params, &mut context.opened_values, &mut network).await;
            let val_3 = val_3_res.expect("Something went wrong in multiplication");
            assert!(context.opened_values.len() == 2); // Each multiplication needs partial opening of two elements.

            // P1 sharing a value: val_p1_2
            let val_p1_2_res = share(
                values[2].clone(),
                &mut context.preprocessed_values.for_sharing,
                &context.params,
                0,
                &mut network,
            )
            .await;
            let val_p1_2 = val_p1_2_res.expect("Something went wrong in sharing elm_p1_2")[0];

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
        type F = Element32;
        let rng = rand_chacha::ChaCha20Rng::from_entropy();
        let mut rng2 = rand_chacha::ChaCha20Rng::from_entropy();
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
            Some(vec![val_p1_1, val_p1_2]),
            None,
        ];
        let values_for_checking = vec![val_p1_1,val_p1_2];
        let mut random_values = vec![F::random(&mut rng2), F::random(&mut rng2)];

        async fn do_mpc(
            mut network: InMemoryNetwork,
            mut context: SpdzContext<F>,
            values: Option<Vec<F>>,
            vals_for_checking: Vec<F>,
            random_value: F,
        ) {
            let val_p1_res = share(
                values.clone(),
                &mut context.preprocessed_values.for_sharing,
                & context.params,
                0,
                &mut network,
            )
            .await.expect("this is a test and the values are there");
            
            let opened_res = open_res_many(val_p1_res, &mut network, &context.params, &context.opened_values, random_value).await;
            let vals_zip = opened_res.iter().zip(vals_for_checking);
            for (res, check_res) in vals_zip {
                assert!(*res==check_res);
            }

        }
        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        contexts.reverse();
        values_both.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(
                network,
                contexts.pop().unwrap(),
                values_both.pop().unwrap(),
                values_for_checking.clone(),
                random_values.pop().unwrap(),
            ));
        }

        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }
    #[tokio::test]
    async fn test_missingpreprocerror() {
        type F = Element32;
        let rng = rand_chacha::ChaCha20Rng::from_entropy();
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
            vec![Some(vec![val_p1_1, val_p1_2]), None],
            vec![None, Some(vec![val_p1_1,val_p1_2])],
        ];

        async fn do_mpc(
            mut network: InMemoryNetwork,
            mut context: SpdzContext<F>,
            values: Vec<Option<Vec<F>>>,
        ) {
            let values_1 = values[0].clone();
            let values_2 = values[1].clone();
            let mut val_p1_res = share(
                values_1,
                &mut context.preprocessed_values.for_sharing,
                & context.params,
                0,
                &mut network,
            )
            .await.expect("this is a test, the value is there");

            let val_2 = val_p1_res.pop().expect("this is a test, the value is there");
            let val_1 = val_p1_res.pop().expect("this is a test, the value is there");
            
            let val_3_res = secret_mult(val_1, val_2, &mut context.preprocessed_values.triplets, &context.params, &mut context.opened_values, &mut network).await;
            let val_3 = val_3_res;
            assert!(val_3.is_err());
            match val_3 {
                Ok(_) => panic!("can't happen, we just checked"),
                Err(e) => println!("Error: {}", e),
            };
            
            let val_p2_res = share(
                values_2,
                &mut context.preprocessed_values.for_sharing,
                & context.params,
                1,
                &mut network,
            )
            .await;
            assert!(val_p2_res.is_err());
            match val_p2_res {
                Ok(_) => panic!("can't happen, we just checked"),
                Err(e) => println!("Error: {}", e),
            };

        }
        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        contexts.reverse();
        values_both.reverse();
        for network in cluster {
            taskset.spawn(do_mpc(
                network,
                contexts.pop().unwrap(),
                values_both.pop().unwrap(),
            ));
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
        let p1_triplet_1 = p1_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");
        let p2_triplet_1 = p2_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");
        let p1_triplet_2 = p1_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");
        let p2_triplet_2 = p2_preprocessed.triplets.get_triplet().expect("This is a test, the triplet is there.");


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
    // TODO: test errors - in general test that stuff fails when it has to.
}
