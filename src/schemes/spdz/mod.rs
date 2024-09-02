//
//! This SPDZ implementation is primarely based on the following lecture by Ivan Damgård:
//! (part one:) <https://www.youtube.com/watch?v=N80DV3Brds0> (and part two:) <https://www.youtube.com/watch?v=Ce45hp24b2E>
//!
//! We will need some homomorphic encryption or oblivious transfer to enable preprocessing.
//! But for now that is handled by a dealer.
//!

// TODO: make costum errors.
use crate::{
    algebra::math::Vector,
    net::{agency::Broadcast, mux, Id},
    protocols::commitments::{commit, verify_commit},
    schemes::{interactive::InteractiveSharedMany, spdz::preprocessing::FuelTank},
};
use crate::{net::Communicate, schemes::interactive::InteractiveShared};
use derive_more::{Add, AddAssign, Sub, SubAssign};
use ff::PrimeField;
use rand::RngCore;
use serde::{de::DeserializeOwned, Serialize};

use futures_concurrency::prelude::*;
pub mod preprocessing;
use std::{convert::Infallible, error::Error};
use tracing::Instrument;

// Should we allow Field or use PrimeField?
#[derive(
    Debug,
    Clone,
    Copy,
    Add,
    Sub,
    AddAssign,
    SubAssign,
    serde::Serialize,
    serde::Deserialize,
    PartialEq,
)]
pub struct Share<F: PrimeField> {
    val: F,
    mac: F,
}

mod ops {
    use std::ops::{AddAssign, SubAssign};

    use ff::PrimeField;

    use crate::{
        net::{agency::Broadcast, Id},
        schemes::spdz::{partial_opening, preprocessing},
    };

    use super::{Share, SpdzParams};

    impl<F: PrimeField> AddAssign<&Self> for Share<F> {
        fn add_assign(&mut self, rhs: &Self) {
            *self += *rhs;
        }
    }

    impl<F: PrimeField> SubAssign<&Self> for Share<F> {
        fn sub_assign(&mut self, rhs: &Self) {
            *self -= *rhs;
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

    impl<F: PrimeField> Share<F> {
        #[must_use]
        pub fn add_public(self, val: F, is_chosen_party: bool, params: &SpdzParams<F>) -> Self {
            let mac_key_share = params.mac_key_share;
            let val_val = if is_chosen_party { val } else { F::ZERO };
            Share {
                val: self.val + val_val,
                mac: self.mac + val * mac_key_share,
            }
        }

        #[must_use]
        pub fn sub_public(self, val: F, chosen_one: bool, params: &SpdzParams<F>) -> Self {
            let mac_key_share = params.mac_key_share;
            let val_val = if chosen_one { val } else { F::ZERO };
            Share {
                val: self.val - val_val,
                mac: self.mac - val * mac_key_share,
            }
        }
    }

    // Harmonize this with the beaver impl.
    pub async fn secret_mult<F>(
        s1: Share<F>,
        s2: Share<F>,
        triplets: &mut preprocessing::Triplets<F>,
        params: &SpdzParams<F>,
        opened_values: &mut Vec<F>,
        network: &mut impl Broadcast,
    ) -> Result<Share<F>, preprocessing::PreProcError>
    where
        F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
    {
        let triplet = match triplets.get_triplet() {
            Ok(tri) => tri,
            Err(e) => return Err(e),
        };
        let is_chosen_party = params.who_am_i == Id(0);
        let (a, b, c) = triplet.shares;

        let e = s1 - a;
        let d = s2 - b;
        let e = partial_opening(&e, params, network, opened_values).await;
        let d = partial_opening(&d, params, network, opened_values).await;

        let res = (c + b * e + a * d).add_public(e * d, is_chosen_party, params);
        Ok(res)
    }
}

impl<F> InteractiveShared for Share<F>
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    type Context = SpdzContext<F>;
    type Value = F;
    type Error = Infallible;

    async fn share(
        ctx: &mut Self::Context,
        secret: Self::Value,
        _rng: impl RngCore + Send,
        mut coms: impl Communicate,
    ) -> Result<Self, Infallible> {
        let params = &ctx.params;
        let (fuel, rand) = &mut ctx.preprocessed.for_sharing.get_own();
        let res = send_shares(&[secret], fuel, rand, params, &mut coms)
            .await
            .unwrap();
        Ok(res[0])
    }

    async fn symmetric_share(
        ctx: &mut Self::Context,
        secret: Self::Value,
        _rng: impl RngCore + Send,
        mut coms: impl Communicate,
    ) -> Result<Vec<Self>, Self::Error> {
        let number_of_parties = Broadcast::size(&coms);
        let me = ctx.params.who_am_i;
        let (gateway, mut muxes) = mux::NetworkGateway::multiplexify(&mut coms, number_of_parties);
        let params = &ctx.params;
        let (my_fueltank, randomness, others) = ctx.preprocessed.for_sharing.split();

        let mut special = muxes.remove(me.0);

        let futs: Vec<_> = muxes
            .into_iter()
            .zip(others)
            .map(async |(mut coms, fueltank)| {
                let id = fueltank.party.0;
                let span = tracing::info_span!("Receiving", from = id);
                let s = receive_shares(&mut coms, fueltank, params)
                    .await
                    .map(|s| s[0]);
                coms.shutdown().instrument(span).await;
                s
            })
            .collect();
        let mine = async {
            let span = tracing::info_span!("Sending");
            let s = send_shares(&[secret], my_fueltank, randomness, params, &mut special)
                .instrument(span)
                .await
                .map(|s| s[0]);
            special.shutdown().await;
            s
        };

        let (my_result, results, driver) = (mine, futs.join(), gateway.drive()).join().await;
        tracing::info!("Complete!");
        let _ = driver.expect("TODO: Error handling for networking");

        let my_share = my_result.unwrap();

        // TODO: Weird issue with `try_join`
        let mut shares: Vec<_> = results.into_iter().map(|x| x.unwrap()).collect();
        shares.insert(me.0, my_share);

        Ok(shares)
    }

    async fn receive_share(
        ctx: &mut Self::Context,
        mut coms: impl Communicate,
        from: Id,
    ) -> Result<Self, Infallible> {
        let params = &ctx.params;
        let fueltank = &mut ctx.preprocessed.for_sharing.get_fuel_mut(from);
        let res = receive_shares(&mut coms, fueltank, params).await.unwrap();
        Ok(res[0])
    }

    async fn recombine(
        ctx: &mut Self::Context,
        share: Self,
        mut network: impl Communicate,
    ) -> Result<F, Infallible> {
        Ok(open_res(share, &mut network, &ctx.params, &ctx.opened_values).await)
    }
}

impl<F> InteractiveSharedMany for Share<F>
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    type VectorShare = Vector<Share<F>>;

    async fn share_many(
        ctx: &mut Self::Context,
        secret: &[Self::Value],
        _rng: impl RngCore + Send,
        mut coms: impl Communicate,
    ) -> Result<Self::VectorShare, Self::Error> {
        let params = &ctx.params;
        let (fuel, rand) = &mut ctx.preprocessed.for_sharing.get_own();
        let res = send_shares(secret, fuel, rand, params, &mut coms)
            .await
            .unwrap();
        Ok(res.into())
    }

    async fn symmetric_share_many(
        ctx: &mut Self::Context,
        secret: &[Self::Value],
        _rng: impl RngCore + Send,
        mut coms: impl Communicate,
    ) -> Result<Vec<Self::VectorShare>, Self::Error> {
        let number_of_parties = Broadcast::size(&coms);
        let me = ctx.params.who_am_i;
        let (gateway, mut muxes) = mux::NetworkGateway::multiplexify(&mut coms, number_of_parties);
        let params = &ctx.params;
        let (my_fueltank, randomness, others) = ctx.preprocessed.for_sharing.split();

        let mut special = muxes.remove(me.0);

        let futs: Vec<_> = muxes
            .into_iter()
            .zip(others)
            .map(async |(mut coms, fueltank)| {
                receive_shares(&mut coms, fueltank, params)
                    .await
                    .map(|s| s.into())
            })
            .collect();
        let mine =
            async { send_shares(secret, my_fueltank, randomness, params, &mut special).await };

        let (my_result, results, driver) = (mine, futs.join(), gateway.drive()).join().await;
        let _ = driver.expect("TODO: Error handling for networking");

        let my_share = my_result.unwrap();

        // TODO: Weird issue with `try_join`
        let mut shares: Vec<_> = results.into_iter().map(|x| x.unwrap()).collect();
        shares.insert(me.0, my_share.into());

        Ok(shares)
    }

    async fn receive_share_many(
        ctx: &mut Self::Context,
        mut coms: impl Communicate,
        from: Id,
    ) -> Result<Self::VectorShare, Self::Error> {
        let params = &ctx.params;
        let fueltank = &mut ctx.preprocessed.for_sharing.get_fuel_mut(from);
        let res = receive_shares(&mut coms, fueltank, params).await.unwrap();
        Ok(res.into())
    }

    async fn recombine_many(
        ctx: &mut Self::Context,
        secrets: Self::VectorShare,
        mut coms: impl Communicate,
    ) -> Result<Vector<Self::Value>, Self::Error> {
        // TODO: make random element
        let random_element = F::from_u128(12);
        let res = open_res_many(
            secrets.to_vec(),
            &mut coms,
            &ctx.params,
            &ctx.opened_values,
            random_element,
        )
        .await;
        Ok(res.into())
    }
}

async fn receive_shares<F: PrimeField + Serialize + DeserializeOwned>(
    network: &mut impl Broadcast,
    fueltank: &mut FuelTank<F>,
    params: &SpdzParams<F>,
) -> Result<Vec<Share<F>>, Box<dyn Error + Send + Sync + 'static>> {
    let corrections: Vec<_> = match network.recv_from(fueltank.party).await {
        Ok(vec) => vec,
        Err(e) => return Err(e.into()),
    };
    Ok(create_foreign_share(
        &corrections,
        &mut fueltank.shares,
        params,
    )?)
}

async fn send_shares<F: PrimeField + Serialize + DeserializeOwned>(
    secrets: &[F],
    fueltank: &mut FuelTank<F>,
    randomness: &mut Vec<F>,
    params: &SpdzParams<F>,
    network: &mut impl Broadcast,
) -> Result<Vec<Share<F>>, Box<dyn Error + Send + Sync + 'static>> {
    // TODO: return some error
    let res = create_shares(secrets, fueltank, randomness, params);
    let (shares, corrections) = match res {
        Ok((shares, corrections)) => (shares, corrections),
        Err(e) => return Err(e.into()),
    };
    match network.broadcast(&corrections).await {
        Ok(()) => (),
        Err(e) => return Err(e.into()),
    };
    Ok(shares)
}

fn create_shares<F: PrimeField>(
    vals: &[F],
    fueltank: &mut FuelTank<F>,
    randomness: &mut Vec<F>,
    params: &SpdzParams<F>,
) -> Result<(Vec<Share<F>>, Vec<F>), preprocessing::PreProcError> {
    assert_eq!(
        fueltank.party, params.who_am_i,
        "Fueltank does not match params"
    );
    let n = vals.len();
    let my_randomness = randomness;
    let their_randomness = &mut fueltank.shares;

    // We will consume the last `n` values, so we need to ensure their are `n`
    if n > my_randomness.len() || n > their_randomness.len() {
        return Err(preprocessing::PreProcError::MissingForSharingElement);
    }

    let (res_share, res_correction) = vals
        .iter()
        .zip(my_randomness.drain(my_randomness.len() - n..).rev())
        .zip(their_randomness.drain(their_randomness.len() - n..).rev())
        .map(|((val, r), r_share)| {
            let correction = *val - r;
            let share = r_share.add_public(correction, params.who_am_i == Id(0), params);
            (share, correction)
        })
        .unzip();

    Ok((res_share, res_correction))
}

// When receiving a share, the party receiving it needs to know who send it.
fn create_foreign_share<F: PrimeField>(
    corrections: &[F],
    preshares: &mut Vec<Share<F>>,
    params: &SpdzParams<F>,
) -> Result<Vec<Share<F>>, preprocessing::PreProcError> {
    // TODO: Kill this vvvv
    // We really should be able to pass in a list that is only `m` long
    // and drain it.

    let n = corrections.len();
    if n > preshares.len() {
        return Err(preprocessing::PreProcError::MissingForSharingElement);
    }
    let shares: Vec<Share<F>> = preshares
        .drain(preshares.len() - n..)
        .rev()
        .zip(corrections)
        .map(|(r, &c)| r.add_public(c, params.who_am_i == Id(0), params))
        .collect();
    Ok(shares)
}

// partial opening that handles the broadcasting and pushing to opended values. All that is partially opened needs to be checked later.
async fn partial_opening<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    candidate_share: &Share<F>,
    params: &SpdzParams<F>,
    network: &mut impl Broadcast,
    partially_opened: &mut Vec<F>,
) -> F {
    let candidate_vals = network
        .symmetric_broadcast(candidate_share.val)
        .await
        .unwrap();
    let candidate_val = candidate_vals.iter().sum();
    partially_opened.push(candidate_val * params.mac_key_share - candidate_share.mac);
    candidate_val
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpdzParams<F: PrimeField> {
    mac_key_share: F,
    pub who_am_i: Id,
}

// The SPDZ context needs to be public atleast to some degree, as it is needed for many operations that we would like to call publicly.
// If we do not want the context to be public, we probably need some getter functions - and some alter functions. (TODO)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpdzContext<F: PrimeField> {
    /// Needs to be verified later
    pub opened_values: Vec<F>,
    /// Base parameters (key + id)
    pub params: SpdzParams<F>,
    pub preprocessed: preprocessing::PreprocessedValues<F>,
}

// Consider keeping both open_res and open_res_many, as open_res_many needs a random element to be picked,
// which is a non negligible overhead when only one element is verified.
pub async fn open_res_many<F>(
    shares_to_open: Vec<Share<F>>,
    network: &mut impl Broadcast,
    params: &SpdzParams<F>,
    prev_opened_values: &[F],
    random_element: F,
) -> Vec<F>
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    // TODO: rerun error instead
    // TODO: consider just calling check all - needs a random element though - either a generator or an element.
    //check_all_d(opened_values, network, random_element)
    assert!(
        prev_opened_values.is_empty(),
        "don't open if there are unchecked open values"
    );

    let n = shares_to_open.len();
    let (vals_to_open, macs_to_shares): (Vec<F>, Vec<F>) = shares_to_open
        .iter()
        .map(|share| (share.val, share.mac))
        .collect();
    let mut opened_vals: Vec<Vec<F>> = network.symmetric_broadcast(vals_to_open).await.unwrap();
    // TODO: find a nicer sollution to adding up the diffrent vectors to one:
    let mut opened_vals_sum: Vec<F> = opened_vals
        .pop()
        .expect("Atleast one element must be opened");
    while let Some(ov) = opened_vals.pop() {
        for i in 0..n {
            opened_vals_sum[i] += ov[i];
        }
    }

    let opened_shares = opened_vals_sum.iter().zip(macs_to_shares);
    let ds: Vec<F> = opened_shares
        .map(|(v, m)| params.mac_key_share * v - m)
        .collect();
    let this_went_well = check_all_d(&ds, network, random_element).await;
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
    opened_values: &[F],
) -> F
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    // TODO rerun error instead
    // TODO: consider just calling check all - needs a random element though - either a generator or an element.
    //check_all_d(opened_values, network, random_element)
    assert!(
        opened_values.is_empty(),
        "don't open if there are unchecked open values"
    );

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
    partially_opened_vals: &[F],
    network: &mut impl Broadcast,
    random_element: F,
) -> bool
where
    F: PrimeField + serde::Serialize + serde::de::DeserializeOwned,
{
    //first we need to pick a random value together
    let random_elms_commitments = network
        .symmetric_broadcast(commit(&random_element))
        .await
        .unwrap();
    let random_val_shares = network.symmetric_broadcast(random_element).await.unwrap();
    let rv_c = random_val_shares
        .clone()
        .into_iter()
        .zip(random_elms_commitments);
    if !rv_c.fold(true, |acc, (v, (c, s))| acc && verify_commit(&v, &c, &s)) {
        return false;
    }

    // Then we make a random linear combination of all the values, commit, broadcast, verify that it is zero
    let lin_comp = linear_combinations(random_val_shares.into_iter().sum(), partially_opened_vals);
    let lin_comps_commitments = network
        .symmetric_broadcast(commit(&lin_comp))
        .await
        .unwrap();
    let lin_comps = network.symmetric_broadcast(lin_comp).await.unwrap();
    let lc_c = lin_comps.clone().into_iter().zip(lin_comps_commitments);
    if !lc_c.fold(true, |acc, (lc, (c, s))| acc && verify_commit(&lc, &c, &s)) {
        return false;
    }

    let zero: F = lin_comps.into_iter().sum();
    zero == F::from_u128(0)
}

fn linear_combinations<F: PrimeField>(random_element: F, elements: &[F]) -> F {
    let r_elms: Vec<F> = (1..=elements.len())
        .map(|i| power(&random_element, i))
        .collect();
    elements
        .iter()
        .zip(r_elms)
        .fold(F::from_u128(0), |acc, (e, r)| acc + r * (*e))
}

// TODO: find a more efficent way to to take the power of an element.
fn power<F: std::ops::MulAssign + Clone + std::ops::Mul<Output = F>>(base: &F, exp: usize) -> F {
    assert!(exp > 0);
    (0..exp - 1).fold(base.clone(), |acc, _| acc * base.clone())
}

#[cfg(test)]
mod test {

    use crate::schemes::spdz::{self, preprocessing::PreShareTank};
    use ff::Field;
    use rand::thread_rng;
    use rand::SeedableRng;
    use std::io::Seek;
    use tracing_subscriber::fmt::format::FmtSpan;

    use crate::{
        algebra::element::Element32, net::network::InMemoryNetwork,
        schemes::spdz::ops::secret_mult, testing::Cluster,
    };

    use super::*;

    #[tokio::test]
    async fn symmetric_sharing() {
        let subscriber = tracing_subscriber::fmt()
            .compact()
            .with_line_number(true)
            .with_target(true)
            .without_time()
            .with_ansi(true)
            .with_span_events(FmtSpan::ACTIVE)
            .finish();
        tracing::subscriber::set_global_default(subscriber).unwrap();

        let rng = rand::rngs::mock::StepRng::new(7, 32);
        let (ctxs, _secrets) = preprocessing::dealer_preproc::<Element32>(rng, &[1, 1, 1], 0, 3);

        let res: Vec<u32> = Cluster::new(3)
            .with_args(ctxs)
            .run_with_args(async |mut coms, mut ctx| {
                let rng = rand::rngs::mock::StepRng::new(7, 32);
                let secret = Element32::from(33u32);
                let shares = spdz::Share::symmetric_share(&mut ctx, secret, rng, &mut coms)
                    .instrument(tracing::info_span!("Sharing"))
                    .await
                    .unwrap();

                let share = shares[0] + shares[1] + shares[2];
                spdz::Share::recombine(&mut ctx, share, &mut coms)
                    .instrument(tracing::info_span!("Recombining"))
                    .await
                    .map(|x| x.into())
                    .unwrap()
            })
            .await
            .unwrap();

        assert_eq!(&res, &[99, 99, 99u32]);
    }

    // Legacy function only used in tests
    pub async fn share<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
        secrets: Option<Vec<F>>, // TODO: remove option.
        for_sharing: &mut PreShareTank<F>,
        params: &SpdzParams<F>,
        who_is_sending: Id,
        network: &mut impl Broadcast,
    ) -> Result<Vec<Share<F>>, Box<dyn Error + Send + Sync + 'static>> {
        let is_chosen_one = params.who_am_i == who_is_sending;
        if is_chosen_one {
            let (fuel, rand) = for_sharing.get_own();
            send_shares(&secrets.unwrap(), fuel, rand, params, network).await
        } else {
            let fueltank = for_sharing.get_fuel_mut(who_is_sending);
            receive_shares(network, fueltank, params).await
        }
    }

    // All these tests use dealer preprosessing
    fn dummie_preproc() -> (
        SpdzContext<Element32>,
        SpdzContext<Element32>,
        preprocessing::SecretValues<Element32>,
    ) {
        let rng = rand::rngs::mock::StepRng::new(42, 7);
        //let rng = thread_rng();
        let known_to_each = vec![1, 1];
        let number_of_triplets = 1;
        let number_of_parties = 2;
        let (mut contexts, secret_values) = preprocessing::dealer_preproc(
            rng,
            &known_to_each,
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
        let (mut contexts, secret_values) = preprocessing::dealer_preproc(
            rng,
            &known_to_each,
            number_of_triplets,
            number_of_parties,
        );

        // unpacking
        let mac: Element32 = secret_values.mac_key;
        let p2_context = contexts.pop().unwrap();
        let p1_context = contexts.pop().unwrap();
        let p1_params = p1_context.params;
        let p2_params = p2_context.params;
        let mut p1_preprocessed = p1_context.preprocessed;
        let mut p2_preprocessed = p2_context.preprocessed;
        let p1_known_to_pi: Vec<Vec<_>> = p1_preprocessed.for_sharing.bad_habits();
        let p2_known_to_pi = p2_preprocessed.for_sharing.bad_habits();
        let p1_known_to_me = p1_preprocessed.for_sharing.my_randomness;
        let p2_known_to_me = p2_preprocessed.for_sharing.my_randomness;
        let p1_triplet_1 = p1_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");
        let p2_triplet_1 = p2_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");
        let p1_triplet_2 = p1_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");
        let p2_triplet_2 = p2_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");

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

        let a1 = p1_triplet_1.shares.0 + p2_triplet_1.shares.0;
        let b1 = p1_triplet_1.shares.1 + p2_triplet_1.shares.1;
        let c1 = p1_triplet_1.shares.2 + p2_triplet_1.shares.2;

        assert!(a1.val * b1.val == c1.val);
        assert!(a1.val * mac == a1.mac);
        assert!(b1.val * mac == b1.mac);
        assert!(c1.val * mac == c1.mac);

        let a2 = p1_triplet_2.shares.0 + p2_triplet_2.shares.0;
        let b2 = p1_triplet_2.shares.1 + p2_triplet_2.shares.1;
        let c2 = p1_triplet_2.shares.2 + p2_triplet_2.shares.2;

        assert!(a2.val * b2.val == c2.val);
        assert!(a2.val * mac == a2.mac);
        assert!(b2.val * mac == b2.mac);
        assert!(c2.val * mac == c2.mac);
    }

    #[test]
    fn test_sharing_using_send_and_resive() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_preproc();
        assert!(p1_context.params.who_am_i == Id(0));
        assert!(p2_context.params.who_am_i == Id(1));
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed;
        let elm1 = F::from_u128(56u128);

        let (fueltank, randomness) = p1_prepros.for_sharing.get_own();
        let (elm1_1_v, correction_v) =
            create_shares(&[elm1], fueltank, randomness, &p1_context.params)
                .expect("Something went wrong while P1 was sending the share.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]);

        let mut p2_prepros = p2_context.preprocessed;
        let elm1_2 = create_foreign_share(
            &[correction],
            &mut p2_prepros.for_sharing.party_fuel[0].shares,
            &p2_context.params,
        )
        .expect("Something went wrong while P2 was receiving the share.")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!((elm1_1.mac + elm1_2.mac) == (elm1 * secret_values.mac_key));

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (fueltank, randomness) = p2_prepros.for_sharing.get_own();
        let (elm2_2_v, correction_v) =
            create_shares(&[elm2], fueltank, randomness, &p2_context.params)
                .expect("Something went wrong when P2 was sending the share");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]);

        let elm2_1 = create_foreign_share(
            &[correction],
            &mut p1_prepros.for_sharing.party_fuel[1].shares,
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
        let (mut contexts, _) = preprocessing::dealer_preproc(
            rng,
            &known_to_each,
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
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(0),
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

        contexts.reverse();
        for (i, network) in cluster.into_iter().enumerate() {
            taskset.spawn(do_mpc(
                network,
                elms[i].clone(),
                values[i],
                contexts.pop().unwrap(),
            ));
        }

        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }

    #[tokio::test]
    async fn test_partial_opening_and_check_all_partial_values_one_by_one() {
        use crate::net::network::InMemoryNetwork;
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_preproc();

        // P1 shares an element
        let (fuel, rand) = p1_context.preprocessed.for_sharing.get_own();
        let elm1 = F::from_u128(56u128);
        let (elm1_1_v, correction_v) = create_shares(&[elm1], fuel, rand, &p1_context.params)
            .expect("Something went wrong when P1 was sending the element.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]);

        let elm1_2 = create_foreign_share(
            &[correction],
            p2_context.preprocessed.for_sharing.get_fuel_vec_mut(Id(0)),
            &p2_context.params,
        )
        .expect("Something went worng when P2 was receiving the element")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (fuel, rand) = p2_context.preprocessed.for_sharing.get_own();
        let (elm2_2_v, correction_v) = create_shares(&[elm2], fuel, rand, &p2_context.params)
            .expect("Something went wrong when P2 was sending the element.");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]);

        let elm2_1 = create_foreign_share(
            &[correction],
            p1_context.preprocessed.for_sharing.get_fuel_vec_mut(Id(1)),
            &p1_context.params,
        )
        .expect("Something went worng when P1 was receiving the element")[0];
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        // Parties opens elm1
        let p1_partially_opened_vals = p1_context.opened_values;
        let p2_partially_opened_vals = p2_context.opened_values;
        let mut partially_opened_vals = vec![p1_partially_opened_vals, p2_partially_opened_vals];
        let params = [p1_context.params, p2_context.params];
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
            let val1_guess =
                partial_opening(&elm, &params, &mut network, &mut partially_opened_vals).await;
            assert!(val1_guess == val1);
            for d in partially_opened_vals {
                if !check_one_d(d, &mut network).await {
                    // TODO: check that it actually fails, if there is a wrong value somewhere.
                    panic!("Someone cheated")
                }
            }
        }

        partially_opened_vals.reverse();
        let cluster = Cluster::new(2).with_args([
            (
                elm1_1,
                elm1,
                partially_opened_vals.pop().unwrap(),
                params[0].clone(),
            ),
            (
                elm1_2,
                elm1,
                partially_opened_vals.pop().unwrap(),
                params[1].clone(),
            ),
        ]);

        cluster
            .run_with_args(|network, (elm, val, par_op, params)| {
                do_mpc(network, elm, val, par_op, params)
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_partial_opening_and_check_all_partial_values_using_check_all() {
        use crate::net::network::InMemoryNetwork;
        type F = Element32;
        let (mut p1_context, mut p2_context, secret_values) = dummie_preproc();

        // P1 shares an element
        let elm1 = F::from_u128(56u128);
        let (fuel, rand) = p1_context.preprocessed.for_sharing.get_own();
        let (elm1_1_v, correction_v) = create_shares(&[elm1], fuel, rand, &p1_context.params)
            .expect("Something went wrong when P1 was sending the element.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]);

        let elm1_2 = create_foreign_share(
            &[correction],
            p2_context.preprocessed.for_sharing.get_fuel_vec_mut(Id(0)),
            &p2_context.params,
        )
        .expect("Something went worng when P2 was receiving the element")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (fuel, rand) = p2_context.preprocessed.for_sharing.get_own();
        let (elm2_2_v, correction_v) = create_shares(&[elm2], fuel, rand, &p2_context.params)
            .expect("Something went wrong when P2 was sending the element.");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]);

        let elm2_1 = create_foreign_share(
            &[correction],
            p1_context.preprocessed.for_sharing.get_fuel_vec_mut(Id(1)),
            &p1_context.params,
        )
        .expect("Something went worng when P1 was receiving the element")[0];
        assert!((elm2_1.val + elm2_2.val) == elm2);
        assert!(elm2_1.mac + elm2_2.mac == elm2 * secret_values.mac_key);

        // Parties opens elm1
        let p1_partially_opened_vals = p1_context.opened_values;
        let p2_partially_opened_vals = p2_context.opened_values;
        let mut partially_opened_vals = vec![p1_partially_opened_vals, p2_partially_opened_vals];
        let params = [p1_context.params, p2_context.params];
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
            let val1_guess =
                partial_opening(&elm, &params, &mut network, &mut partially_opened_vals).await;
            assert!(val1_guess == val1);
            //if !check_all_d(&partially_opened_vals, &mut network).await {
            if !check_all_d(&partially_opened_vals, &mut network, random_element).await {
                panic!("Someone cheated")
            }
            partially_opened_vals.clear();
        }

        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(2); //asuming two players
        let elm1_v = [elm1_1, elm1_2];
        partially_opened_vals.reverse();
        for (i, network) in cluster.into_iter().enumerate() {
            taskset.spawn(do_mpc(
                network,
                elm1_v[i],
                elm1,
                partially_opened_vals.pop().unwrap(),
                params[i].clone(),
            ));
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }

    #[test]
    fn test_addition() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_preproc();

        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed;
        let elm1 = F::from_u128(56u128);
        let (fuel, rand) = p1_prepros.for_sharing.get_own();
        let (elm1_1_v, correction_v) = create_shares(&[elm1], fuel, rand, &p1_context.params)
            .expect("Something went wrong when P1 was sending the element.");
        let (elm1_1, correction) = (elm1_1_v[0], correction_v[0]);

        let mut p2_prepros = p2_context.preprocessed;
        let elm1_2 = create_foreign_share(
            &[correction],
            p2_prepros.for_sharing.get_fuel_vec_mut(Id(0)),
            &p2_context.params,
        )
        .expect("Something went wrong when P2 was receiving the element.")[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (fuel, rand) = p2_prepros.for_sharing.get_own();
        let (elm2_2_v, correction_v) = create_shares(&[elm2], fuel, rand, &p2_context.params)
            .expect("Something went wrong when P2 was sending the element.");
        let (elm2_2, correction) = (elm2_2_v[0], correction_v[0]);

        let elm2_1 = create_foreign_share(
            &[correction],
            p1_prepros.for_sharing.get_fuel_vec_mut(Id(1)),
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
        let (p1_context, p2_context, secret_values) = dummie_preproc();

        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed;
        let elm1 = F::from_u128(56u128);
        let (fuel, rand) = p1_prepros.for_sharing.get_own();
        let (elm1_1, correction) = match create_shares(&[elm1], fuel, rand, &p1_context.params) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed;
        let elm1_2 = match create_foreign_share(
            &[correction],
            p2_prepros.for_sharing.get_fuel_vec_mut(Id(0)),
            &p2_context.params,
        ) {
            Ok(s) => s[0],
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (fuel, rand) = p2_prepros.for_sharing.get_own();
        let (elm2_2, correction) = match create_shares(&[elm2], fuel, rand, &p2_context.params) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let elm2_1 = match create_foreign_share(
            &[correction],
            p1_prepros.for_sharing.get_fuel_vec_mut(Id(1)),
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
        let (p1_context, p2_context, secret_values) = dummie_preproc();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed;
        let (fuel, rand) = p1_prepros.for_sharing.get_own();
        let elm1 = F::from_u128(56u128);
        let (elm1_1, correction) = match create_shares(&[elm1], fuel, rand, &p1_context.params) {
            Ok((e, c)) => (e[0], c[0]),
            Err(e) => {
                println!("Error: {}", e);
                panic!()
            }
        };

        let mut p2_prepros = p2_context.preprocessed;
        let elm1_2 = match create_foreign_share(
            &[correction],
            p2_prepros.for_sharing.get_fuel_vec_mut(Id(0)),
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
        let (mut p1_context, mut p2_context, secret_values) = dummie_preproc();

        // P1 shares an element
        let p1_prepros = &mut p1_context.preprocessed;
        let elm1 = F::from_u128(56u128);
        let (fuel, rand) = p1_prepros.for_sharing.get_own();
        let (elm1_1, correction) = match create_shares(&[elm1], fuel, rand, &p1_context.params) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let p2_prepros = &mut p2_context.preprocessed;
        let elm1_2 = match create_foreign_share(
            &[correction],
            p2_prepros.for_sharing.get_fuel_vec_mut(Id(0)),
            &p2_context.params,
        ) {
            Ok(s) => s[0],
            Err(_) => panic!(),
        };
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // P2 shares an element
        let elm2 = F::from_u128(18u128);
        let (fuel, rand) = p2_prepros.for_sharing.get_own();
        let (elm2_2, correction) = match create_shares(&[elm2], fuel, rand, &p2_context.params) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let elm2_1 = match create_foreign_share(
            &[correction],
            p1_prepros.for_sharing.get_fuel_vec_mut(Id(1)),
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
            let res_share_result = secret_mult(
                s1,
                s2,
                &mut context.preprocessed.triplets,
                &context.params,
                &mut context.opened_values,
                &mut network,
            )
            .await;
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
            assert!(check_all_d(&context.opened_values, &mut network, F::random(rng)).await);
            context.opened_values.clear();

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
        let elm1_v = [elm1_1, elm1_2];
        let elm2_v = [elm2_1, elm2_2];
        let mut context = vec![p1_context, p2_context];
        for (i, network) in cluster.into_iter().enumerate() {
            let context_here = context.pop().unwrap();
            taskset.spawn(do_mpc(
                network,
                elm1_v[i],
                elm2_v[i],
                context_here,
                expected_res,
            ));
        }
        while let Some(res) = taskset.join_next().await {
            res.unwrap();
        }
    }

    #[test]
    fn test_add_with_public_constant() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_preproc();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed;
        let elm1 = F::from_u128(56u128);
        let (fuel, rand) = p1_prepros.for_sharing.get_own();
        let (elm1_1, correction) = match create_shares(&[elm1], fuel, rand, &p1_context.params) {
            Ok((e, c)) => (e[0], c[0]),
            Err(_) => panic!(),
        };

        let mut p2_prepros = p2_context.preprocessed;
        let elm1_2 = match create_foreign_share(
            &[correction],
            p2_prepros.for_sharing.get_fuel_vec_mut(Id(0)),
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
            Id(0) == p1_context.params.who_am_i,
            &p1_context.params,
        );

        let elm3_2 = elm1_2.add_public(
            pub_constant,
            Id(0) == p2_context.params.who_am_i,
            &p2_context.params,
        );

        assert!(elm1 + pub_constant == elm3_1.val + elm3_2.val);
        assert!(elm3_1.mac + elm3_2.mac == (elm1 + pub_constant) * secret_values.mac_key);
    }

    #[test]
    fn test_sub_with_public_constant() {
        type F = Element32;
        let (p1_context, p2_context, secret_values) = dummie_preproc();
        let pub_constant = F::from_u128(8711u128);
        // P1 shares an element
        let mut p1_prepros = p1_context.preprocessed;
        let elm1 = F::from_u128(56u128);
        let (fuel, rand) = p1_prepros.for_sharing.get_own();
        let (elm1_1, correction) = {
            let (e, c) = create_shares(&[elm1], fuel, rand, &p1_context.params).unwrap();
            (e[0], c[0])
        };

        let mut p2_prepros = p2_context.preprocessed;
        let elm1_2 = create_foreign_share(
            &[correction],
            p2_prepros.for_sharing.get_fuel_vec_mut(Id(0)),
            &p2_context.params,
        )
        .unwrap()[0];
        assert!((elm1_1.val + elm1_2.val) == elm1);
        assert!(elm1_1.mac + elm1_2.mac == elm1 * secret_values.mac_key);

        // Adding with pub_constant
        let elm3_1 = elm1_1.sub_public(
            pub_constant,
            Id(0) == p1_context.params.who_am_i,
            &p1_context.params,
        );

        let elm3_2 = elm1_2.sub_public(
            pub_constant,
            Id(0) == p2_context.params.who_am_i,
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
        let (mut contexts, _secret_values) = preprocessing::dealer_preproc(
            rng,
            &known_to_each,
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
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(0),
                &mut network,
            )
            .await;
            let val_p1_1 = val_p1_1_res.expect("Something went wrong in sharing elm_p1_1")[0];

            // P2 sharing a value: val_p2
            let val_p2_res = share(
                values[1].clone(),
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(1),
                &mut network,
            )
            .await;
            let val_p2 = val_p2_res.expect("Something went wrong in sharing elm_p2")[0];

            // multiplying val_p1_1 and val_p2: val_3
            let val_3_res = secret_mult(
                val_p1_1,
                val_p2,
                &mut context.preprocessed.triplets,
                &context.params,
                &mut context.opened_values,
                &mut network,
            )
            .await;
            let val_3 = val_3_res.expect("Something went wrong in multiplication");
            assert!(context.opened_values.len() == 2); // Each multiplication needs partial opening of two elements.

            // P1 sharing a value: val_p1_2
            let val_p1_2_res = share(
                values[2].clone(),
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(0),
                &mut network,
            )
            .await;
            let val_p1_2 = val_p1_2_res.expect("Something went wrong in sharing elm_p1_2")[0];

            // Adding val_3 and val_p1_2: val_4
            let val_4 = val_3 + val_p1_2;

            // Adding val_4 with public constant const_1: val_5
            let const_1 = constant;
            let val_5 =
                val_4.add_public(const_1, context.params.who_am_i == Id(0), &context.params);
            // Checking all partially opened values
            let mut rng = rand_chacha::ChaCha20Rng::from_entropy();
            let random_element = F::random(&mut rng);
            assert!(check_all_d(&context.opened_values, &mut network, random_element).await);
            context.opened_values.clear();

            // opening(and checking) val_5
            let res = open_res(val_5, &mut network, &context.params, &context.opened_values).await;
            assert!(res == F::from_u128(20u128));
        }
        let mut taskset = tokio::task::JoinSet::new();
        let cluster = InMemoryNetwork::in_memory(number_of_parties); //asuming two players
        contexts.reverse();
        values_both.reverse();
        for (i, network) in cluster.into_iter().enumerate() {
            taskset.spawn(do_mpc(
                network,
                contexts.pop().unwrap(),
                values_both.pop().unwrap(),
                constants[i],
            ));
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
        let (mut contexts, _secret_values) = preprocessing::dealer_preproc(
            rng,
            &known_to_each,
            number_of_triplets,
            number_of_parties,
        );

        let val_p1_1 = F::from_u128(2u128);
        let val_p1_2 = F::from_u128(3u128);
        let mut values_both = vec![Some(vec![val_p1_1, val_p1_2]), None];
        let values_for_checking = vec![val_p1_1, val_p1_2];
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
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(0),
                &mut network,
            )
            .await
            .expect("this is a test and the values are there");

            let opened_res = open_res_many(
                val_p1_res,
                &mut network,
                &context.params,
                &context.opened_values,
                random_value,
            )
            .await;
            let vals_zip = opened_res.iter().zip(vals_for_checking);
            for (res, check_res) in vals_zip {
                assert!(*res == check_res);
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
        let (mut contexts, _secret_values) = preprocessing::dealer_preproc(
            rng,
            &known_to_each,
            number_of_triplets,
            number_of_parties,
        );

        let val_p1_1 = F::from_u128(2u128);
        let val_p1_2 = F::from_u128(3u128);
        let mut values_both = vec![
            vec![Some(vec![val_p1_1, val_p1_2]), None],
            vec![None, Some(vec![val_p1_1, val_p1_2])],
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
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(0),
                &mut network,
            )
            .await
            .expect("this is a test, the value is there");

            let val_2 = val_p1_res
                .pop()
                .expect("this is a test, the value is there");
            let val_1 = val_p1_res
                .pop()
                .expect("this is a test, the value is there");

            let val_3_res = secret_mult(
                val_1,
                val_2,
                &mut context.preprocessed.triplets,
                &context.params,
                &mut context.opened_values,
                &mut network,
            )
            .await;
            let val_3 = val_3_res;
            assert!(val_3.is_err());
            match val_3 {
                Ok(_) => panic!("can't happen, we just checked"),
                Err(e) => println!("Error: {}", e),
            };

            let val_p2_res = share(
                values_2,
                &mut context.preprocessed.for_sharing,
                &context.params,
                Id(1),
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
    fn test_power() {
        type F = Element32;
        let a: F = F::from_u128(12u128);
        assert!(a == power(&a, 1));
        let aa = a * a;
        assert!(aa == power(&a, 2));
        let aaa = a * a * a;
        assert!(aaa == power(&a, 3));
        let aaaa = a * a * a * a;
        assert!(aaaa == power(&a, 4));
        let aaaaa = a * a * a * a * a;
        assert!(aaaaa == power(&a, 5));
        let aaaaaa = a * a * a * a * a * a;
        assert!(aaaaaa == power(&a, 6));
    }

    #[test]
    fn test_dealer_writing_to_file() {
        // preprosessing by dealer
        type F = Element32;
        let mut files = [tempfile::tempfile().unwrap(), tempfile::tempfile().unwrap()];
        let known_to_each = vec![1, 2];
        let number_of_triplets = 2;
        preprocessing::write_context(
            &mut files,
            known_to_each,
            number_of_triplets,
            F::from_u128(0u128),
        )
        .unwrap();
        files[0].rewind().unwrap();
        files[1].rewind().unwrap();
        let p1_context: SpdzContext<F> = preprocessing::load_context(&mut files[0]);
        let p2_context: SpdzContext<F> = preprocessing::load_context(&mut files[1]);
        // unpacking
        let p1_params = p1_context.params;
        let p2_params = p2_context.params;
        let mac = p1_params.mac_key_share + p2_params.mac_key_share;
        let mut p1_preprocessed = p1_context.preprocessed;
        let mut p2_preprocessed = p2_context.preprocessed;
        let p1_known_to_pi = p1_preprocessed.for_sharing.bad_habits();
        let p2_known_to_pi = p2_preprocessed.for_sharing.bad_habits();
        let p1_known_to_me = p1_preprocessed.for_sharing.my_randomness;
        let p2_known_to_me = p2_preprocessed.for_sharing.my_randomness;
        let p1_triplet_1 = p1_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");
        let p2_triplet_1 = p2_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");
        let p1_triplet_2 = p1_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");
        let p2_triplet_2 = p2_preprocessed
            .triplets
            .get_triplet()
            .expect("This is a test, the triplet is there.");

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

        let a1 = p1_triplet_1.shares.0 + p2_triplet_1.shares.0;
        let b1 = p1_triplet_1.shares.1 + p2_triplet_1.shares.1;
        let c1 = p1_triplet_1.shares.2 + p2_triplet_1.shares.2;

        assert!(a1.val * b1.val == c1.val);
        assert!(a1.val * mac == a1.mac);
        assert!(b1.val * mac == b1.mac);
        assert!(c1.val * mac == c1.mac);

        let a2 = p1_triplet_2.shares.0 + p2_triplet_2.shares.0;
        let b2 = p1_triplet_2.shares.1 + p2_triplet_2.shares.1;
        let c2 = p1_triplet_2.shares.2 + p2_triplet_2.shares.2;

        assert!(a2.val * b2.val == c2.val);
        assert!(a2.val * mac == a2.mac);
        assert!(b2.val * mac == b2.mac);
        assert!(c2.val * mac == c2.mac);
    }
    // TODO: test errors - in general test that stuff fails when it has to.
}
