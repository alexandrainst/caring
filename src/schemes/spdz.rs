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

use crate::{net::agency::Broadcast, protocols::cointoss::CoinToss};

// Should we allow Field or use PrimeField?
#[derive(Debug, Clone, Copy, Add, Sub, AddAssign, SubAssign, serde::Serialize, serde::Deserialize)]
pub struct Share<F: PrimeField> {
    // This field is nice and I like it
    pub val: F,
    // This field is scary and I don't know how it should be handled
    pub mac: F,
}

impl<F: PrimeField> Share<F> {
    // The share of the mac key is the same for the hole computation, and is only revield in the end. 
    // Therefor it is probably appropriate to place it in the context - that might alredy be the case as there is a field named alfa.
    pub fn add_public(self, val: F, is_chosen_party: bool, mac_key_share: F) -> Self {
        let val_val = if is_chosen_party { val } else { F::ZERO };
        Share {
            val: self.val + val_val,
            mac: self.mac + val_val * mac_key_share,
            ..self
        }
    }

    pub fn sub_public(self, val: F, chosen_one: bool, mac_key_share: F) -> Self {
        let val_val = if chosen_one { val } else { F::ZERO };
        Share {
            val: self.val - val_val,
            mac: self.mac - val_val * mac_key_share,
            ..self
        }
    }
}
// Will validation actually ever be done like this - is it not too expensive? 
// (haing a key for each share?)
// Why not do the validation is bulk?
impl<F: PrimeField> Share<F> {
    pub fn validate(&self, key: F) -> bool {
        let Share { val, mac } = *self;
        val * key == mac
    }
}

// Bad nameing change to "make_share_from_field_element" or something like that.
// This needs to be changed. If We use only one mac_key, the mac needs to add up. 
// (remember that this is used for preprosessing, so it can't need prepros'ed values)
pub fn make_random_share<F: PrimeField>(val: F, mac: F) -> Share<F> {
    //Share{val: F::random(&mut rng), mac: F::random(&mut rng)}
    Share{val: val, mac: mac}
}

/// Mutliplication between a share and a public value
///
/// This operation is symmetric
impl<F: PrimeField> std::ops::Mul<F> for Share<F> {
    type Output = Share<F>;

    fn mul(self, rhs: F) -> Self::Output {
        Share {
            val: self.val * rhs,
            mac: self.mac * rhs,
        }
    }
}


//struct SpdzParams<F: PrimeField> {
//    key: F,
//}

// TODO: Implement multiplication between shares. Use triplets.

pub fn share<F: PrimeField>(val: F, n: usize, key: F, mut rng: &mut impl RngCore) -> Vec<Share<F>> {
    // HACK: This is really not secure at all. - why not, because of the way the random values are chosen or is there something else?
    let mut shares: Vec<_> = (0..n).map(|_| F::random(&mut rng)).collect();

    let sum: F = shares.iter().sum();
    shares[0] -= sum - val;

    // This is only possible if the party doing the sharing knows the hole key.
        // Maybe it is a possiblity if we go for [[x]] instead of [x]. 
        // But then all parties who resives a share has to be able multiply it with there mac_key, 
        // and thats not really possible as they only know a share of x. 
    shares
        .into_iter()
        .map(|x| Share {
            val: x,
            mac: key * x,
        })
        .collect()
}

// Sharing using prepros'ed values
pub fn input<F: PrimeField>(_val: F, _n: usize) -> Vec<Share<F>> {
    // 1. Everyone sends party `i` their share (partial opening)
    // 2. Party `i` then broadcasts `x^i - r`.
    //    Party 1 sets their share to `x^i_1 = r1 + x^i - r`.
    //    (In practice this might not always be 1)
    // 3. Profit

    todo!("Implement the function")
}

// when shares are reconstructed, the mac probably also needs to be reconstructed. 
pub fn reconstruct<F: PrimeField>(shares: &[Share<F>]) -> F {
    shares.iter().map(|x| x.val).sum()
}

// IDEA:
//
// Okay hear me out, since we 'have' to check the opened values at some point
// during the computation, it somehow acts as sort of 'release of a resource'.
// As such would it be apt to have the SpdzContext be some sort of manual garbage collector?
    // Well that actually depends on what kind of SPDZ we use. 
    // The kind I was imagining only does the checking in the end, 
    // therefor the keys last until the end.
#[derive(Debug)]
pub struct SpdzContext<F: PrimeField> {
    opened_values: Vec<F>,
    closed_values: Vec<Share<F>>,
    alpha: F, // I want to change the name to mac_key_share - if that is infact what it is.
    // dbgr supplier (det. random bit generator)
    is_chosen_party: bool,
}

// TODO: Convert to associated function?
pub async fn mac_check<Rng: SeedableRng + RngCore, F: PrimeField + Serialize + DeserializeOwned>(
    ctx: &mut SpdzContext<F>,
    cx: &mut impl Broadcast,
) -> Result<(), ()> {
    // TODO: More specific errors
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
    let delta = gamma - ctx.alpha * a;

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

    use super::*;

    #[test]
    fn sharing() {
        use crate::algebra::element::Element32;
        let mut rng = rand::rngs::mock::StepRng::new(42, 7);
        // Key should probablu be derived from a random somewhere
        let key = 7u32.into();
        let v = 42u32;
        let shares = share(Element32::from(v), 3, key, &mut rng);

        let v2: u32 = reconstruct(&shares).into();
        assert_eq!(v, v2);
    }
}
