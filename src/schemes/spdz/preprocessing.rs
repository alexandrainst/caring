// Preprocessing

use crate::{
    algebra::math::Vector,
    net::Id,
    protocols::beaver,
    schemes::{
        spdz::{self, SpdzContext},
        Reserve,
    },
};
use bincode;
use ff::PrimeField;
use rand::{Rng, SeedableRng};
use serde::{de::DeserializeOwned, Serialize};
use std::{error::Error, fmt, fs::File, io::Write, path::Path};

#[derive(Debug, Clone, PartialEq)]
pub enum PreProcError {
    MissingTriplet,
    MissingForSharingElement,
}

impl fmt::Display for PreProcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PreProcError::MissingTriplet => {
                write!(f, "Not enough preprocessed triplets.")
            }
            PreProcError::MissingForSharingElement => {
                write!(f, "Not enough pre shared random elements.")
            }
        }
    }
}

impl<F: ff::PrimeField> Reserve for spdz::SpdzContext<F> {
    fn reserve(&mut self, amount: usize) -> Self {
        fn inner<T>(list: &mut Vec<T>, n: usize) -> Vec<T> {
            let mut new = list.split_off(n);
            std::mem::swap(&mut new, list);
            new
        }

        let reserved = inner(
            &mut self.preprocessed.triplets.multiplication_triplets,
            amount,
        );

        let party_fuel = {
            let old = &mut self.preprocessed.for_sharing.party_fuel;
            old.iter_mut()
                .map(|tank| FuelTank {
                    party: tank.party,
                    shares: inner(&mut tank.shares, amount),
                })
                .collect()
        };
        let preprocessed = PreprocessedValues {
            triplets: Triplets {
                multiplication_triplets: reserved,
            },
            for_sharing: PreShareTank {
                party_fuel,
                my_randomness: inner(&mut self.preprocessed.for_sharing.my_randomness, amount),
                me: self.preprocessed.for_sharing.me,
            },
        };
        Self {
            opened_values: vec![],
            params: self.params.clone(),
            preprocessed,
        }
    }

    fn put_back(&mut self, mut other: Self) {
        self.opened_values.append(&mut other.opened_values);
        self.preprocessed
            .triplets
            .multiplication_triplets
            .append(&mut other.preprocessed.triplets.multiplication_triplets);
        self.preprocessed
            .for_sharing
            .my_randomness
            .append(&mut other.preprocessed.for_sharing.my_randomness);
        self.preprocessed
            .for_sharing
            .party_fuel
            .iter_mut()
            .zip(other.preprocessed.for_sharing.party_fuel)
            .for_each(|(tank, mut old)| tank.shares.append(&mut old.shares));
    }
}

impl Error for PreProcError {}

// ToDo: we should probably make getters for all the fields, and make them private, spdz needs to use the values, but not alter them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessedValues<F: PrimeField> {
    // change to beaver module
    pub triplets: Triplets<F>,

    pub for_sharing: PreShareTank<F>,
}

// TODO: Document this
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreShareTank<F: PrimeField> {
    /// Fuel per Party
    pub party_fuel: Vec<FuelTank<F>>, // consider boxed slices for the outer vec
    pub my_randomness: Vec<F>,
    pub me: Id,
}

impl<F: PrimeField> PreShareTank<F> {
    pub fn empty(me: Id, party_size: usize) -> Self {
        let party_fuel = (0..party_size).map(|id| FuelTank::empty(Id(id))).collect();
        Self {
            party_fuel,
            my_randomness: vec![],
            me,
        }
    }

    pub fn get_fuel_vec_mut(&mut self, id: Id) -> &mut Vec<spdz::Share<F>> {
        &mut self.party_fuel[id.0].shares
    }

    pub fn get_fuel_mut(&mut self, id: Id) -> &mut FuelTank<F> {
        &mut self.party_fuel[id.0]
    }

    pub fn get_own(&mut self) -> (&mut FuelTank<F>, &mut Vec<F>) {
        (&mut self.party_fuel[self.me.0], &mut self.my_randomness)
    }

    pub fn split(
        &mut self,
    ) -> (
        &mut FuelTank<F>,
        &mut Vec<F>,
        impl IntoIterator<Item = &mut FuelTank<F>>,
    ) {
        let (head, tail) = self.party_fuel.split_at_mut(self.me.0);
        let (my_fueltank, tail) = tail.split_first_mut().unwrap();
        let others = head.iter_mut().chain(tail);
        debug_assert_eq!(self.me, my_fueltank.party);

        (my_fueltank, &mut self.my_randomness, others)
    }
}

impl<F: PrimeField> PreShareTank<F> {
    pub fn bad_habits(&self) -> Vec<Vec<spdz::Share<F>>> {
        self.party_fuel.iter().map(|f| f.shares.clone()).collect()
    }
}

/// Multiplication triplet fuel tank
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Triplets<F: PrimeField> {
    pub(crate) multiplication_triplets: Vec<MultiplicationTriple<F>>,
}

// TODO: Use beaver module
impl<F: PrimeField> Triplets<F> {
    pub fn get_triplet(&mut self) -> Result<MultiplicationTriple<F>, PreProcError> {
        self.multiplication_triplets
            .pop()
            .ok_or(PreProcError::MissingTriplet)
    }
}

pub type MultiplicationTriple<F> = beaver::BeaverTriple<spdz::Share<F>>;

impl<F: PrimeField> MultiplicationTriple<F> {
    pub fn new(a: spdz::Share<F>, b: spdz::Share<F>, c: spdz::Share<F>) -> Self {
        MultiplicationTriple { shares: (a, b, c) }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FuelTank<F: PrimeField> {
    pub party: Id,
    pub shares: Vec<spdz::Share<F>>,
}

impl<F: PrimeField> FuelTank<F> {
    pub fn empty(id: Id) -> Self {
        Self {
            party: id,
            shares: Vec::new(),
        }
    }

    pub fn subtank(&mut self, take: usize) -> Self {
        let len = self.shares.len();
        let shares = self.shares.drain(len - take..).collect();
        Self {
            party: self.party,
            shares,
        }
    }
}

pub struct SecretValues<F> {
    pub mac_key: F,
}

pub fn write_context<F: PrimeField + Serialize + DeserializeOwned>(
    files: &mut [File],
    known_to_each: &[usize],
    number_of_triplets: usize,
    _: F,
) -> Result<(), Box<dyn Error>> {
    let number_of_parties = files.len();
    assert!(number_of_parties == known_to_each.len());
    let rng = rand_chacha::ChaCha20Rng::from_entropy();
    // Notice here that the secret values are not written to the file, No party is allowed to know the value.
    let (contexts, _): (Vec<SpdzContext<F>>, _) =
        dealer_preproc(rng, known_to_each, number_of_triplets, number_of_parties);
    let names_and_contexts = files.iter_mut().zip(contexts);
    for (file, context) in names_and_contexts {
        let data: Vec<u8> = bincode::serialize(&context)?;
        file.write_all(&data)?;
    }
    Ok(())
}

pub fn load_context<F: PrimeField + Serialize + DeserializeOwned>(
    file: &mut File,
) -> Result<SpdzContext<F>, bincode::Error> {
    // TODO: return Result instead.
    let buffered = std::io::BufReader::new(file);
    bincode::deserialize_from(buffered)
}

pub async fn load_context_async<F: PrimeField + Serialize + DeserializeOwned>(
    file: &Path,
) -> Result<SpdzContext<F>, bincode::Error> {
    let contents = tokio::fs::read(file).await.unwrap();
    // TODO: return Result instead.
    bincode::deserialize_from(&*contents)
}

fn dealer_preshares<F: PrimeField>(
    mut rng: impl Rng,
    per_party: &[usize],
    num_of_parties: usize,
) -> (Vec<F>, Vec<PreShareTank<F>>) {
    let mac_keys: Vec<F> = (0..num_of_parties).map(|_| F::random(&mut rng)).collect();
    let mac_key = mac_keys.iter().sum();

    // Filling the context
    // First with values used for easy sharing
    let mut parties: Vec<_> = (0..num_of_parties)
        .map(|id| PreShareTank::empty(Id(id), num_of_parties))
        .collect();
    for (me, &kte) in per_party.iter().enumerate() {
        let r = Vector::from_vec(vec![F::random(&mut rng); kte]);
        let mut r_mac = r.clone() * mac_key;
        let mut ri_rest = r.clone();

        // party_i
        for (i, party) in parties[0..num_of_parties - 1].iter_mut().enumerate() {
            let ri = Vector::from_vec(vec![F::random(&mut rng); kte]);
            ri_rest -= &ri;
            let r_mac_i = Vector::from_vec(vec![F::random(&mut rng); kte]);
            r_mac -= &r_mac_i;

            let mut ri_share: Vec<_> = ri
                .into_iter()
                .zip(r_mac_i)
                .map(|(val, mac)| spdz::Share { val, mac })
                .collect();
            party.party_fuel[me].shares.append(&mut ri_share);
            if me == i {
                let mut r = r.to_vec();
                party.my_randomness.append(&mut r);
            }
        }
        let r2 = ri_rest;
        let mut r2_share: Vec<_> = r2
            .into_iter()
            .zip(r_mac)
            .map(|(val, mac)| spdz::Share { val, mac })
            .collect();
        parties[num_of_parties - 1].party_fuel[me]
            .shares
            .append(&mut r2_share);
        if me == num_of_parties - 1 {
            let mut r = r.to_vec();
            parties[me].my_randomness.append(&mut r);
        }
    }

    (mac_keys, parties)
}

pub fn dealer_preproc<F: PrimeField + Serialize + DeserializeOwned>(
    mut rng: impl rand::Rng,
    known_to_each: &[usize],
    number_of_triplets: usize,
    number_of_parties: usize,
) -> (Vec<SpdzContext<F>>, SecretValues<F>) {
    // TODO: tjek that the arguments are consistent
    let (mac_keys, parties) = dealer_preshares(&mut rng, known_to_each, number_of_parties);

    // TODO: don't recalculate this
    let mac_key = mac_keys.iter().sum();

    let mut ctxs: Vec<SpdzContext<F>> = (0..number_of_parties)
        .map(|i| SpdzContext::empty(number_of_parties, mac_keys[i], Id(i)))
        .collect();

    for (ctx, for_sharing) in ctxs.iter_mut().zip(parties.into_iter()) {
        ctx.preprocessed.for_sharing = for_sharing
    }

    // Now filling in triplets

    for _ in 0..number_of_triplets {
        let mut a = F::random(&mut rng);
        let mut b = F::random(&mut rng);
        let mut c = a * b;

        let mut a_mac = a * mac_key;
        let mut b_mac = b * mac_key;
        let mut c_mac = c * mac_key;

        #[allow(clippy::needless_range_loop)]
        for i in 0..number_of_parties - 1 {
            let ai = F::random(&mut rng);
            let bi = F::random(&mut rng);
            let ci = F::random(&mut rng);
            a -= ai;
            b -= bi;
            c -= ci;

            let a_mac_i = F::random(&mut rng);
            let b_mac_i = F::random(&mut rng);
            let c_mac_i = F::random(&mut rng);
            a_mac -= a_mac_i;
            b_mac -= b_mac_i;
            c_mac -= c_mac_i;

            let ai_share = spdz::Share {
                val: ai,
                mac: a_mac_i,
            };
            let bi_share = spdz::Share {
                val: bi,
                mac: b_mac_i,
            };
            let ci_share = spdz::Share {
                val: ci,
                mac: c_mac_i,
            };
            let triplet = MultiplicationTriple::new(ai_share, bi_share, ci_share);

            ctxs[i]
                .preprocessed
                .triplets
                .multiplication_triplets
                .push(triplet);
        }
        let ai_share = spdz::Share { val: a, mac: a_mac };
        let bi_share = spdz::Share { val: b, mac: b_mac };
        let ci_share = spdz::Share { val: c, mac: c_mac };
        let triplet = MultiplicationTriple::new(ai_share, bi_share, ci_share);

        ctxs[number_of_parties - 1]
            .preprocessed
            .triplets
            .multiplication_triplets
            .push(triplet)
    }

    // TODO: The secret values are only there for testing/ develompent purposes, consider removing it.
    let secret_values = SecretValues { mac_key };
    (ctxs, secret_values)
}

impl<F: PrimeField + Serialize + DeserializeOwned> SpdzContext<F> {
    fn empty(number_of_parties: usize, mac_key_share: F, who_am_i: Id) -> Self {
        generate_empty_context(number_of_parties, mac_key_share, who_am_i)
    }

    pub fn from_file(mut file: File) -> Result<Self, bincode::Error> {
        load_context(&mut file)
    }
}

fn generate_empty_context<F: PrimeField>(
    number_of_parties: usize,
    mac_key_share: F,
    who_am_i: Id,
) -> SpdzContext<F> {
    let rand_known_to_i = (0..number_of_parties)
        .map(|id| FuelTank::empty(Id(id)))
        .collect();
    let rand_known_to_me = vec![];
    let triplets = Triplets {
        multiplication_triplets: vec![],
    };
    let p_preprosvals = PreprocessedValues {
        triplets,
        for_sharing: PreShareTank {
            party_fuel: rand_known_to_i,
            my_randomness: rand_known_to_me,
            me: who_am_i,
        },
    };
    SpdzContext {
        opened_values: vec![],
        params: spdz::SpdzParams {
            mac_key_share,
            who_am_i,
        },
        preprocessed: p_preprosvals,
    }
}
