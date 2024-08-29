// Preprocessing

use crate::{
    net::Id,
    schemes::spdz::{self, SpdzContext},
};
use bincode;
use ff::PrimeField;
use rand::SeedableRng;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    error::Error,
    fmt,
    fs::File,
    io::{self, Write},
    path::Path,
};

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

impl Error for PreProcError {}

// ToDo: we should probably make getters for all the fields, and make them private, spdz needs to use the values, but not alter them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessedValues<F: PrimeField> {
    // change to beaver module
    pub triplets: Triplets<F>,

    pub for_sharing: ForSharing<F>,
}

// TODO: Document this
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForSharing<F: PrimeField> {
    /// Fuel per Party
    pub party_fuel: Vec<Fuel<F>>, // consider boxed slices for the outer vec
    pub my_randomness: Vec<F>,
}

impl<F: PrimeField> ForSharing<F> {
    pub fn bad_habits(&self) -> Vec<Vec<spdz::Share<F>>> {
        self.party_fuel.iter().map(|f| f.shares.clone()).collect()
    }
}

/// Multiplication triplet fuel tank
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Triplets<F: PrimeField> {
    multiplication_triplets: Vec<MultiplicationTriple<F>>,
}

// TODO: Use beaver module
impl<F: PrimeField> Triplets<F> {
    pub fn get_triplet(&mut self) -> Result<MultiplicationTriple<F>, PreProcError> {
        self.multiplication_triplets
            .pop()
            .ok_or(PreProcError::MissingTriplet)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiplicationTriple<F: PrimeField> {
    pub a: spdz::Share<F>,
    pub b: spdz::Share<F>,
    pub c: spdz::Share<F>,
}

impl<F: PrimeField> MultiplicationTriple<F> {
    pub fn new(a: spdz::Share<F>, b: spdz::Share<F>, c: spdz::Share<F>) -> Self {
        MultiplicationTriple { a, b, c }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Fuel<F: PrimeField> {
    pub shares: Vec<spdz::Share<F>>,
}

pub struct SecretValues<F> {
    pub mac_key: F,
}

pub fn write_context<F: PrimeField + Serialize + DeserializeOwned>(
    files: &mut [File],
    known_to_each: Vec<usize>,
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
) -> SpdzContext<F> {
    // TODO: return Result instead.
    let buffered = std::io::BufReader::new(file);
    bincode::deserialize_from(buffered).unwrap()
}

pub async fn load_context_async<F: PrimeField + Serialize + DeserializeOwned>(
    file: &Path,
) -> SpdzContext<F> {
    let contents = tokio::fs::read(file).await.unwrap();
    // TODO: return Result instead.
    bincode::deserialize_from(&*contents).unwrap()
}

pub fn dealer_preproc<F: PrimeField + Serialize + DeserializeOwned>(
    mut rng: impl rand::Rng,
    known_to_each: Vec<usize>,
    number_of_triplets: usize,
    number_of_parties: usize,
) -> (Vec<SpdzContext<F>>, SecretValues<F>) {
    // TODO: tjek that the arguments are consistent
    let mac_keys: Vec<F> = (0..number_of_parties)
        .map(|_| F::random(&mut rng))
        .collect();
    let mut contexts: Vec<SpdzContext<F>> = (0..number_of_parties)
        .map(|i| SpdzContext::empty(number_of_parties, mac_keys[i], Id(i)))
        .collect();
    let mac_key = mac_keys.into_iter().sum();

    // Filling the context
    // First with values used for easy sharing
    for (me, kte) in known_to_each.into_iter().enumerate() {
        for _ in 0..kte {
            let r = F::random(&mut rng);
            let mut r_mac = r * mac_key;
            let mut ri_rest = r;
            for i in 0..number_of_parties - 1 {
                let ri = F::random(&mut rng);
                ri_rest -= ri;
                let r_mac_i = F::random(&mut rng);
                r_mac -= r_mac_i;
                let ri_share = spdz::Share {
                    val: ri,
                    mac: r_mac_i,
                };
                contexts[i].preprocessed.for_sharing.party_fuel[me]
                    .shares
                    .push(ri_share);
                if me == i {
                    contexts[me].preprocessed.for_sharing.my_randomness.push(r);
                }
            }
            let r2 = ri_rest;
            //let r_mac_2 = r_mac;
            let r2_share = spdz::Share {
                val: r2,
                mac: r_mac,
            };
            contexts[number_of_parties - 1]
                .preprocessed
                .for_sharing
                .party_fuel[me]
                .shares
                .push(r2_share);
            if me == number_of_parties - 1 {
                contexts[me].preprocessed.for_sharing.my_randomness.push(r);
            }
        }
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

            contexts[i]
                .preprocessed
                .triplets
                .multiplication_triplets
                .push(triplet);
        }
        let ai_share = spdz::Share { val: a, mac: a_mac };
        let bi_share = spdz::Share { val: b, mac: b_mac };
        let ci_share = spdz::Share { val: c, mac: c_mac };
        let triplet = MultiplicationTriple::new(ai_share, bi_share, ci_share);

        contexts[number_of_parties - 1]
            .preprocessed
            .triplets
            .multiplication_triplets
            .push(triplet)
    }

    // TODO: The secret values are only there for testing/ develompent purposes, consider removing it.
    let secret_values = SecretValues { mac_key };
    (contexts, secret_values)
}

impl<F: PrimeField + Serialize + DeserializeOwned> SpdzContext<F> {
    fn empty(number_of_parties: usize, mac_key_share: F, who_am_i: Id) -> Self {
        generate_empty_context(number_of_parties, mac_key_share, who_am_i)
    }

    pub fn from_file(mut file: File) -> Result<Self, io::Error> {
        Ok(load_context(&mut file))
    }
}

fn generate_empty_context<F: PrimeField>(
    number_of_parties: usize,
    mac_key_share: F,
    who_am_i: Id,
) -> SpdzContext<F> {
    let rand_known_to_i = vec![Fuel { shares: vec![] }; number_of_parties];
    let rand_known_to_me = vec![];
    let triplets = Triplets {
        multiplication_triplets: vec![],
    };
    let p_preprosvals = PreprocessedValues {
        triplets,
        for_sharing: ForSharing {
            party_fuel: rand_known_to_i,
            my_randomness: rand_known_to_me,
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
