// Preprocessing

use crate::schemes::spdz::{self, SpdzContext};
use bincode;
use ff::PrimeField;
use rand::SeedableRng;
use std::{
    error::Error,
    fmt,
    fs::File,
    io::{self, Read, Write},
    path::Path,
};
#[derive(Debug, Clone, PartialEq)]
pub enum MissingPreProcErrorType {
    MissingTriplet,
    MissingForSharingElement,
}
#[derive(Debug, Clone)]
pub struct MissingPreProcError {
    pub e_type: MissingPreProcErrorType,
} // TODO: consider ways to show what type of preproc there is missing

impl fmt::Display for MissingPreProcError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.e_type == MissingPreProcErrorType::MissingTriplet {
            write!(f, "Not enough preprocessed triplets.")
        } else if self.e_type == MissingPreProcErrorType::MissingForSharingElement {
            write!(f, "Not enough pre shared random elements.")
        } else {
            write!(f, "Not enough preprocessing available")
        }
    }
}
impl Error for MissingPreProcError {}

// ToDo: we should probably make getters for all the fields, and make them private, spdz needs to use the values, but not alter them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PreprocessedValues<F: PrimeField> {
    pub triplets: Triplets<F>,
    pub for_sharing: ForSharing<F>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForSharing<F: PrimeField> {
    pub rand_known_to_i: RandomKnownToPi<F>, // consider boxed slices for the outer vec
    pub rand_known_to_me: RandomKnownToMe<F>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Triplets<F: PrimeField> {
    multiplication_triplets: Vec<MultiplicationTriple<F>>,
}

impl<F: PrimeField> Triplets<F> {
    pub fn get_triplet(&mut self) -> Result<MultiplicationTriple<F>, MissingPreProcError> {
        self.multiplication_triplets
            .pop()
            .ok_or(MissingPreProcError {
                e_type: MissingPreProcErrorType::MissingTriplet,
            })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiplicationTriple<F: PrimeField> {
    pub a: spdz::Share<F>,
    pub b: spdz::Share<F>,
    pub c: spdz::Share<F>,
}

pub fn make_multiplicationtriplet<F: PrimeField>(
    a: spdz::Share<F>,
    b: spdz::Share<F>,
    c: spdz::Share<F>,
) -> MultiplicationTriple<F> {
    MultiplicationTriple { a, b, c }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomKnownToPi<F: PrimeField> {
    pub shares: Vec<Vec<spdz::Share<F>>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomKnownToMe<F: PrimeField> {
    pub vals: Vec<F>,
}

pub struct SecretValues<F> {
    pub mac_key: F,
}
pub fn write_preproc_to_file<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    file_names: &[&Path],
    known_to_each: Vec<usize>,
    number_of_triplets: usize,
    _: F,
) -> Result<(), Box<dyn Error>> {
    let number_of_parties = file_names.len();
    assert!(number_of_parties == known_to_each.len());
    let rng = rand_chacha::ChaCha20Rng::from_entropy();
    // Notice here that the secret values are not written to the file, No party is allowed to know the value.
    let (contexts, _): (Vec<SpdzContext<F>>, _) =
        dealer_prepross(rng, known_to_each, number_of_triplets, number_of_parties);
    let names_and_contexts = file_names.iter().zip(contexts);
    for (name, context) in names_and_contexts {
        let data: Vec<u8> = bincode::serialize(&context)?;
        let mut file = File::create(name)?;
        file.write_all(&data)?;
        file.sync_all()?;
    }
    Ok(())
}


pub fn read_preproc_from_file<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
    file_name: &Path,
) -> SpdzContext<F> {
    // TODO: return Result instead.
    let mut data = Vec::new();
    let mut file = File::open(file_name).expect("open file");
    file.read_to_end(&mut data).expect("read to end");
    let new_context: SpdzContext<F> = bincode::deserialize(&data).expect("deserialize");
    new_context
}


pub fn dealer_prepross<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned>(
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
        .map(|i| SpdzContext::empty(number_of_parties, mac_keys[i], i))
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
                contexts[i]
                    .preprocessed_values
                    .for_sharing
                    .rand_known_to_i
                    .shares[me]
                    .push(ri_share);
                if me == i {
                    contexts[me]
                        .preprocessed_values
                        .for_sharing
                        .rand_known_to_me
                        .vals
                        .push(r);
                }
            }
            let r2 = ri_rest;
            //let r_mac_2 = r_mac;
            let r2_share = spdz::Share {
                val: r2,
                mac: r_mac,
            };
            contexts[number_of_parties - 1]
                .preprocessed_values
                .for_sharing
                .rand_known_to_i
                .shares[me]
                .push(r2_share);
            if me == number_of_parties - 1 {
                contexts[me]
                    .preprocessed_values
                    .for_sharing
                    .rand_known_to_me
                    .vals
                    .push(r);
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
            let triplet = make_multiplicationtriplet(ai_share, bi_share, ci_share);

            contexts[i]
                .preprocessed_values
                .triplets
                .multiplication_triplets
                .push(triplet);
        }
        let ai_share = spdz::Share { val: a, mac: a_mac };
        let bi_share = spdz::Share { val: b, mac: b_mac };
        let ci_share = spdz::Share { val: c, mac: c_mac };
        let triplet = make_multiplicationtriplet(ai_share, bi_share, ci_share);

        contexts[number_of_parties - 1]
            .preprocessed_values
            .triplets
            .multiplication_triplets
            .push(triplet)
    }

    // TODO: The secret values are only there for testing/ develompent purposes, consider removing it.
    let secret_values = SecretValues { mac_key };
    (contexts, secret_values)
}

impl<F: PrimeField + serde::Serialize + serde::de::DeserializeOwned> SpdzContext<F> {
    fn empty(number_of_parties: usize, mac_key_share: F, who_am_i: usize) -> Self {
        generate_empty_context(number_of_parties, mac_key_share, who_am_i)
    }

    pub fn from_file(path: &Path) -> Result<Self, io::Error> {
        Ok(read_preproc_from_file(path))
    }
}

fn generate_empty_context<F: PrimeField>(
    number_of_parties: usize,
    mac_key_share: F,
    who_am_i: usize,
) -> SpdzContext<F> {
    let rand_known_to_i = RandomKnownToPi {
        shares: vec![vec![]; number_of_parties],
    };
    let rand_known_to_me = RandomKnownToMe { vals: vec![] };
    let triplets = Triplets {
        multiplication_triplets: vec![],
    };
    let p_preprosvals = PreprocessedValues {
        triplets,
        for_sharing: ForSharing {
            rand_known_to_i,
            rand_known_to_me,
        },
    };
    SpdzContext {
        opened_values: vec![],
        params: spdz::SpdzParams {
            mac_key_share,
            who_am_i,
        },
        preprocessed_values: p_preprosvals,
    }
}
