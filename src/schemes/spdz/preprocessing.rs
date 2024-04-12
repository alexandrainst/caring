// Preprocessing
// Making triplets
// Making random values known to some specific party
// MAC'ing elements

// To do the preprosessing we need some HE scheme

// This has to be done interactivly - look at triplets for inspiration

use ff::PrimeField;
use rand::RngCore;

use crate::{
    net::agency::Broadcast,
    schemes::spdz::{self, SpdzContext},
};

// ToDo: we should probably make getters for all the fields, and make them private, spdz needs to use the values, but not alter them.
#[derive(Debug)]
pub struct PreprocessedValues<F: PrimeField> {
    pub triplets: Vec<MultiplicationTriple<F>>,
    pub rand_known_to_i: RandomKnownToPi<F>, // consider boxed slices for the outer vec
    pub rand_known_to_me: RandomKnownToMe<F>,
}

#[derive(Debug)]
pub struct MultiplicationTriple<F: PrimeField> {
    //pub shares: (spdz::Share<F>, spdz::Share<F>, spdz::Share<F>),
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

#[derive(Debug)]
pub struct RandomKnownToPi<F: PrimeField> {
    pub shares: Vec<Vec<spdz::Share<F>>>,
}

#[derive(Debug)]
pub struct RandomKnownToMe<F: PrimeField> {
    pub vals: Vec<F>,
}

pub struct SecretValues<F> {
    pub mac_key: F,
    //secret_shared_elements: Vec<F>,
}

// A dealer who is not colluding with either of the other parties.
pub fn dealer_prepross<F: PrimeField>(
    mut rng: rand::rngs::mock::StepRng,
    known_to_each: Vec<usize>,
    number_of_triplets: usize,
    number_of_parties: usize,
) -> (Vec<SpdzContext<F>>, SecretValues<F>) {
    // TODO: tjek that the arguments are consistent
    //type F = Element32;
    // The mac key is secret and only known to the dealer. No party can know this key.
    // Alternativly, if there is no dealer the key can be chosen by letting each party choose there share of the key at random.
    //let mac_key = F::random(&mut rng);
    let mut mac_keys: Vec<F> = vec![];
    // Generating an empty context
    let mut contexts: Vec<SpdzContext<F>> = vec![];
    for i in 0..number_of_parties {
        let rand_known_to_i = RandomKnownToPi {
            shares: vec![vec![]; number_of_parties],
        };
        let rand_known_to_me = RandomKnownToMe { vals: vec![] };
        let triplets = vec![];
        let p_preprosvals = PreprocessedValues {
            triplets,
            rand_known_to_i,
            rand_known_to_me,
        };
        let mac_key_share = F::random(&mut rng);
        mac_keys.push(mac_key_share);
        let p_context = SpdzContext {
            opened_values: vec![],
            closed_values: vec![],
            params: spdz::SpdzParams {
                mac_key_share,
                who_am_i: i,
            },
            preprocessed_values: p_preprosvals,
        };

        contexts.push(p_context);
    }
    let mac_key = mac_keys.into_iter().sum();
    // Filling the context
    // First with values used for easy sharing
    let mut me = 0; // TODO: find a nicer way to have a loop counter.
    for kte in known_to_each {
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
                contexts[i].preprocessed_values.rand_known_to_i.shares[me].push(ri_share);
                if me == i {
                    contexts[me]
                        .preprocessed_values
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
                .rand_known_to_i
                .shares[me]
                .push(r2_share);
            if me == number_of_parties - 1 {
                contexts[me]
                    .preprocessed_values
                    .rand_known_to_me
                    .vals
                    .push(r);
            }
        }
        me += 1;
    }
    // Now filling in triplets
    for _ in 0..number_of_triplets {
        let mut a = F::random(&mut rng);
        let mut b = F::random(&mut rng);
        let mut c = a * b;

        let mut a_mac = a * mac_key;
        let mut b_mac = b * mac_key;
        let mut c_mac = c * mac_key;

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

            contexts[i].preprocessed_values.triplets.push(triplet);
        }
        let ai_share = spdz::Share { val: a, mac: a_mac };
        let bi_share = spdz::Share { val: b, mac: b_mac };
        let ci_share = spdz::Share { val: c, mac: c_mac };
        let triplet = make_multiplicationtriplet(ai_share, bi_share, ci_share);

        contexts[number_of_parties - 1]
            .preprocessed_values
            .triplets
            .push(triplet)
    }

    let secret_values = SecretValues { mac_key };
    (contexts, secret_values)
}
