use std::{net::SocketAddr, time::Duration};
use caring::{net::network::TcpNetwork, schemes::{feldman, shamir::ShamirParams, spdz::{self, preprocessing}}};
use rand::thread_rng;
use std::path::Path;

pub struct AdderEngine {
    network: TcpNetwork,
    runtime: tokio::runtime::Runtime,
    threshold: u64,
    context: spdz::SpdzContext<curve25519_dalek::Scalar>,
}

impl AdderEngine {
    pub fn shutdown(self) {
        let AdderEngine { network, runtime, .. } = self;
        runtime.spawn(network.shutdown());
        runtime.shutdown_timeout(Duration::from_secs(5));
    }
}

type SignedFix = fixed::FixedI128<64>;
fn to_offset(num: f64) -> u128 {
    let num: i128 = SignedFix::from_num(num).to_bits(); // convert to signed fixed point
                                                        // Okay this a some voodoo why this 'just works', but the idea is that
                                                        // twos complement is actually just an offset from the max value.
                                                        // https://en.wikipedia.org/wiki/Offset_binary
    num as u128
}

fn from_offset(num: u128) -> f64 {
    // Same applies as above
    let num = SignedFix::from_bits(num as i128);
    num.to_num()
}

// We start allowing just one element.
pub fn mpc_sum(engine: &mut AdderEngine, nums: &[f64]) -> Option<Vec<f64>> {
    let AdderEngine {
        network,
        runtime,
        threshold,
        context,
    } = engine;
    let nums: Vec<_> = nums
        .iter()
        .map(|&num| {
            let num = to_offset(num);
            curve25519_dalek::Scalar::from(num)
        })
        .collect();

    let res : Option<_> = runtime.block_on(async {
        // construct
        let parties: Vec<_> = network
            .participants()
            .map(|id| (id + 1))
            .map(curve25519_dalek::Scalar::from)
            .collect();


        let number_of_parties = parties.len();
        let val = curve25519_dalek::Scalar::from(nums[0]);
        let who_am_i = context.params.who_am_i();
        let mut shares = vec![];
        for i in 0..number_of_parties{
            if i == who_am_i {
                shares.push(spdz::share(Some(val), &mut context.preprocessed_values.for_sharing, i, &context.params, network).await.expect("TODO: look at error")); 
            } else {
                shares.push(spdz::share(None, &mut context.preprocessed_values.for_sharing, i, &context.params, network).await.expect("TODO: look at error")); 
            }

        }
        // TODO: we might need a share_many and a open_res_many that works vectorized... 

        // compute
        //let my_result: curve25519_dalek::Scalar = shares.into_iter().sum();
        let mut my_result = shares.pop().unwrap();
        while shares.len() > 0{
            my_result += shares.pop().unwrap()
        }

        // reconstruct
        //assert!(spdz::check_all_d(context.open_values, network, random_element).await); - there have been no partial openings. 
        let res = from_offset(
            u128::from_le_bytes(
                spdz::open_res(my_result, network, &context.params, &context.opened_values).await
                    .as_bytes()[0..128/8]
                    .try_into().expect("convertion between types should go well"
                )
            )
        );
        //let res = feldman::reconstruct_many(&ctx, &open_shares).expect("Failed to validate")
            //.into_iter()
            //.map(|x| x.as_bytes()[0..128/8].try_into().expect("Should be infalliable"))
            //.map(u128::from_le_bytes)
            //.map(from_offset)
            //.collect();
        // NOTE: Since we are only using half of this space, we have
        // a possibility of 'checking' for computation failures.

        Some(vec![f64::from(res)])
    });
    res
}


#[derive(Debug)]
pub struct MpcError(pub &'static str);

impl std::fmt::Display for MpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = self.0;
        write!(f, "Bad thing happened: {msg}")
    }
}

impl std::error::Error for MpcError {}

//pub fn setup_engine(my_addr: &str, others: &[impl AsRef<str>], file_name: String) -> Result<AdderEngine<F>, MpcError> {
pub fn setup_engine(my_addr: &str, others: &[impl AsRef<str>], file_name: &Path) -> Result<AdderEngine, MpcError> {
    let my_addr: SocketAddr = my_addr.parse().unwrap();
    let others: Vec<SocketAddr> = others.iter().map(|s| s.as_ref().parse().unwrap()).collect();

    let threshold = ((others.len() + 1) / 2 + 1) as u64;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let network = TcpNetwork::connect(my_addr, &others);
    let network = runtime.block_on(network).map_err(|_| MpcError("Failed to setup network"))?;
    let mut context = preprocessing::read_preproc_from_file(file_name);
    // Notice: This is a hack and only works as long as the parties share the same number of elements. 
    // To make a propper solotion, the id must be known befor the preprocessing is made.
    // To ensure that the right number of elements are made for each party.
    context.params.who_am_i = network.index;
    let engine = AdderEngine {
        network,
        runtime,
        threshold,
        context,
    };
    Ok(engine)
}

pub fn do_preproc(){
    let file_names = vec![Path::new("src/context1.bin"), Path::new("src/context2.bin")];
    let known_to_each = vec![2, 2];
    let number_of_triplets = 2;
    let num = to_offset(0.0);
    preprocessing::write_preproc_to_file(
        file_names,
        known_to_each,
        number_of_triplets,
        curve25519_dalek::Scalar::from(num),
    ).unwrap();
}

#[cfg(test)]
mod test {
    use std::time::Duration;

    use super::*;

    #[test]
    fn offset_binary() {
        use curve25519_dalek::Scalar;
        let a: Scalar = to_offset(1.1).into();
        let b: Scalar = to_offset(2.2).into();
        let c: Scalar = to_offset(3.3).into();
        let sum = a + b + c;
        let sum: [u8; 16] = sum.as_bytes()[0..16].try_into().unwrap();
        let sum = u128::from_le_bytes(sum);
        let sum = from_offset(sum);
        assert!(sum - 6.6 < 0.01);
    }

    #[test]
    fn sunshine() {
        use std::thread;
        do_preproc();
        let t1 = thread::spawn(|| {
            println!("[1] Setting up...");
            let mut engine = setup_engine("127.0.0.1:1234", &["127.0.0.1:1235"], Path::new("src/context1.bin") ).unwrap();
            println!("[1] Ready");
            let res = mpc_sum(&mut engine, &[32.0]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(|| {
            println!("[2] Setting up...");
            let mut engine = setup_engine("127.0.0.1:1235", &["127.0.0.1:1234"], Path::new("src/context2.bin") ).unwrap();
            println!("[2] Ready");
            let res = mpc_sum(&mut engine, &[32.0]).unwrap();
            println!("[2] Done");
            drop(engine);
            res
        });
        let a = t1.join().expect("joining")[0];
        let b = t2.join().expect("joining")[0];
        assert_eq!(a, 64.0);
        assert_eq!(b, 64.0);
    }
}