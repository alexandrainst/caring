use std::{net::SocketAddr, time::Duration};
use caring::{net::network::TcpNetwork, schemes::spdz::{self, preprocessing}};
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
        threshold: _,
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
        let vals: Vec<_> = nums.clone();
        let who_am_i = context.params.who_am_i();
        let mut shares: Vec<Vec<spdz::Share<_>>> = vec![];
        for i in 0..number_of_parties{
            if i == who_am_i {
                let s: Vec<spdz::Share<_>> = spdz::share(Some(vals.clone()), &mut context.preprocessed_values.for_sharing, &context.params, i, network).await.expect("TODO: look at error");
                shares.push(s); 
            } else {
                let s: Vec<spdz::Share<_>> = spdz::share(None, &mut context.preprocessed_values.for_sharing, &context.params, i, network).await.expect("TODO: look at error"); 
                shares.push(s); 
            }

        }
        // Compute
        let mut shares = shares.into_iter();
        let mut my_result: Vec<spdz::Share<_>> = shares.next().expect("atleast one result");
        let my_res_len = my_result.len();
        for s in shares {
            println!("my_result: {}", my_result.len());
            for i in 0..my_res_len {
                my_result[i] += s[i];
            }
        }

        // TODO: make random element
        let random_element = curve25519_dalek::Scalar::from(12u128);
        let res: Vec<_> = 
            if my_result.len() == 1 {
                vec![
                    from_offset(
                        u128::from_le_bytes(
                            spdz::open_res(my_result.pop().expect("we just checked, it is there."), network, &context.params, &context.opened_values).await
                                .as_bytes()[0..128/8]
                                .try_into().expect("convertion between types should go well"
                            )
                        )
                    )
                ]
            } else {
                spdz::open_res_many(my_result, network, &context.params, &context.opened_values, random_element).await
                    .into_iter()
                    .map(|x| x.as_bytes()[0..128/8].try_into().expect("convertion between types should go well"))
                    .map(u128::from_le_bytes)
                    .map(from_offset)
                    .collect()
            };
        Some(res)
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

pub fn do_preproc(filenames: &[&Path], number_of_shares: Vec<usize>){
    assert_eq!(filenames.len(), number_of_shares.len());
    let known_to_each = vec![number_of_shares[0], number_of_shares[1]];
    let number_of_triplets = 0;
    let num = to_offset(0.0);
    preprocessing::write_preproc_to_file(
        filenames,
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
        do_preproc(&[Path::new("/tmp/context1.bin"), Path::new("/tmp/context2.bin")], vec![1,1]);
        let t1 = thread::spawn(|| {
            println!("[1] Setting up...");
            let mut engine = setup_engine("127.0.0.1:1234", &["127.0.0.1:1235"], Path::new("/tmp/context1.bin") ).unwrap();
            println!("[1] Ready");
            let res = mpc_sum(&mut engine, &[32.0]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(|| {
            println!("[2] Setting up...");
            let mut engine = setup_engine("127.0.0.1:1235", &["127.0.0.1:1234"], Path::new("/tmp/context2.bin") ).unwrap();
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

    #[test]
    fn sunshine_for_two() {
        use std::thread;
        do_preproc(&[Path::new("/tmp/context3.bin"), Path::new("/tmp/context4.bin")], vec![2,2]);
        let t1 = thread::spawn(|| {
            println!("[1] Setting up...");
            let mut engine = setup_engine("127.0.0.1:2234", &["127.0.0.1:2235"], Path::new("/tmp/context3.bin") ).unwrap();
            println!("[1] Ready");
            let res = mpc_sum(&mut engine, &[32.0, 11.9]).unwrap();
            println!("[1] Done");
            drop(engine);
            res
        });
        std::thread::sleep(Duration::from_millis(50));
        let t2 = thread::spawn(|| {
            println!("[2] Setting up...");
            let mut engine = setup_engine("127.0.0.1:2235", &["127.0.0.1:2234"], Path::new("/tmp/context4.bin") ).unwrap();
            println!("[2] Ready");
            let res = mpc_sum(&mut engine, &[32.0, 24.1]).unwrap();
            println!("[2] Done");
            drop(engine);
            res
        });
        let a = t1.join().expect("joining");
        let a1 = a[0];
        let a2 = a[1];
        let b = t2.join().expect("joining");
        let b1 = b[0];
        let b2 = b[1];
        assert_eq!(a1, 64.0);
        assert_eq!(b1, 64.0);
        assert_eq!(a2, 36.0);
        assert_eq!(b2, 36.0);
        assert_eq!(a,[64.0, 36.0]);
    }
}
