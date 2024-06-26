use ff::Field;
use rand::Rng;

use crate::{net::agency::Unicast, schemes::Shared};

pub async fn pick_random<C, F, S>(
    ctx: &C,
    rng: &mut impl Rng,
    cx: &mut impl Unicast,
    upper: u64, /* should be `F` */
) -> S
where
    F: From<u64>, /* <-- not nice */
    F: Field,
    S: Shared<Value = F, Context = C> + std::iter::Sum,
{
    // Currently there are no enforcement that the random value be in the given range
    // or even being random at all.
    //
    // Assuming the range requirement is upheld, a malicious party can simply choose the
    // largest or smallest value, and they will know that it lies thus lies closer to either
    // the top or bottom.
    let parties = cx.size() as u64;
    let num = rng.gen_range(0..(upper / parties)); // NOTE: This might not be a good idea.
    assert!(num * parties <= upper, "{num} * {parties} <= {upper}");
    dbg!(num);
    let num = F::from(num);

    let issued_shares = S::share(ctx, num, rng);
    let holded_shares = cx
        .symmetric_unicast(issued_shares)
        .await
        .expect("Proper error handling");
    let share_sum: S = holded_shares.into_iter().sum();

    share_sum
}

#[cfg(test)]
mod test {
    use rand::rngs::mock;

    use crate::{
        algebra::element::Element32,
        protocols::rand::pick_random,
        schemes::{shamir::ShamirParams, Shared},
        testing::*,
    };

    #[tokio::test]
    async fn check_rand() {
        type F = Element32;
        type Share = crate::schemes::shamir::Share<F>;

        let initial = 13093535812822639135;
        let increment = 11456580978330466844;
        Cluster::new(4)
            .run(|mut net| async move {
                let upper = 128;
                let ids = (1..=4u64).map(F::from).collect();
                let ctx = ShamirParams { threshold: 3, ids };
                let mut rng = mock::StepRng::new(initial, increment);

                for i in 0..10 {
                    let num: Share = pick_random(&ctx, &mut rng, &mut net, upper).await;
                    let shares = net.symmetric_broadcast(num).await.unwrap();
                    let num = Share::recombine(&ctx, &shares).unwrap();
                    let num: u64 = num.into();

                    println!("{i}: {num}");
                    assert!(num < upper);
                }
            })
            .await
            .unwrap();
    }
}
