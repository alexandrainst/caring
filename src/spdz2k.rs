use rand::RngCore;
use derive_more::{Add, Sub};

#[derive(Add, Sub, Clone, Copy)]
pub struct Share {
    val: u128,
    mac: u128,
}

struct GlobalKey<T>(T);


impl std::ops::Mul<u128> for Share {
    type Output = Share;

    fn mul(self, rhs: u128) -> Self::Output {
        Share{val:self.val * rhs, mac: self.mac * rhs}
    }
}


pub fn share(val: u128, n: usize, key: u128, rng: &mut impl RngCore) -> Vec<Share> {
    // HACK: This is really not secure at all.
    let mut shares : Vec<_> = (0..n).map(|_| {
        let mut buf = [0u8; 16];
        rng.fill_bytes(&mut buf);
        u128::from_le_bytes(buf)
    }).collect();

    let sum : u128 = shares.iter().sum();
    shares[0] -=  sum - val;
    // In Fresco, this is all very interactive

    shares.into_iter().map(|x| {
        Share{val: x, mac: key * x}
    }).collect()
}


pub fn reconstruct(shares: &[Share]) -> u128 {
    shares.iter().map(|x| x.val).sum()
}
