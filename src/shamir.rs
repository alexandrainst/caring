use ff::{derive::rand_core::RngCore, Field};

#[derive(Clone, Copy, Debug)]
pub struct Share<F: Field> {
    // NOTE: Consider removing 'x' as it should be implied by the user handling it
    pub(crate) x: F,
    pub(crate) y: F,
}

impl<F: Field> std::ops::Add for Share<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.x == rhs.x);
        Self {
            x: self.x,
            y: self.y + rhs.y,
        }
    }
}

impl<F: Field> std::ops::Add<F> for Share<F> {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        Self {
            x: self.x,
            y: self.y + rhs,
        }
    }
}

impl<F: Field> std::ops::Mul<F> for Share<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self { y: self.y * rhs, ..self }
    }
}


impl<F: Field> std::ops::Mul<Share<F>> for Share<F> {
    type Output = MultipliedShare<F>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.x == rhs.x);
        MultipliedShare {x: self, y: rhs}
    }
}

#[derive(Clone, Copy)]
pub struct MultipliedShare<F: Field>{
    x: Share<F>, y: Share<F>
}

#[derive(Clone, Copy)]
pub struct ExplodedShare<F: Field>{
    ax: F, by: F
}

impl<F: Field> MultipliedShare<F> {
    // mask with beaver triple
    pub fn explode(self, a: F, b: F) -> ExplodedShare<F> {
        ExplodedShare{ax: self.x.y + a, by: self.y.y + b}
    }

    // unmask with last
    pub fn implode(self, shards: &[ExplodedShare<F>], c: F) -> Share<F> {
        // open the shares (now shards)
        let (a,b) = shards.iter()
            .map(|s| (s.ax, s.by))
            .fold((F::ZERO, F::ZERO), |(x,y), (a,b)| ((x+a), (y+b)));

        // Yea, the naming is pretty bad
        // Should be fixed if we axe 'x' from the share.
        let x = self.x.y;
        let y = self.y.y;
        let z = a*y - b*x + c;
        Share { x: self.x.x,  y: z }
    }
}




/// Share/shard a secret value `v` into `n` shares
/// where `n` is the number of the `ids`
///
/// * `v`: secret value to share
/// * `ids`: ids to share to
/// * `threshold`: threshold to reconstruct it
/// * `rng`: rng to generate shares from
pub fn share<F: Field>(v: F, ids: &[F], threshold: u64, rng: &mut impl RngCore) -> Vec<Share<F>> {
    let n = ids.len();
    assert!(
        n >= threshold as usize,
        "Threshold should be less-than-equal to the number of shares"
    );

    // Sample random t-degree polynomial
    let mut polynomial: Vec<F> = Vec::with_capacity(threshold as usize);
    polynomial.push(v);
    for _ in 1..threshold {
        let a = F::random(&mut *rng);
        polynomial.push(a);
    }

    // Sample n points from 1..=n in the polynomial
    let mut shares: Vec<Share<F>> = Vec::with_capacity(n);
    for x in ids {
        let x = *x;
        let share = polynomial
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let exp: [u64; 1] = [i as u64];
                (*a) * x.pow(exp)
            })
            .fold(F::ZERO, |sum, x| sum + x);
        shares.push(Share::<F> { x, y: share });
    }

    shares
}

/// Reconstruct or open shares
///
/// * `shares`: shares to be combined into an open value
pub fn reconstruct<F: Field>(shares: &[Share<F>]) -> F {
    // Lagrange interpolation
    let mut sum = F::ZERO;
    for share in shares.iter() {
        let xi = share.x;
        let yi = share.y;

        let mut prod = F::ONE;
        for Share { x: xk, y: _ } in shares.iter() {
            let xk = *xk;
            if xk != xi {
                prod *= -xk * (xi - xk).invert().unwrap();
            }
        }
        sum += yi * prod;
    }
    sum
}

#[cfg(test)]
mod test {
    use crate::field::Element32;

    use super::*;

    #[test]
    fn simple() {
        let mut rng = rand::thread_rng();
        let v = Element32::from(42u32);
        let ids: Vec<_> = (1..=5u32).map(Element32::from).collect();
        let shares = share(v, &ids, 4, &mut rng);
        let v = reconstruct(&shares);
        assert_eq!(v, Element32::from(42u32));
    }

    #[test]
    fn addition() {
        const PARTIES : std::ops::Range<u32> = 1..5u32;
        let a = 3;
        let b = 7;
        let vs1 = {
            let v = Element32::from(a);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };

        // MPC
        let shares : Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a,&b)| a+b).collect();
        let v = reconstruct(&shares);
        let v : u32 = v.into();
        assert_eq!(v, a + b);
    }

    use fixed::FixedU32;

    #[test]
    fn addition_fixpoint() {
        const PARTIES : std::ops::Range<u32> = 1..5u32;
        type Fix = FixedU32::<16>;
        let a = 1.0;
        let b = 3.0;
        let a = Fix::from_num(a);
        let b = Fix::from_num(b);

        let vs1 = {
            let v = Element32::from(a.to_bits() as u64);
            dbg!(&v);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b.to_bits() as u64);
            dbg!(&v);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };

        // MPC
        let shares : Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a,&b)| a+b).collect();
        let v = reconstruct(&shares);
        dbg!(v);

        // back to fixed
        let v : u32 = v.into();
        let v = Fix::from_bits(v as u32);
        assert_eq!(v, a+b);
    }

    #[test]
    fn multiplication() {
        const PARTIES : std::ops::Range<u32> = 1..5u32;

        let a = 3;
        let b = 7;
        let vs1 = {
            let v = Element32::from(a);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let vs2 = {
            let v = Element32::from(b);
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            share(v, &ids, 4, &mut rng)
        };
        let (bt_a, bt_b, bt_c) = {
            let (a,b,c) = (Element32::from(7u32), Element32::from(13u32), Element32::from(7*13u32));
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = PARTIES.map(Element32::from).collect();
            let a = share(a, &ids, 4, &mut rng);
            let b = share(b, &ids, 4, &mut rng);
            let c = share(c, &ids, 4, &mut rng);
            (a,b,c)
        };

        // MPC
        let mult_shares : Vec<_> = vs1.iter().zip(vs2.iter()).map(|(&a,&b)| a*b).collect();
        // explode out to other share with other parties
        let exp_shares : Vec<_> = mult_shares.iter()
            .zip(bt_a)
            .zip(bt_b)
            .map(|((s,a),b)| s.explode(a.y,b.y)).collect();
        // implode back into the multiplactions
        let shares : Vec<_> = mult_shares.iter()
            .zip(bt_c)
            .map(|(s,c)| s.implode(&exp_shares, c.y)).collect();

        let v = reconstruct(&shares);
        let v : u32 = v.into();
        assert_eq!(v, a * b);
    }
}
