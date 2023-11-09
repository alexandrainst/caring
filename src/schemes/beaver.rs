use std::marker::PhantomData;

use ff::Field;
use rand::RngCore;

use crate::{
    agency::Broadcast, schemes::Shared,
};

#[derive(Clone)]
pub struct BeaverTriple<F, S: Shared<F>> {
    phantom: PhantomData<F>,
    shares: (S, S, S),
}

impl<F: Field, C, S: Shared<F, Context = C>> BeaverTriple<F, S> {
    /// Fake a set of beaver triples.
    ///
    /// This produces `n` shares corresponding to a shared beaver triple,
    /// however locally with the values known.
    ///
    /// * `ids`: ids to produce for
    /// * `threshold`: threshold to reconstruct
    /// * `rng`: rng to sample from
    pub fn fake(ctx: &mut C, mut rng: &mut impl RngCore) -> Vec<Self> {
        let a = F::random(&mut rng);
        let b = F::random(&mut rng);
        let c: F = a * b;
        // Share (preproccess)
        let a = S::share(ctx, a);
        let b = S::share(ctx, b);
        let c = S::share(ctx, c);
        itertools::izip!(a, b, c)
            .map(|(a, b, c)| Self {
                shares: (a, b, c),
                phantom: PhantomData,
            })
            .collect()
    }
}

/// Perform multiplication using beaver triples
///
/// * `x`: first share to multiply
/// * `y`: second share to multiply
/// * `triple`: beaver triple
/// * `network`: unicasting network
pub async fn beaver_multiply<
    C,
    S: Shared<F, Context=C> + Copy + std::ops::Mul<F, Output=S>,
    F: Field + serde::Serialize + serde::de::DeserializeOwned,
    E,
>(
    ctx: &mut C,
    x: S,
    y: S,
    triple: BeaverTriple<F, S>,
    network: &mut impl Broadcast,
) -> Option<S> {
    let BeaverTriple {
        shares: (a, b, c),
        phantom: _,
    } = triple;
    let ax: S = a + x;
    let by: S = b + y;

    // Sending both at once it more efficient.
    let resp = network.symmetric_broadcast::<_>((ax, by)).await.unwrap();
    let (ax, by): (Vec<_>, Vec<_>) = itertools::multiunzip(resp);

    let ax = S::recombine(ctx, &ax)?;
    let by = S::recombine(ctx, &by)?;

    Some(y * ax + a * (-by) + c)
}
