use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{algebra::field::Field, net::Id, schemes::Shared};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Share<F> {
    pub value: F,
    pub issued_to: Id,
    pub issued_by: Option<Id>,
}

impl<F: Field> std::ops::Add for Share<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.issued_to, rhs.issued_to, "Not issued to same party!");
        Self {
            value: self.value + rhs.value,
            issued_to: self.issued_to,
            issued_by: None,
        }
    }
}

impl<F: Field> std::ops::Sub for Share<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.issued_to, rhs.issued_to, "Not issued to same party!");
        Self {
            value: self.value - rhs.value,
            issued_to: self.issued_to,
            issued_by: None,
        }
    }
}

impl<F: Field> std::ops::Mul<F> for Share<F> {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        Self {
            value: self.value * rhs,
            ..self
        }
    }
}

impl<F> Share<F> {
    pub fn new_context(id: Id, total_parties: usize) -> Context {
        Context {
            all_parties: total_parties,
            me: id,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Context {
    pub all_parties: usize,
    pub me: Id,
}

impl Context {
    pub fn parties(&self) -> std::ops::Range<usize> {
        0..self.all_parties
    }
}

pub fn assert_holding_same<F>(shares: &[Share<F>]) {
    assert!(
        shares.iter().map(|s| s.issued_to).all_equal(),
        "Not issued to same party!"
    );
}

pub fn assert_holded_by<F>(shares: impl IntoIterator<Item = Share<F>>, party: Id) {
    let val = shares
        .into_iter()
        .map(|s| s.issued_to)
        .all_equal_value()
        .expect("Not all values were the same");
    assert_eq!(val, party, "Shares not issued to party {party:?}");
}

impl<F: Field + Serialize + DeserializeOwned + PartialEq + Send + Sync> Shared for Share<F> {
    type Context = Context;

    type Value = F;

    fn share(
        ctx: &Self::Context,
        secret: Self::Value,
        _rng: impl rand::prelude::RngCore,
    ) -> Vec<Self> {
        (0..ctx.all_parties)
            .map(|i| Share {
                value: secret,
                issued_to: Id(i),
                issued_by: Some(ctx.me),
            })
            .collect()
    }

    fn recombine(_ctx: &Self::Context, shares: &[Self]) -> Option<Self::Value> {
        let mut all_is_well = true;
        for (i, share) in shares.iter().enumerate() {
            all_is_well &= share.issued_to.0 == i
        }

        if !all_is_well {
            let ids: Vec<_> = shares.iter().map(|s| s.issued_to).collect();
            panic!("Mismatch in issued shares and order. Received them as:\n {ids:#?}");
        }

        assert!(
            shares.iter().map(|s| s.value).all_equal(),
            "Not all shares agreed!"
        );

        Some(shares[0].value)
    }
}
