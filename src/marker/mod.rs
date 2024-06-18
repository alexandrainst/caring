mod exptree;

use rand::RngCore;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    net::Communicate,
    schemes::{interactive::InteractiveShared, Shared, Verify},
};

#[repr(transparent)]
#[derive(Serialize, Debug, Clone)]
pub struct Verified<S>(S);

#[repr(transparent)]
#[derive(Serialize, Debug, Clone)]
pub struct Unverified<S>(S);

impl<'de, S: Shared + DeserializeOwned> Deserialize<'de> for Unverified<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Deserialize::deserialize(deserializer).map(|s| Unverified(s))
    }
}

impl<S: Verify> Unverified<S> {
    pub async fn verify(self, coms: impl Communicate, args: S::Args) -> Option<Verified<S>> {
        if self.0.verify(coms, args).await {
            Some(Verified(self.0))
        } else {
            None
        }
    }
}

impl<S> Unverified<S> {
    pub fn assume_verified(self) -> Verified<S> {
        Verified(self.0)
    }
}

impl<'ctx, S: InteractiveShared<'ctx>> Verified<S> {
    pub async fn open(
        self,
        ctx: &mut S::Context,
        coms: impl Communicate,
    ) -> Result<S::Value, S::Error> {
        S::recombine(ctx, self.0, coms).await
    }

    pub async fn share(
        val: S::Value,
        ctx: &mut S::Context,
        rng: impl RngCore + Send,
        coms: impl Communicate,
    ) -> Result<Self, S::Error> {
        let s = S::share(ctx, val, rng, coms).await?;
        Ok(Self(s))
    }
}

impl<'ctx, S: InteractiveShared<'ctx>> Unverified<S> {
    pub async fn share_symmetric(
        val: S::Value,
        ctx: &mut S::Context,
        rng: impl RngCore + Send,
        coms: impl Communicate,
    ) -> Result<Vec<Self>, S::Error> {
        let s = S::symmetric_share(ctx, val, rng, coms).await?;
        Ok(s.into_iter().map(Self).collect())
    }

    pub async fn receive_share(
        ctx: &mut S::Context,
        coms: impl Communicate,
        from: usize,
    ) -> Result<Self, S::Error> {
        let s = S::receive_share(ctx, coms, from).await?;
        Ok(Self(s))
    }
}

impl<T> From<Unverified<Vec<T>>> for Vec<Unverified<T>> {
    fn from(value: Unverified<Vec<T>>) -> Self {
        value.0.into_iter().map(|t| Unverified(t)).collect()
    }
}

impl<T> From<Verified<Vec<T>>> for Vec<Verified<T>> {
    fn from(value: Verified<Vec<T>>) -> Self {
        value.0.into_iter().map(|t| Verified(t)).collect()
    }
}

impl<S: Verify> Unverified<Vec<S>> {
    pub async fn verify_all(
        self,
        coms: impl Communicate,
        args: S::Args,
    ) -> Verified<Vec<Option<S>>> {
        let res = S::verify_many(&self.0, coms, args).await;
        let res = res
            .into_iter()
            .zip(self.0)
            .map(|(verified, t)| verified.then_some(t))
            .collect();
        Verified(res)
    }
}

// Pure boring manual operator implementations
// Could be done with some macros instead.
mod ops {
    use crate::schemes::Shared;
    use std::ops::{Add, Mul, Sub};

    use super::*;

    impl<S: Shared> Add for Verified<S> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl<S: Shared> Sub for Verified<S> {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl<S: Shared> Add<Unverified<S>> for Verified<S> {
        type Output = Unverified<S>;

        fn add(self, rhs: Unverified<S>) -> Self::Output {
            Unverified(self.0 + rhs.0)
        }
    }

    impl<S: Shared> Sub<Unverified<S>> for Verified<S> {
        type Output = Unverified<S>;

        fn sub(self, rhs: Unverified<S>) -> Self::Output {
            Unverified(self.0 - rhs.0)
        }
    }

    impl<S: Shared> Add for Unverified<S> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl<S: Shared> Sub for Unverified<S> {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl<S: Shared> Mul<S::Value> for Verified<S>
    where
        S: Mul<S::Value, Output = S>,
    {
        type Output = Self;

        fn mul(self, rhs: S::Value) -> Self::Output {
            Self(self.0 * rhs)
        }
    }

    impl<S: Shared> Mul<S::Value> for Unverified<S>
    where
        S: Mul<S::Value, Output = S>,
    {
        type Output = Self;

        fn mul(self, rhs: S::Value) -> Self::Output {
            Self(self.0 * rhs)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use rand::rngs;

    use crate::{
        algebra::element::Mod11,
        testing::mock::{self, Share},
    };

    #[test]
    fn serdede() {
        let ctx = mock::Context {
            all_parties: 1,
            me: 0,
        };
        let mut rng = rngs::mock::StepRng::new(0, 0);
        let s = <mock::Share<Mod11> as Shared>::share(&ctx, Mod11(3), &mut rng);
        let s = Verified(s[0]);
        let s0 = s.clone();
        let s = s0 + s;

        let to_send = bincode::serialize(&s).unwrap();
        // sending...
        let back_again: Unverified<Share<Mod11>> = bincode::deserialize(&to_send).unwrap();

        println!("Hello again {back_again:?}");
    }
}
