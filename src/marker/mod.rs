mod exptree;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{net::Communicate, schemes::{Shared, Verify}};


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

    pub fn assume_verified(self) -> Verified<S> {
        Verified(self.0)
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
    pub async fn verify_all(self, coms: impl Communicate, args: S::Args) -> Verified<Vec<Option<S>>> {
        let res = S::verify_many(&self.0, coms, args).await;
        let res = res.into_iter().zip(self.0).map(|(verified, t)| {
            verified.then_some(t)
        }).collect();
        Verified(res)
    }


    pub fn assume_verified(self) -> Verified<S> {
        todo!()
    }
}

// Pure boring manual operator implementations
// Could be done with some macros instead.
mod ops {
    use std::ops::{Add, Sub};

    use crate::schemes::Shared;

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
        let s  = mock::Share::share(&ctx, Mod11(3), &mut rng);
        let s = Verified(s[0]);
        let s0 = s.clone();
        let s = s0 + s;

        let to_send = bincode::serialize(&s).unwrap();
        // sending...
        let back_again: Unverified<Share<Mod11>> = bincode::deserialize(&to_send).unwrap();

        println!("Hello again {back_again:?}");
    }
}
