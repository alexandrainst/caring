use std::env::Args;

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::net::Communicate;

#[derive(Clone, Serialize, Deserialize)]
pub struct Proposal<T> {
    initative: T,
    creator: usize,
}


pub trait Initiative: Serialize + DeserializeOwned + Sync {
    type Args;
    fn name() -> &'static str;
    fn execute(args: Args);
}

impl<T: Initiative> Proposal<T> {

    pub async fn initiate<C: Communicate>(initative: T, mut coms: C) -> Result<Option<Self>, C::BroadcastError> {
        let creator = coms.id();
        let proposal = Self {initative, creator};
        let request = ProposalRequest(proposal);

        coms.broadcast(&request).await?;
        request.accept(coms).await
    }

    pub fn execute(self, args: Args) {
        T::execute(args)
    }

}

#[repr(transparent)]
#[derive(Clone, Serialize, Deserialize)]
pub struct ProposalRequest<T>(Proposal<T>);

impl<T: Initiative> ProposalRequest<T> {

    async fn vote<C: Communicate>(self, vote: bool, mut coms: C) -> Result<Option<Proposal<T>>, C::BroadcastError> {
        let votes = coms.symmetric_broadcast(vote).await?;
        let n =  votes.len();
        let yes_votes = votes.into_iter().filter(|&v| v).count();
        let res = if yes_votes > n {
            Some(self.0)
        } else {
            None
        };
        Ok(res)
    }

    pub async fn accept<C: Communicate>(self, coms: C) -> Result<Option<Proposal<T>>, C::BroadcastError> {
        self.vote(true, coms).await
    }

    pub async fn reject<C: Communicate>(self, coms: C) -> Result<Option<Proposal<T>>, C::BroadcastError>{
        self.vote(false, coms).await

    }
}
