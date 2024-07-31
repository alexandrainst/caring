use crate::net::Communicate;
use std::{
    collections::BTreeMap,
    ops::AsyncFn,
    sync::{
        atomic::{AtomicU32, Ordering::SeqCst},
        Mutex,
    },
};
#[derive(Clone)]
pub enum Status<T> {
    Verified,
    Unverified {
        parents: (u32, u32),
        data: Option<T>,
    },
    Failure,
}

pub struct ExpTree<T> {
    tree: Mutex<BTreeMap<u32, Status<T>>>,
    issuer: AtomicU32,
}

impl<T: Clone> ExpTree<T> {
    fn issue(&self) -> u32 {
        let curr = self.issuer.fetch_update(SeqCst, SeqCst, |n| Some(n + 1));
        curr.expect("Never fails since lambda always returns Some")
    }

    pub fn add_dependent(&self, a: u32, b: u32) -> u32 {
        let new = self.issue();
        let status: Status<T> = Status::Unverified {
            parents: (a, b),
            data: None,
        };
        {
            let mut tree = self.tree.lock().unwrap();
            tree.insert(new, status);
        }
        new
    }

    pub fn add_dependent_with(&self, a: u32, b: u32, data: T) -> u32 {
        let new = self.issue();
        let data = Some(data);
        let status = Status::Unverified {
            parents: (a, b),
            data,
        };
        {
            let mut tree = self.tree.lock().unwrap();
            tree.insert(new, status);
        }
        new
    }

    pub fn add_root(&self) -> u32 {
        let new = self.issue();
        let status = Status::Verified;
        {
            let mut tree = self.tree.lock().unwrap();
            tree.insert(new, status);
        }
        new
    }

    pub async fn verify<F, C>(&mut self, id: u32, verify_fn: F, coms: C) -> Option<bool>
    where
        F: AsyncFn(Vec<T>, C) -> Option<bool>,
        C: Communicate,
    {
        let mut to_check = vec![];
        let mut datas: Vec<T> = vec![];
        let mut checking = vec![];
        checking.push(id);

        let tree = self.tree.get_mut().unwrap();

        while let Some(id) = checking.pop() {
            let t = tree.get(&id)?;
            match t {
                Status::Failure => todo!("Taint dependent values -or- crash and burn"),
                Status::Verified => (),
                Status::Unverified { parents, data } => {
                    checking.push(parents.0);
                    checking.push(parents.1);
                    if let Some(data) = data {
                        datas.push(data.clone());
                    }
                    to_check.push(id);
                }
            }
        }

        let res = verify_fn.async_call((datas, coms)).await?;
        let status = if res {
            Status::Verified
        } else {
            Status::Failure
        };

        for id in to_check {
            tree.insert(id, status.clone());
        }

        Some(res)
    }
}
