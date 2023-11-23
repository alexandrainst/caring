/// The purposes of this module is providing a shared interface abstraction over the notion of
/// shared into the notion of 'Secret' and 'Public' variables in a computation.
///
/// Furthermore is the idea that secret numbers 'just work' as regular numbers would,
/// in the sense that they implement the basic numeric functionality related to numbers,
/// and as should would be able to swap out regular numbers in a workflow.
///
/// There will probably be some performance loss without specialization as we can't
/// specialize for different schemes.
/// Likewise there will be some trade-offs since we are trying with generics instead of dynamic
/// dispatch or enums. These approaches could also explored in how a system could dynamically
/// reason about schemes. However dynamic-dispatch can be a bit difficult since some of the underlying
/// traits are not object-safe.
///
use std::{marker::PhantomData, sync::{Arc, Mutex}};

use crate::schemes::{Shared, beaver::{BeaverTriple, beaver_multiply}};
use ff::Field;

use crate::net::network::InMemoryNetwork;


// INFO: Maybe drop this and use generics in Secret instead?
// This could make sense if we have a big context object anyway,
// since we can just implement the required traits anyway.

pub struct Engine<F, S: Shared<F>> {
    context: S::Context,
    resources: Mutex<Vec<BeaverTriple<F, S>>>,
    network: Mutex<InMemoryNetwork>,
    runtime: Mutex<tokio::runtime::Runtime>,
}

pub struct Secret<F, S: Shared<F>> {
    share: S,
    phantom: PhantomData<F>,
    // HACK: Summarize these different Arcs/refs into one single 'context',
    // since they all should be the same object.
    engine: Arc<Engine<F,S>>,
}

impl<F, S: Shared<F>> std::ops::Add for Secret<F,S> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share + rhs.share,
            ..self
        }
    }
}


impl<F, S: Shared<F>> std::ops::Sub for Secret<F,S> where S: std::ops::Sub<Output=S> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share - rhs.share,
            ..self
        }
    }
}



impl<Ctx, F, S: Shared<F, Context=Ctx> + Copy> std::ops::Mul for Secret<F,S> where F: serde::Serialize + serde::de::DeserializeOwned + Field, S: std::ops::Mul<F, Output=S> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let share = {
            let Engine {context, resources, network, runtime} = &*self.engine;
            let triple = resources.lock().unwrap().pop().unwrap();
            let network = &mut *network.lock().unwrap();
            let runtime = &*runtime.lock().unwrap();

            let res = beaver_multiply::<Ctx, S, F>(context, self.share, rhs.share, triple, network);
            runtime.block_on(res).unwrap()
        };

        Self {
            share,
            ..self
        }
    }
}
