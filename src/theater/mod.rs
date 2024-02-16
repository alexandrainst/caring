//! Currently unused.

pub mod properties;

/// # Design
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
///
/// Anyway, the big purpose here is the possible to use ordinary programming constructs to write
/// the MPC program. There might be suttleties where this could fall apart, such as round
/// syncronization, but using a seperate thread/process behind the scenes could work.
///
/// Other difficulties involve the use of blocking, but as we can't really use Num traits with
/// async.
///
/// The same goes for the idea of using `const` to build the program at compile-time and optimize
/// for it. We 'almost' do that here, but could deviate from it a bit if we introduce too much
/// abstraction. Anyway, the biggest problem is probably the async stuff not running concurrently.
///
/// Lastly another approach could be to use a DSL or macros to write out the program statically.
///
/// *But* the current design is made to allow for interopability with existing generic numeric
/// code, thus leveraging a possible huge library.
///
/// ... Regarding the async/blocking we could also just do both, and have the one wrap to other?
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use ff::Field;
use rand::thread_rng;

use crate::{
    net::network::InMemoryNetwork,
    protocols::beaver::{beaver_multiply, BeaverTriple},
    schemes::Shared,
};

// Maybe we just need a single Mutex to this struct instead?
// It would be more efficient
pub struct Engine<F, S: Shared<F>> {
    context: S::Context,
    resources: Vec<BeaverTriple<F, S>>,
    network: InMemoryNetwork, // TODO: Be generic over network type
    runtime: tokio::runtime::Runtime,
}

impl<F, S: Shared<F>> Engine<F, S> {
    // TODO: Proper error handling
    // INFO: Consider if we should block on all these or actually use async?
    // It is kind of weird since we have a tokio runtime in the engine, but we could just
    // schedule things on that, and put it in an Arc or something?

    pub fn input(&mut self, value: F) -> S {
        let mut rng = thread_rng();

        let mut shares = S::share(&self.context, value, &mut rng);
        let index = self.network.index;
        let my_share = shares.remove(index);
        self.network.unicast(&shares);
        my_share
    }

    pub fn recv_input(&mut self, id: usize) -> S {
        let fut = async { self.network[id].recv().await.unwrap() };
        self.runtime.block_on(fut)
    }

    pub fn symmetric_input(&mut self, value: F) -> Vec<S> {
        let mut rng = thread_rng();
        let shares = S::share(&self.context, value, &mut rng);
        let fut = async { self.network.symmetric_unicast(shares).await.unwrap() };
        self.runtime.block_on(fut)
    }

    pub fn open(&mut self, to_open: S) -> F {
        let fut = async { self.network.symmetric_broadcast(to_open).await.unwrap() };
        let shares = self.runtime.block_on(fut);
        S::recombine(&self.context, &shares).unwrap()
    }
}

pub struct Secret<F, S: Shared<F>> {
    share: S,
    phantom: PhantomData<F>,
    // HACK: Summarize these different Arcs/refs into one single 'context',
    // since they all should be the same object.
    engine: Arc<Mutex<Engine<F, S>>>,
}

impl<F, S: Shared<F>> std::ops::Add for Secret<F, S> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share + rhs.share,
            ..self
        }
    }
}

impl<F, S: Shared<F>> std::ops::Sub for Secret<F, S>
where
    S: std::ops::Sub<Output = S>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            share: self.share - rhs.share,
            ..self
        }
    }
}

impl<Ctx, F, S: Shared<F, Context = Ctx> + Copy> std::ops::Mul for Secret<F, S>
where
    F: serde::Serialize + serde::de::DeserializeOwned + Field,
    S: std::ops::Mul<F, Output = S>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(
            Arc::ptr_eq(&self.engine, &rhs.engine),
            "Secret shares are not run on the same engine!"
        );
        let share = {
            let Engine {
                context,
                resources,
                network,
                runtime,
                ..
            } = &mut *self.engine.lock().unwrap();
            let triple = resources.pop().unwrap();

            let res = beaver_multiply::<Ctx, S, F>(context, self.share, rhs.share, triple, network);
            runtime.block_on(res).unwrap()
        };

        Self { share, ..self }
    }
}

impl<Ctx, F, S: Shared<F, Context = Ctx>> std::ops::Div for Secret<F, S> {
    type Output = Self;

    fn div(self, _rhs: Self) -> Self::Output {
        todo!("implement divison")
    }
}

// TODO: Implement Num + NumRef + NumOps traits where possible.
