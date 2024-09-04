#![deny(unsafe_code)]
#![allow(refining_impl_trait)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![feature(async_fn_traits)]
#![feature(async_closure)]
#![feature(iterator_try_collect)]

pub mod algebra;
mod help;
pub mod marker;
pub mod net;
pub mod ot;
mod protocols;
pub mod schemes;
pub mod vm;

#[cfg(any(test, feature = "test"))]
pub mod testing;
