#![deny(unsafe_code)]
#![allow(refining_impl_trait)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![feature(async_fn_traits)]

pub mod algebra;
pub mod net;
pub mod ot;
mod protocols;
pub mod schemes;

mod help;
pub mod marker;

#[cfg(test)]
mod testing;
pub mod vm;
