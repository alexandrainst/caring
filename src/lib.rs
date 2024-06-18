#![deny(unsafe_code)]
#![allow(refining_impl_trait)]
#![allow(dead_code)]
#![feature(async_fn_traits)]
#![feature(associated_type_defaults)]
#![allow(clippy::cast_possible_truncation)]

pub mod algebra;
pub mod net;
pub mod ot;
mod protocols;
pub mod schemes;

mod help;
pub mod marker;

#[cfg(test)]
mod testing;
