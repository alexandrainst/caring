This module should probably be outsourced to another crate.
Most of the functionality are dependent on either abstract algebra (modular arithemetic and prime fields),
polynomials and vectors. Most of this is not that complicated, however the `Vector` struct is bit weird,
as its name overlap with `Vec`. The purpose is more as a wrapper for adding vectorized addition and constant multiplication.
It is mostly similar to the `numpy` 'array'.

Maybe use the [ndarray](https://crates.io/crates/ndarray) crate?
