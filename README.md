# Caring
_(Secret) Sharing is caring. (Name to change)_ 

This is a library for performing secret sharing.

## To Build
We use a few things that require nightly, so switch to that.
```sh
rustup default nightly
cargo build
```

## Run Demo
Pass your socket address as the first argument, then the rest as of adresses as the latter.
```sh
cargo run -- my_socket [other_sockets]
```
To run three parties you can do the following:
```sh
# Party 1
cargo run -- 127.0.0.1:1234 127.0.0.1:1235 127.0.0.1:1236
# Party 2
cargo run -- 127.0.0.1:1235 127.0.0.1:1234 127.0.0.1:1236
# Party 3
cargo run -- 127.0.0.1:1236 127.0.0.1:1235 127.0.0.1:1234
```

## Testing
Our testing is done by cargo test
```sh
cargo test
```

## Documentation
The ease the continued development it is best practice to refer to a high-level description of the given scheme/protocol in an implementation.


## Useful Crates
We depend on a couple of crates that map nicely to mathematical and cryptograhic terms.
These allow us to be more generic, and thus provide a heap of different concrete schemes.
- First of all is the `ff` crate, which provides a trait for (finite) fields and `group` for (elliptic curve) groups.
- We have the `rand` crate which allows us to model randomnness.
- We have `num_traits` that provide different traits for numbers.
- We have `digest` for abstracting hash functions.
