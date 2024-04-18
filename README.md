<h1 align="center">Caring</h1>
<div align="center">is Secret Sharing</div>
<br/><br/>

This is a library for performing secure multiparty computation by (mostly) secret sharing.
The aim of this is to provide datatypes and constructs to build larger MPC programs,
and be generic in the underlying protocols and settings a givens scheme can use.

Currently, we are working with the following schemes:
- Shamir Secret Sharing (complete)
- Feldman's Secret Sharing (lack. multiplication)
- Pedersen Secret Sharing (lack. addition + multiplication)
- Rep3 (addition)
- SPDZ (wip)

*Note*: This is prototype software and not suited to be used in security critical applications as of yet.
## Subprojects

The base crate here is to provide a library with secret-sharing and other MPC functionality.
The project contains a sample of a consuming library [`wecare`](./wecare) provinding a subroutine for the securely computing a simple sum.
This is further used by [`ccare`](./ccare) and [`pycare`]('./pycare') for C and Python bindings respectively.


# Inspiration
* [MP-SPDZ](https://github.com/data61/MP-SPDZ)
* [Fresco](https://github.com/aicis/fresco)
