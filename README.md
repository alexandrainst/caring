<h1 align="center">Caring</h1>
<div align="center">is Secret Sharing</div>
<br/><br/>

This is a library for performing secure multiparty computation by (mostly) secret sharing.
The aim of this is to provide datatypes and constructs to build larger MPC programs,
and be generic in the underlying protocols and settings a givens scheme can use.

Currently, we are working with the following schemes:
- Shamir Secret Sharing (complete)
- Feldman's Secret Sharing (lack. multiplication)
- SPDZ (only online phase)
- Pedersen Secret Sharing (lack. addition + multiplication)
- Rep3 (addition)

*Note*: This is prototype software and not suited to be used in security critical applications as of yet.
## Subprojects

[`wecare`](./wecare) provides an abstraction over `caring` by removing the generics and baking the finite fields in directly,
which can then be selected at runtime.

[`pycare`](./pycare) takes `wecare` and provides python bindings to it allowing easily secret sharing in Python.
See [pycare/examples](./pycare/examples) for example applications.


# Inspiration
* [MP-SPDZ](https://github.com/data61/MP-SPDZ)
* [Fresco](https://github.com/aicis/fresco)
