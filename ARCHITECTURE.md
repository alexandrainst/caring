# Architecture


## Layers

The `caring` crate is loosely split into the following different "modules" or traits,
with lower levels being more concrete and higher levels more abstract.

```
              VM Engine
  Protocols      |
     \__________InteractiveShared
                  /           \
               Communicate   Shared
                |              |
               Network       Schemes (shamir, spdz, etc.) 
                |
               Connection
```

At the lowest level we have the raw `Connection<R,W>` implementation,
which can be anything that implements `tokio::AsyncRead`/`tokio::AsyncWrite`,
such as the `TcpConnection` or (in-memory) `DuplexConnection`.
This is abstracted away to a `SendBytes`/`RecvBytes` and `Channel`/`SplitChannel` traits,
which allows for any kind of abstract channel for sending and receiving bytes,
such as the multiplexing module `caring::net::mux`.

A group of connections then form a `Network` which allows for more broad communication by
broadcast, uni(que)cast or tuning to a specific channel.
This enables the traits `Broadcast`, `Unicast`, `Tuneable` forming `Communicate`.
The reasoning behind this allowing protocols to specify if they only need to broadcast/unicast,
and such can specialize in different kinds depending on the scenario.
Furthermore, it allows different schemes of verifiable broadcasts to be swapped in place.





## Networking

> Note
> A lot of functions are defined in terms of items that belong to specific party,
> thus using a `Vec` or slice. The order of these matter, since it will be implied that
> index `i` belongs to player/party `i`.


## Schemes


### Interactive vs Non-interactive Schemes

There are two main traits defining a secret sharing scheme,
the first one is the `Shared` which is for schemes such as `shamir` and `feldman` where
the protocols for sharing, reconstruction, etc., does not need interactive communication from the other parties,
but can simply be made locally.

The `InteractiveShared` trait however does define asynchronous functions for performing sharing and reconstruction
with the aid of the `Communication` trait.
All non-interactive `Shared` schemes can all function as interactive due to a blanket implementation.


## Shares

The share is simply a struct containing (usually) the finite field of the secret shared data.
In the case of Shamir Secret Sharing (and derivatives) the (public) `x` field is left out,
as it is implied by the holder, except for the cases of sharing and reconstruction,
where it is implied by the index.
This is both to minimize the memory footprint of shares and to more strictly enforce checks on the `x`,
as it could be *cheated* with, and we would then require context based parsing for all shares,
which would be difficult.


## Protocols



## Testing Framework

