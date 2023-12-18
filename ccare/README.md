<h1 align="center">C Bindings</h1>
<h2 align="center">*Lingua Franca MPC Routines*</h2>


## Build
```sh
cargo build (--release)
```


## Generating headers
Headers can be generated using [`cbindgen`](https://github.com/mozilla/cbindgen).
```sh
cbindgen  --lang=C --output=target/release/libcaring.h
```

## Sample Matlab Code
A sample matlab code can be found in [test.m](./test.m) which showcases how to consume the C library.
Note that the script expects the library and headers to be placed in `target/release`.
