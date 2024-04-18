<h1 align="center">Caring for Python</h1>
<div align="center">Because we care about Python too!</div>

Python bindings to perform a MPC summation.
```py
    import pycare
    # Setup the system over TCP
    engine = pycare.setup("<your address>", "<first peer address>", "<more peer addresses>")
    # Some floating point number and receive output
    res = engine.sum(2.45)
    # Takedown the system again.
    engine.takedown()
```
Note that there is currently no support for channel encryption.


## Install
We currently build using GitHub actions for Python 3.7 and above for Linux x86_64 and aarch64.
The targets can be found [actions](https://github.com/alexandrainst/caring/actions/workflows/pyo3.yml) for the PyO3 CI as wheels.
To install a wheel run the following `pip install caring-<version>.whl` for the version for your given platform.


## Examples
There are two test files in [examples](./examples) showcasing the use between two different parties test1.py and test2.py.


## Develop
To install the package locally you want to setup a virtual environment and install `maturin`.
```bash
    $ python -m venv .env
    $ source .env/bin/activate
    $ pip install maturin
```
Then you should be able to run `maturin develop` and have a version of `pycare` available.

*Built with ❤️ using [PyO3](https://github.com/PyO3/pyo3)*
