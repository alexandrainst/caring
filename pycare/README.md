<h1 align="center">Pycare</h1>
<h2 align="center">Because Python Cares Too!</h2>

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

## Tests
There are two test files showcasing the use between two different parties test1.py and test2.py.

## Develop
To install the package locally you want to setup a virtual environment and install `maturin`.
```bash
$ python -m venv .env
$ source .env/bin/activate
$ pip install maturin
```
Then you should be able to run `maturin develop` and have a version of `pycare` available.



*Built with ❤️ using [PyO3](https://github.com/PyO3/pyo3)*
