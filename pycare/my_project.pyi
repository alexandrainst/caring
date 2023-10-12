"""
    Setup a MPC Engine for adding numbers together.
    The engine will connect to the given addresses and listen
    on the first socket address.
    :param my_addr: the address to listen on
    :param horsepower: the addresses to connect to
"""
def setup(my_addr: str, *others: str) -> None: ...


"""
    Performs a summation with the connected parties.
    Returns the sum of all the numbers.
    :param a: number to summate with
"""
def sum(a: float) -> float: ...


"""
    Takedown the MPC Engine, releasing the resources and dropping connections.
"""
def takedown() -> None: ...
