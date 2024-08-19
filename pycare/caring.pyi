class Engine:
    """
        Performs a summation with the connected parties.
        Returns the sum of all the numbers.
        :param a: number to summate with
    """
    def sum(self, a: float) -> float: ...

    """
        Performs a summation of a vector with the connected parties.
        Returns the sum of all the vector (element-wise) of all the numbers.
        :param a: vector to summate with.

        Note: that all parties must supply the same length!
    """
    def sum_many(self, a: list[float]) -> list[float]: ...

    """
        Takedown the MPC Engine, releasing the resources and dropping connections.
    """
    def takedown(self): ...


"""
    Setup a MPC Engine for adding numbers together using SPDZ
    The engine will connect to the given addresses and listen
    on the first socket address.
    :path_to_pre: path to precomputed triples
    :param my_addr: the address to listen on
    :param others: the addresses to connect to
"""
def spdz(path_to_pre: str, my_addr: str, *others: str) -> Engine: ...


"""
    Setup a MPC Engine for adding numbers together using Shamir Secret Sharing.
    The engine will connect to the given addresses and listen
    on the first socket address.
    :threshold: threshold to use
    :param my_addr: the address to listen on
    :param horsepower: the addresses to connect to
"""
def shamir(threshold: int, my_addr: str, *others: str) -> Engine: ...


"""
    Setup a MPC Engine for adding numbers together using Feldman Verifiable Secret Sharing.
    The engine will connect to the given addresses and listen
    on the first socket address.
    :threshold: threshold to use
    :param my_addr: the address to listen on
    :param horsepower: the addresses to connect to
"""
def feldman(threshold: int, my_addr: str, *others: str) -> Engine: ...


"""
    Setup a MPC Engine for adding numbers together.
    The engine will connect to the given addresses and listen
    on the first socket address.
    :param my_addr: the address to listen on
    :param horsepower: the addresses to connect to
"""
def preproc(num_of_shares: int, *paths_to_pre: str) -> None: ...
