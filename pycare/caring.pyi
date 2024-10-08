class Id:
    """
        Id of a party
    """


class Opened:
    """
        An opened value.

        Use the Engine to evaluate it
    """


class Computed:
    """ 
        A computed result, needs to be recast into a value.

        The size needs to be manually tracked,
        in the case of a scalar the size will be 1,

        and for any vectors given it will be the corresponding size
    """


    def as_float(self) -> list[float]:
        """
            Cast as a float
        """



    def as_int(self) -> list[int]:
        """
            Cast as an integer
        """




class Expr:
    """
        An expression of an MPC routine

        Note:
            This allows mixing between different vector sizes and scalars,
            it will however be a runtime error to add, subcract or multiply
            different sized vectors.
    """

    @staticmethod
    def share(num: int | float | list[int] | list[float]) -> Expr:
         """ Share a given value or vector

         Warning:
             - Integer-mode only supports unsigned (non-negative) integers.
               Negative integers will be converted to floats
             - Floats will be converted to fixed point, and as such
               multiplication is currently unsupported.

         Returns:
             An expression for the given share
         """


    @staticmethod
    def recv(id: Id) -> Expr:
        """ 
        Receive a share from another party

        Returns:
            An expression for the given share
        """


    @staticmethod
    def symmetric_share(num: int | float | list[int] | list[float], id: Id, size: int) -> list[Expr]:
        """ Secret-share (symmetrically, all-parties at-once) a given value or vector

         Warning:
             - Integer-mode only supports unsigned (non-negative) integers.
               Negative integers will be converted to floats
             - Floats will be converted to fixed point, and as such
               multiplication is currently unsupported.

        Returns:
            A list of expressions, ordered by Ids
        """

    def open(self) -> Opened:
        """ Open the given value

        An opened value needs to be processed in the engine to compute it's result
        """

    def __add__(self, other: Expr) -> Expr: ...
    def __sub__(self, other: Expr) -> Expr: ...
    def __mul__(self, other: Expr) -> Expr: ...
    def __iadd__(self, other: Expr) -> None: ...
    def __isub__(self, other: Expr) -> None: ...
    def __imul__(self, other: Expr) -> None: ...

class Engine:
    """ Create a new Engine

    Args:
        scheme: 'shamir-32' or 'shamir-25519' or 'spdz-32' or 'spdz-25519' or 'feldman-25519'
        address: address to listen on
        peers: peer addresses (party members) to connect to
        multithreaded: use a multithreaded runtime
        threshold: in case of a threshold scheme, use the given threshold
        preprocessed: if using spdz, use the following preprocessed data file.
    """
    def __init__(
        self,
        scheme: str,
        address: str,
        peers: list[str],
        multithreaded: bool = False,
        threshold: int | None = None,
        preprocessed_path: str | None = None,
    ) -> None: ...

    def execute(self, script: Opened) -> Computed:
        """
            Execute the opened value to evaluate it.
            This runs the nesscary protocols to output it.

            Errors: This might error if the underlying computation fails

            Returns:
                The result of the computation

        """


    def id(self) -> Id:
        """
            Returns:
                Your id in the network
        """

    def peers(self) -> list[Id]:
        """
            List the party members in the network

            Returns:
                a list of party id's for each member
        """


"""
    Preprocess mult. triples and preshares

    :param num_of_shares: number of shares and triples
    :param path_to_pre: path(s) to write the preprocessed material
    :param scheme: scheme to share in ('spdz-25519'|'spdz-32')

    The paths parameter also implicitly defines the amount of parties
    that there will preprocessed for, as each will get their own file.
"""
def preproc(num_shares: int, num_triplets: int, *paths_to_pre: str, scheme: str = "spdz-25519") -> None: ...

#
# Old stuff
#

class OldEngine:
    def sum(self, a: float) -> float:
        """
            Performs a summation with the connected parties.
            Returns the sum of all the numbers.
            :param a: number to summate with
        """

    def sum_many(self, a: list[float]) -> list[float]:
        """
            Performs a summation of a vector with the connected parties.
            Returns the sum of all the vector (element-wise) of all the numbers.
            :param a: vector to summate with.

            Note: that all parties must supply the same length!
        """

    def takedown(self) -> None:
        """
            Takedown the MPC Engine, releasing the resources and dropping connections.
        """


def spdz(path_to_pre: str, my_addr: str, *others: str) -> Engine:
    """
        Setup a MPC Engine for adding numbers together using SPDZ
        The engine will connect to the given addresses and listen
        on the first socket address.
        :path_to_pre: path to precomputed triples
        :param my_addr: the address to listen on
        :param others: the addresses to connect to
    """


def shamir(threshold: int, my_addr: str, *others: str) -> Engine:
    """
        Setup a MPC Engine for adding numbers together using Shamir Secret Sharing.
        The engine will connect to the given addresses and listen
        on the first socket address.
        :threshold: threshold to use
        :param my_addr: the address to listen on
        :param horsepower: the addresses to connect to
    """


def feldman(threshold: int, my_addr: str, *others: str) -> Engine:
    """
        Setup a MPC Engine for adding numbers together using Feldman Verifiable Secret Sharing.
        The engine will connect to the given addresses and listen
        on the first socket address.
        :threshold: threshold to use
        :param my_addr: the address to listen on
        :param horsepower: the addresses to connect to
    """


