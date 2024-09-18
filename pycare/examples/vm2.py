from caring import Expr, Engine

engine = Engine(
    scheme="spdz-25519",
    address="localhost:1235",
    peers=["localhost:1234"],
    threshold=1,
    preprocessed_path="./context2.bin"
)

[a, b]  = Expr.symmetric_share([7, 2], id=engine.id(), size=2)

c = a * b;

script = c.open()

res = engine.execute(script).as_int()

print(res)
