from caring import Expr, Engine, preproc

preproc(12, 10, "./context1.bin", "./context2.bin")

engine = Engine(
    scheme="spdz-25519",
    address="localhost:1234",
    peers=["localhost:1235"],
    threshold=1,
    preprocessed_path="./context1.bin"
)

[a, b]  = Expr.symmetric_share([23, 3], id=engine.id(), size=2)

c = a * b;

script = c.open()

res = engine.execute(script).as_int()

print(res)
