from caring import Expr, Engine

engine = Engine(scheme="shamir-25519", address="localhost:1234", peers=["localhost:1235"], threshold=1)

[a, b]  = Expr.symmetric_share(23, id=engine.id(), size=2)

c = a + b;

script = c.open()

res = engine.execute(script).as_float()

print(res)
