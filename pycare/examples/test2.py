from caring import Engine

engine = Engine(
    scheme="shamir-25519",
    address="localhost:1235",
    peers=["localhost:1234"],
    threshold=2
)

res = engine.sum(-5)
print(f"2.5 - 5 = {res}")

res = engine.sum([3.2, 0.5])
print(f"[2.5, 3.5] + [3.2, 0.5] = {res}")

res = engine.sum(3.14159265359)
print(f"pi + pi = {res}")

res = engine.sum(-3.14159265359)
print(f"pi - pi = {res}")

res = engine.sum(-2)
print(f"-1 - 2 = {res}")

res = engine.sum(2222.2222)
print(f"1111.1111 + 2222.2222 = {res}")

res = engine.sum(-2222.2222)
print(f"1111.1111 - 2222.2222 = {res}")

res = engine.sum(5.32e13)
print(f"3.23e13 + 5.32e13 = {res}")

res = engine.sum(0.0)
print(f"0 + 0 = {res}")

res = engine.sum(0.02)
print(f"0.01 + 0.02 = {res}")

res = engine.sum(2.02)
print(f"8.0 + 2.02 = {res}")

