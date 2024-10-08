import caring
# TODO: We need to make som preprocessing. Non of the participating parties should be allowed to do this, 
# as knowing the other parties preprocedsed values breaks privacy. 
# However for starters, and testing purposses ONLY we will allow party 1 to do it 
# and save it where both party one and party two can find it. 

caring.preproc(12, 0, "./context1.bin", "./context2.bin")
# engine = caring.spdz("./context1.bin", "127.0.0.1:1234", "127.0.0.1:1235")
engine = caring.shamir(2, "127.0.0.1:1234", "127.0.0.1:1235")

res = engine.sum(2.5)
print(f"2.5 - 5 = {res}")

res = engine.sum_many([2.5, 3.5])
print(f"[2.5, 3.5] + [3.2, 0.5] = {res}")

res = engine.sum(3.14159265359)
print(f"pi + pi = {res}")

res = engine.sum(3.14159265359)
print(f"pi - pi = {res}")

res = engine.sum(-1)
print(f"-1 - 2 = {res}")

res = engine.sum(1111.1111)
print(f"1111.1111 + 2222.2222 = {res}")

res = engine.sum(1111.1111)
print(f"1111.1111 - 2222.2222 = {res}")

res = engine.sum(3.23e13)
print(f"3.23e13 + 5.32e13 = {res}")

res = engine.sum(0.0)
print(f"0 + 0 = {res}")

res = engine.sum(0.01)
print(f"0.01 + 0.02 = {res}")

res = engine.sum(8.0)
print(f"8.0 + 2.02 = {res}")

engine.takedown()
