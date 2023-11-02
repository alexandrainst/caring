import pycare

print("Hi am Python, and today we are going to add numbers")
pycare.setup("127.0.0.1:1234", "127.0.0.1:1235")
res = pycare.sum([2.5, 3.5])
print(f"Hi am Python, and I just did MPC, here is my result {res}")

print("Let's try some more")
res = pycare.sum(2.5)
print(f"2.5 - 5 = {res}")

res = pycare.sum(3.14159265359)
print(f"pi + pi = {res}")

res = pycare.sum(3.14159265359)
print(f"pi - pi = {res}")

res = pycare.sum(-1)
print(f"-1 - 2 = {res}")

res = pycare.sum(1111.1111)
print(f"1111.1111 + 2222.2222 = {res}")

res = pycare.sum(1111.1111)
print(f"1111.1111 - 2222.2222 = {res}")

res = pycare.sum(3.23e13)
print(f"3.23e13 + 5.32e13 = {res}")

res = pycare.sum(0.0)
print(f"0 + 0 = {res}")

res = pycare.sum(0.01)
print(f"0.01 + 0.02 = {res}")
