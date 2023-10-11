import pycare

print("Hi am Python, and today we are going to add numbers")
pycare.setup("127.0.0.1:1234", "127.0.0.1:1235")
res = pycare.sum(2)
print(f"Hi am Python, and I just did MPC, here is my result {res}")
