import pycare

print("Hi am Python, and today we are going to add numbers")
pycare.setup("127.0.0.1:1235", "127.0.0.1:1234")
res = pycare.sum(3)
print(f"Hi am Python, and I just did MPC, here is my result {res}")
