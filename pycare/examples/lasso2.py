from addm import lasso_ADMM
import caring
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n_samples, n_features = 100, 100
A = np.random.randn(n_samples, n_features)
w = np.random.randn(n_features)
b = A.dot(w) + np.random.randn(n_samples)

# .. make b be {-1, 1} since its a classification problem ..
b = np.sign(A.dot(w) + np.random.randn(n_samples))


A_1, A_2 = np.array_split(A, 2)
b_1, b_2 = np.array_split(b, 2)


engine = caring.Engine(
    scheme="spdz-25519",
    address="localhost:1235",
    peers=["localhost:1234"],
    threshold=1,
    preprocessed_path="./context2.bin"
)

theta_1, func_vals = lasso_ADMM(engine, A_2, b_2)
# lets plot the objective values of the function
# to make sure it has converged
plt.plot(func_vals)
plt.ylabel('function values')
plt.xlabel('iterations')

plt.grid()
plt.show()
