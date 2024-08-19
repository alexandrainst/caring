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


# Processing (only for spdz)
print("Preprocessing...")
caring.preproc(100000, "./ctx1.bin", "./ctx2.bin")


print("Running...")
engine = caring.spdz("./ctx1.bin", "127.0.0.1:1234", "127.0.0.1:1235")
theta_1, func_vals = lasso_ADMM(engine, A_1, b_1)
# lets plot the objective values of the function
# to make sure it has converged
plt.plot(func_vals)
plt.ylabel('function values')
plt.xlabel('iterations')

plt.grid()
plt.show()
