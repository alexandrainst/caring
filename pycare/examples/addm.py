import numpy as np
import caring

def obj_fun(theta, A, b, lamb, n_samples):
    # .. the lasso objective function ..
    loss = (0.5 / n_samples) * np.linalg.norm(A.dot(theta) - b)**2
    return loss + lamb * np.sum(np.abs(theta))



def lasso_ADMM(engine: caring.Engine, A, b, max_iter=100, lam=1.):
    # .. initialize variables ..
    tau = 1.
    n_samples, n_features = A.shape
    rho = np.zeros(n_features)
    u = np.zeros(n_features)

    # .. to keep track of progress ..
    obj_fun_history = []

    # .. cache inverse matrix ..
    AtA_inv = np.linalg.pinv(A.T.dot(A) / n_samples + tau * np.eye(n_features))

    for i in range(max_iter):
        theta = AtA_inv.dot(A.T.dot(b) / n_samples + tau * (rho - u))

        u0 : list[float] = engine.sum_many(u.tolist())
        u = np.array(u0) / 2.
        theta0 = engine.sum_many(theta.tolist())
        theta = np.array(theta0) / 2.
        print(f"u = {u}")
        print(f"theta = {theta}")

        rho = np.fmax(theta + u - lam /tau, 0) - np.fmax(-lam/tau - theta - u, 0)
        u = u + theta - rho
        obj_fun_history.append(obj_fun(theta, A, b, lam, n_samples))

    return theta, obj_fun_history
