import numpy as np
import caring
import time

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
    
    time_mpc_sum = 0
    rest = 0

    for _ in range(max_iter):
        start = time.time()

        theta = AtA_inv.dot(A.T.dot(b) / n_samples + tau * (rho - u))

        t0 = time.time()
        n = len(u)
        u_and_theta = u.tolist()
        u_and_theta.extend(theta)
        u_and_theta : list[float] = engine.sum_many(u_and_theta)
        time_mpc_sum +=  (time.time() - t0)
        u = np.array(u_and_theta[:n]) / 2.
        theta = np.array(u_and_theta[n:]) / 2.

        rho = np.fmax(theta + u - lam /tau, 0) - np.fmax(-lam/tau - theta - u, 0)
        u = u + theta - rho

        rest +=  (time.time() - start)
        obj_fun_history.append(obj_fun(theta, A, b, lam, n_samples))


    print(f"Avg time spent summing u = {time_mpc_sum / max_iter * 1000} ms")
    #print(f"Avg time spent summing theta = {time_theta_sum / max_iter * 1000} ms")
    print(f"Avg time total = {rest / max_iter * 1000} ms")
    perc = time_mpc_sum / rest * 100
    print(f"Time spent in MPC {perc}%")
    return theta, obj_fun_history
