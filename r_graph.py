import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from joblib import Parallel, delayed


def initialconditions(N, T, dt):
    w = np.random.normal(0, 1, N)
    matrix = np.zeros((N, int(T/dt)))
    matrix[:, 0] = np.random.uniform(0, 2* np.pi, N)

    return w, matrix

@jit
def euler(w, matrix, K, N, T, dt):
    for s in range(1, int(T/dt)):
        currentstate = matrix[:, s-1]

        # Compute the phase differences using broadcasting
        phase_diff = currentstate - currentstate[:, None] 
        
        # Compute the sine matrix
        sinematrix = np.sin(phase_diff)

        matrix[:, s] = matrix[:, s - 1] + dt * (w + (K / N) * np.sum(sinematrix, axis=1))

    return matrix

def compute_rs(K, N, T, dt):
    w, initialmatrix = initialconditions(N, T, dt)
    matrix = euler(w, initialmatrix, K, N, T, dt)
    rs = (1 / N) * np.abs(np.sum(np.exp(1j * matrix), axis=0))
    return rs

N = 1000
T = 100
dt = 0.01

t_values = np.arange(0, T, dt)
K_values = [1,2]

# Parallel computation of r_infty
r_trajectories = Parallel(n_jobs=-1)(delayed(compute_rs)(K, N, T, dt) for K in K_values)

for K, rs in zip(K_values, r_trajectories):
    plt.plot(t_values, rs, label=f'K = {K}')

plt.xlabel("Time ($t$)")
plt.ylabel("Order Parameter ($r_{\infty}$)")
plt.title("Order Parameter over Time ")
plt.grid(True)
plt.legend()
plt.show()