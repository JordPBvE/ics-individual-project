import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from joblib import Parallel, delayed


def initialconditions(N, T, dt):
    w = np.random.uniform(-1/2, 1/2, N)
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

def compute_r_infty(K, N, T, dt):
    print(K)
    w, initialmatrix = initialconditions(N, T, dt)
    matrix = euler(w, initialmatrix, K, N, T, dt)
    r_infty = (1 / N) * np.abs(np.sum(np.exp(1j * matrix[:, -1])))
    return r_infty

N = 2000
T = 200
dt = 0.05

Kmin = 0
Kmax = 1.5
dK = 0.03


K_values = np.arange(Kmin, Kmax + dK, dK)

# Parallel computation of r_infty
r_infties = Parallel(n_jobs=-1)(delayed(compute_r_infty)(K, N, T, dt) for K in K_values)


plt.plot(K_values, r_infties, marker='o')
plt.xlabel("Coupling Strength (K)")
plt.ylabel("Order Parameter ($r_\infty$)")
plt.title("Order Parameter vs Coupling Strength")
plt.grid(True)
plt.show()