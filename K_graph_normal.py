import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from joblib import Parallel, delayed
from scipy.integrate import quad

def g(omega):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * omega**2)

def compute_integral(K, r, g):
    # Define the integrand function
    def integrand(theta):
        return np.cos(theta)**2 * g(K * r * np.sin(theta))
    
    # Compute the integral from -π/2 to π/2
    result, _ = quad(integrand, -np.pi/2, np.pi/2)
    return K * result


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

def compute_r_infty(K, N, T, dt):
    w, initialmatrix = initialconditions(N, T, dt)
    matrix = euler(w, initialmatrix, K, N, T, dt)
    r_infty = (1 / N) * np.abs(np.sum(np.exp(1j * matrix[:, -1])))
    return r_infty

N = 1000
T = 100
dt = 0.01

Kmin = 0
Kmax = 5
dK = 0.5


K_values = np.arange(Kmin, Kmax + dK, dK)

# Parallel computation of r_infty
r_infties = Parallel(n_jobs=-1)(delayed(compute_r_infty)(K, N, T, dt) for K in K_values)


plt.plot(K_values, r_infties, marker='o')
plt.xlabel("Coupling Strength (K)")
plt.ylabel("Order Parameter ($r_\infty$)")
plt.title("Order Parameter vs Coupling Strength")
plt.grid(True)
plt.show()

for (K, r) in zip(K_values, r_infties):
    print(compute_integral(K, r, g))