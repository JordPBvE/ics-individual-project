import numpy as np
import matplotlib.pyplot as plt

def generateEMs(matrix, T, n, epsilon, A, sigma, N):
    srw = (1 / np.sqrt(n)) * generateRWs(int(T * n), N)

    for s in range(1, int(T * n)):
        currentstate = matrix[:, s - 1]

        diff = (
            (1 / epsilon) * (currentstate - currentstate**3 + A * np.cos((s - 1) / n)) * (1 / n)
            + (sigma / np.sqrt(epsilon)) * (srw[:, s] - srw[:, s - 1])
        )

        matrix[:, s] = matrix[:, s - 1] + diff

    return matrix

def generateRWs(steps, N):
    matrix = np.zeros((N, steps))

    for i in range(1, steps):
        matrix[:, i] = matrix[:, i - 1] + np.random.choice([-1, 1], size=N)

    return matrix

# Parameters
N = 100
n = 1000
T = 20
A = 0
epsilon = 0.003
sigma = 0.3


# Initialize and generate paths
initial_matrix = np.zeros((N, int(T * n)))
matrix = generateEMs(initial_matrix, T, n, epsilon, A, sigma, N)

plt.figure(figsize=(12, 6))
for i in range(len(matrix)):  
    plt.plot(np.arange(0, T, 1/n), matrix[i])

plt.title("Generated Paths")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.ylim(-1.5, 1.5)
plt.show()
