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
    matrix[:,0] = 0

    for i in range(1, steps):
        matrix[:, i] = matrix[:, i - 1] + np.random.choice([-1, 1], size=N)

    return matrix

def average_traveltime(matrix, n, N):
    at_1_times = np.argmax(matrix>1, axis=1) 
    
    zero_count = np.count_nonzero(at_1_times == 0)

    if zero_count > 0: 
        print(f'{zero_count} paths have not yet reached value 1')
        return np.zeros(N), -1
    

    return at_1_times, np.average(at_1_times) * (1/n)


# Parameters
N = 1000
n = 1000
T = 300
A = 0
epsilon = 0.003
sigma = 0.3


# Initialize and generate paths
initial_matrix = np.zeros((N, int(T * n)))
initial_matrix[:, 0] = -1

matrix = generateEMs(initial_matrix, T, n, epsilon, A, sigma, N)

at_1_times, mean_travel_time = average_traveltime(matrix, n, N)

print(f'mean travel time: {mean_travel_time}')

plt.figure(figsize=(12, 6))
for i in range(N):  
    # Plot only the masked values
    plt.plot(np.arange(0, at_1_times[i]+1)/n, matrix[i][:at_1_times[i]+1])

plt.title("Generated Paths")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.ylim(-1.5, 1.5)
plt.show()
