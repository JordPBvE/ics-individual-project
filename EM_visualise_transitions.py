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

def isolate_transitions(matrix):
    # Filter rows based on range condition
    matrix = matrix[np.ptp(matrix, axis=1) > 2]
    matrix[matrix < -1] = -1
    matrix[matrix >  1] = 1

    transitions = []

    for path in matrix:
        possible_transitions = []
        path_transitions = []
        intermediate = -1

        for s, x in enumerate(path):
            if intermediate != -1 and abs(x) >= 0.99:
                # Record transition start and end
                possible_transitions.append((intermediate - 1, s))
                intermediate = -1
            elif intermediate == -1 and abs(x) < 0.99:
                # Mark start of a potential transition
                intermediate = s

        # Filter valid transitions
        for (b, e) in possible_transitions:
            if abs(path[e] - path[b]) > 1.5 and b != -1:
                path_transitions.append((b, e))

        transitions += path_transitions

        preserve_mask = np.zeros_like(path, dtype=bool)

        # Mark indices within transition intervals as True
        for (b, e) in path_transitions:
            path[b] *= 0.999
            path[e] *= 0.999
            preserve_mask[b:e + 1] = True

        # Adjust all other indices in the path
        for i in range(len(path)):
            if not preserve_mask[i]:  # Only adjust if not in a transition interval
                path[i] = -1 if abs(path[i] - (-1)) < abs(path[i] - 1) else 1

    return matrix, mean_transition_time(transitions, n)

def mean_transition_time(transitions, n):
    if len(transitions) == 0: return -1, 0

    return np.sum([(e-b) for (b, e) in transitions])/(len(transitions) * n), len(transitions)


# Parameters
N = 100
n = 1000
T = 10
A = 0
epsilon = 0.003
sigma = 0.3


# Initialize and generate paths
initial_matrix = np.zeros((N, int(T * n)))
matrix = generateEMs(initial_matrix, T, n, epsilon, A, sigma, N)

matrix, (transitiontime, n_transitions) = isolate_transitions(matrix)

print(f'{n_transitions} transitions with an average transition time of {transitiontime}')

plt.figure(figsize=(12, 6))
for i in range(len(matrix)):  
    # Mask out the parts where the y-value is 1 or -1
    mask = np.abs(matrix[i]) != 1
    
    # Plot only the masked values
    plt.plot(np.arange(0, T, 1/n)[mask], matrix[i][mask])

plt.title("Generated Paths")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.ylim(-1.5, 1.5)
plt.show()
