# Importing required libraries
import matplotlib.pyplot as plt
import numpy as np

# K values in millions
K_values = np.array([1, 5, 10, 50, 100])

# Execution times in seconds for both experiments
# Experiment 1
CPU = np.array([0.003938, 0.021249, 0.04251, 0.20999, 0.42241])

# Experiment 2
GPU = np.array([0.001157, 0.003624, 0.006465, 0.031354, 0.061792])

# Creating the plots
plt.figure(figsize=(10, 6))

# Plot for Experiment 1
plt.plot(K_values, CPU, label='CPU', marker='o')

# Plot for Experiment 2
plt.plot(K_values, GPU, label='GPU', marker='x')

# Setting the scale to log-log
plt.xscale('log')
plt.yscale('log')

# Adding title and labels
plt.title('with Unified Memory')
plt.xlabel('K (Millions)')
plt.ylabel('Execution Time (seconds)')

# Adding a grid for better readability
plt.grid(True)

# Adding a legend to distinguish between the experiments
plt.legend()

# Display the plot
plt.savefig('Q4_2.png')
