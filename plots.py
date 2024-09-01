import matplotlib.pyplot as plt
from collections import defaultdict
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import time
import random
import numpy as np
import psutil

from Genetic_algorithm import GeneticAlgorithmNQueens
from MIN_CONFLICTS_algorithm import MinConflictsAlgorithm
import Genetic_algorithm
from naive_algorithm import NaiveAlgorithm

sizes = [4,6,10]
max_step_available = range(10, 400, 20)
runs_per_n = 4
mutation_rate = 0.1
max_generations = range(10, 400, 20)


def plot_time_vs_n():
    """Plot the average time taken to solve the n-queens problem for different algorithms.
    red color = Naive
    green color = MinConflicts
    blue color = Genetic
    X axis: N
    Y axis: Average Time
    """
    # Initialize a dictionary to store times for each algorithm
    times = defaultdict(list)

    for n in sizes:
        naive_times = []
        min_conflicts_times = []
        genetic_times = []

        # Run each algorithm runs_per_n times and collect the times
        for j in range(runs_per_n):
            print(f"Running for N={n}, run={j + 1}/{runs_per_n}")
            start = time.time()
            NaiveAlgorithm(n).solve()
            naive_times.append(time.time() - start)

            start = time.time()
            MinConflictsAlgorithm(n, 100).solve()
            min_conflicts_times.append(time.time() - start)

            start = time.time()
            # GeneticAlgorithmNQueens(n, 12, mutation_rate, 100).solve()
            Genetic_algorithm.GeneticAlgorithmNQueens(n, 12).solve()

            genetic_times.append(time.time() - start)

        # Calculate the average times for each algorithm
        times['Naive'].append(np.mean(naive_times))
        times['MinConflicts'].append(np.mean(min_conflicts_times))
        times['Genetic'].append(np.mean(genetic_times))

    # Plot the results
    plt.figure()
    plt.title('Average Time vs N')
    plt.xlabel('N')
    plt.ylabel('Average Time (s)')

    # Plot lines with markers for each algorithm
    plt.plot(sizes, times['Naive'], 'o-r', label='Naive')
    plt.plot(sizes, times['MinConflicts'], 'o-g', label='MinConflicts')
    plt.plot(sizes, times['Genetic'], 'o-b', label='Genetic')

    plt.legend()
    plt.show()

# Call the function to generate the plot


def plot_time_vs_max_steps():
    times = defaultdict(list)
    for max_steps in max_step_available:
        for _ in range(runs_per_n):
            start = time.time()
            MinConflictsAlgorithm(8, max_steps).solve()
            times[max_steps].append(time.time() - start)
    plt.figure()
    plt.title('Time vs Max Steps')
    plt.xlabel('Max Steps')
    plt.ylabel('Time (s)')
    for max_steps, t in times.items():
        plt.scatter([max_steps] * len(t), t, color='blue')
    plt.show()

def plot_time_vs_max_generations():
    times = defaultdict(list)
    for max_generation in max_generations:
        for _ in range(runs_per_n):
            start = time.time()
            GeneticAlgorithmNQueens(8, 8, mutation_rate, max_generation).solve()
            times[max_generation].append(time.time() - start)
    plt.figure()
    plt.title('Time vs Max Generations')
    plt.xlabel('Max Generations')
    plt.ylabel('Time (s)')
    for max_generation, t in times.items():
        plt.scatter([max_generation] * len(t), t, color='blue')
    plt.show()

plot_time_vs_n()