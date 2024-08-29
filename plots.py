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

sizes = [4, 8, 12, 16]
max_step_available = [10, 20, 50, 70, 90, 100, 120, 130, 150, 180, 200]
runs_per_n = 100


def steps_success_plot_for_min_conflicts():
    """
    X axis: number of steps
    Y axis: conflicts in average for the N and the number of steps
    color for each N
    """
    plt.figure(figsize=(10, 6))

    for n in sizes:
        x_vals = []
        y_vals = []

        for max_steps in max_step_available:
            conflicts_sum = 0
            for _ in range(runs_per_n):
                min_conflicts_algorithm = MinConflictsAlgorithm(n, max_steps)
                min_conflicts_algorithm.solve()
                conflicts_sum += min_conflicts_algorithm.get_all_conflicts()

            conflicts_avg = conflicts_sum / runs_per_n
            x_vals.append(max_steps)
            y_vals.append(conflicts_avg)

        plt.plot(x_vals, y_vals, label=f'N={n}', marker='o')

    plt.xlabel('Number of Steps')
    plt.ylabel(f' Average Number of Conflicts over {runs_per_n} iterations')
    plt.title('MIN-CONFLICTS:   Average Number of Conflicts vs. Number of Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_running_time_per_n_for_genetic_min_conflicts():
    ns = range(4, 17)
    times = []

    for n in ns:
        print(f"Processing N={n}")
        running_times_sum = 0
        for i in range(runs_per_n):
            alg = MinConflictsAlgorithm(n, limit_steps=False)
            start = time.time()
            alg.solve()
            end = time.time()
            running_times_sum += end - start
        times.append(running_times_sum / runs_per_n)

    plt.figure(figsize=(10, 6))
    plt.plot(ns, times, marker='o', linestyle='-', color='b', label='Average Running Time')
    plt.xlabel('N')
    plt.ylabel('Average Running Time (seconds)')
    plt.title('MIN-CONFLICTS:    Running Time vs. N for Genetic Min Conflicts Algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(times)



sizes = [4, 8]
mutation_rate = 0.1
max_generations = 200
runs_per_n = 10
population_sizes = [100, 150]


def genetic_algorithm_results():
    """
    Runs the genetic algorithm for different board sizes and population sizes.
    Collects average running times and conflicts for each configuration.
    """
    results = defaultdict(list)

    for n in sizes:
        for population_size in population_sizes:
            print(f"Processing N={n}, Population Size={population_size}")
            running_times_sum = 0
            conflicts_sum = 0
            for _ in range(runs_per_n):
                genetic_algorithm = GeneticAlgorithmNQueens(n, population_size, mutation_rate, max_generations)
                start = time.time()
                genetic_algorithm.solve()
                end = time.time()
                running_times_sum += end - start
                conflicts_sum += genetic_algorithm.calculate_solution_conflicts()
            # Store the results: (N, average running time, average conflicts)
            results[population_size].append((n, running_times_sum / runs_per_n, conflicts_sum / runs_per_n))

    return results


def plot_genetic_algorithm_results(results):
    """
    Plots the results of the genetic algorithm:
    1. Running time vs. population size for different N values.
    2. Conflicts vs. population size for different N values.
    """

    # Extract all unique N values to use in the plot
    N_values = sorted(set(n for pop_size in results for n, _, _ in results[pop_size]))

    # First Plot: Running time vs. Population size
    plt.figure(figsize=(12, 6))
    for n in N_values:
        # Extract population sizes and corresponding average running times for each N
        x_vals = [pop_size for pop_size in population_sizes]
        y_vals = [next(time_avg for n_val, time_avg, _ in results[pop_size] if n_val == n) for pop_size in
                  population_sizes]
        plt.plot(x_vals, y_vals, marker='o', label=f'N = {n}')

    plt.xlabel("Population Size")
    plt.ylabel("Average Running Time (s)")
    plt.title("Average Running Time vs. Population Size for Different N Values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Second Plot: Conflicts vs. Population size
    plt.figure(figsize=(12, 6))
    for n in N_values:
        # Extract population sizes and corresponding average conflicts for each N
        x_vals = [pop_size for pop_size in population_sizes]
        y_vals = [next(conflicts_avg for n_val, _, conflicts_avg in results[pop_size] if n_val == n) for pop_size in
                  population_sizes]
        plt.plot(x_vals, y_vals, marker='o', label=f'N = {n}')

    plt.xlabel("Population Size")
    plt.ylabel("Average Conflicts")
    plt.title("Average Conflicts vs. Population Size for Different N Values")
    plt.legend()
    plt.grid(True)
    plt.show()


# Generate results by running the genetic algorithm
results = genetic_algorithm_results()

# Plot the results
plot_genetic_algorithm_results(results)



plot_genetic_algorithm_results()