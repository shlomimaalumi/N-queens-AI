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


class solution:
    def __init__(self, time, conflicts):
        self.time = time
        self.conflicts = conflicts
        # self.StepsOrGenerations = StepsOrGenerations


genetic_algorithm_dict = defaultdict(list)
min_conflicts_algorithm_dict = defaultdict(list)
naive_algorithm_dict = defaultdict(list)
sizes = [4, 8, 12, 16]
max_step_avialable = [10, 20, 50, 100, 120, 150, 180, 200]
population_size_avialable = [50, 100]
generations_avialable = [700, 1000]


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

        for max_steps in max_step_avialable:
            conflicts_sum = 0
            for _ in range(10):
                min_conflicts_algorithm = MinConflictsAlgorithm(n, max_steps)
                min_conflicts_algorithm.solve()
                conflicts_sum += min_conflicts_algorithm.get_all_conflicts()

            conflicts_avg = conflicts_sum / 10
            x_vals.append(max_steps)
            y_vals.append(conflicts_avg)

        plt.plot(x_vals, y_vals, label=f'N={n}', marker='o')

    plt.xlabel('Number of Steps')
    plt.ylabel('Average Number of Conflicts')
    plt.title('Average Number of Conflicts vs. Number of Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_success_plot_for_genetic_algorithm():
    """
    X axis: number of generations
    Y axis: conflicts in average for the N and the number of generations
    color for each N and population size
    """
    plt.figure(figsize=(10, 6))

    for n in sizes:
        for population_size in population_size_avialable:
            x_vals = []
            y_vals = []

            for generations in generations_avialable:
                conflicts_sum = 0
                for _ in range(1):
                    genetic_algorithm = GeneticAlgorithmNQueens(n, population_size, 0.1, generations)
                    genetic_algorithm.solve()
                    conflicts_sum += genetic_algorithm.calculate_solution_conflicts()

                conflicts_avg = conflicts_sum / 1
                x_vals.append(generations)
                y_vals.append(conflicts_avg)

            plt.plot(x_vals, y_vals, label=f'N={n}, Population Size={population_size}', marker='o')

    plt.xlabel('Number of Generations')
    plt.ylabel('Average Number of Conflicts')
    plt.title('Average Number of Conflicts vs. Number of Generations')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_running_time_per_n_for_genetic_min_conflicts():
    ns = range(4, 18)
    times = []

    for n in ns:
        print(f"Processing N={n}")
        running_times_sum = 0
        for i in range(109):
            alg = MinConflictsAlgorithm(n, limit_steps=False)
            start = time.time()
            alg.solve()
            end = time.time()
            running_times_sum += end - start


        times.append(running_times_sum)

    plt.figure(figsize=(10, 6))
    plt.plot(ns, times, marker='o', linestyle='-', color='b', label='Average Running Time')
    plt.xlabel('N')
    plt.ylabel('Average Running Time (seconds)')
    plt.title('Running Time vs. N for Genetic Min Conflicts Algorithm')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(times)



def plot_running_time_compare(genetic: dict, min_conflicts: dict, naive_algorithm_dict: dict):
    "x axis: N, y axis: time, three lines: genetic, min conflicts and naive"
    fig, ax = plt.subplots()
    for n in sizes:
        gen_avg = np.mean([sol.time for sol in genetic[n]])
        min_avg = np.mean([sol.time for sol in min_conflicts[n]])
        naive_avg = np.mean([sol.time for sol in naive_algorithm_dict[n]])
        ax.plot(n, gen_avg, 'ro')
        ax.plot(n, min_avg, 'bo')
        ax.plot(n, naive_avg, 'go')
    ax.set(xlabel='N', ylabel='Time',
           title='Time compare')
    ax.grid()
    plt.show()


def plot_conflicts_compare(genetic: dict, min_conflicts: dict, naive: dict):
    "x axis: N, y axis: conflicts, three lines: genetic, min conflicts and naive"
    fig, ax = plt.subplots()
    for n in genetic:
        gen_avg = np.mean([sol.conflicts for sol in genetic[n]])
        min_avg = np.mean([sol.conflicts for sol in min_conflicts[n]])
        naive_avg = np.mean([sol.conflicts for sol in naive[n]])
        ax.plot(n, gen_avg, 'ro')
        ax.plot(n, min_avg, 'bo')
        ax.plot(n, naive_avg, 'go')
    ax.set(xlabel='N', ylabel='Conflicts',
           title='Conflicts compare')
    ax.grid()
    plt.show()


# for n in sizes:
#     genetic_algorithm_dict[n] = []
#     min_conflicts_algorithm_dict[n] = []
#     naive_algorithm_dict[n] = []
#
#     for i in range(32 // n):
#         genetic_algorithm = GeneticAlgorithmNQueens(n, population_size=50, mutation_rate=0.05, max_generations=1000)
#         start = time.time()
#         genetic_algorithm.solve()
#         end = time.time()
#
#         genetic_algorithm_dict[n].append(solution(end - start, genetic_algorithm.calculate_solution_conflicts()))
#
#         min_conflicts_algorithm = MinConflictsAlgorithm(n)
#         start = time.time()
#         min_conflicts_algorithm.solve()
#         end = time.time()
#         min_conflicts_algorithm_dict[n].append(solution(end - start, min_conflicts_algorithm.get_all_conflicts()))
#
#         naive_algorithm = NaiveAlgorithm(n)
#         start = time.time()
#         naive_algorithm.solve()
#         end = time.time()
#         naive_algorithm_dict[n].append(solution(end - start, naive_algorithm.get_all_conflicts()))
#         print("done with N = ", n, " run ", i)

# plot_running_time_compare(genetic_algorithm_dict, min_conflicts_algorithm_dict, naive_algorithm_dict)
# plot_conflicts_compare(genetic_algorithm_dict, min_conflicts_algorithm_dict, naive_algorithm_dict)


# steps_success_plot_for_min_conflicts()

# plot_success_plot_for_genetic_algorithm()
plot_running_time_per_n_for_genetic_min_conflicts()