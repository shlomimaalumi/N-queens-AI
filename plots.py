import threading
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import Genetic_algorithm
from Genetic_algorithm import GeneticAlgorithmNQueens
from MIN_CONFLICTS_algorithm import MinConflictsAlgorithm
from naive_algorithm import NaiveAlgorithm

sizes = [4, 6, 10, 13]
max_step_available = range(10, 400, 20)
runs_per_n = 2
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
            sol = Genetic_algorithm.GeneticAlgorithmNQueens(n, 1000, 0.8, 5000)
            ret_val = sol.solve()
            if not ret_val:
                print("no solution found")
            # NQueensGenetic(nq=n, population_size=500, max_generations=200).solve()

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


def plot_mutation_rate_vs_n():
    """
    color red for N=4, Blue for N=6, Green for N=10, Yellow for N=13
    X axis: Mutation Rate for each one of range(0.05, 1, 0.05)
    Y axis: success rate (no conflicts)
    this function use threads to run the algorithm multiple times and count and print each success
    :return:
    """
    mutation_rates = np.arange(0.05, 1, 0.1)
    success_rates = defaultdict(list)
    for n in sizes:
        for mutation_rate in mutation_rates:
            success_count = 0
            iterations = 20
            # using threads to run the algorithm multiple times and count and print each success
            # create a list of threads
            threads = []
            sols = [GeneticAlgorithmNQueens(n, 5 * n ** 2, mutation_rate, 50 * n) for _ in range(iterations)]
            for i in range(iterations):
                t = threading.Thread(target=sols[i].solve)
                threads.append(t)
                t.start()
            # wait for all threads to finish
            for t in threads:
                t.join()
            for sol in sols:
                if sol.calculate_solution_conflicts() == 0:
                    success_count += 1
            success_rates[n].append(success_count / iterations)
    plt.figure()
    plt.title('Mutation Rate vs N')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Success Rate')
    colors = ['r', 'b', 'g', 'y']
    for i, n in enumerate(sizes):
        plt.plot(mutation_rates, success_rates[n], 'o-', color=colors[i], label=f'N={n}')
    plt.legend()
    plt.show()


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


def plot_min_conflict_success_rate():
    success_rates = defaultdict(list)
    for n in sizes:
        for max_steps in max_step_available:
            success_count = 0
            iterations = 20 * n ** 2
            threads = []
            sols = [MinConflictsAlgorithm(n, max_steps) for _ in range(iterations)]
            for i in range(iterations):
                t = threading.Thread(target=sols[i].solve)
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            for sol in sols:
                if sol.get_all_conflicts() == 0:
                    success_count += 1
            success_rates[n].append(success_count / iterations)
    plt.figure()
    plt.title('Success Rate vs Max Steps')
    plt.xlabel('Max Steps')
    plt.ylabel('Success Rate')
    colors = ['r', 'b', 'g', 'y']
    for i, n in enumerate(sizes):
        plt.plot(max_step_available, success_rates[n], 'o-', color=colors[i], label=f'N={n}')
    plt.legend()
    plt.show()


def plt_avarege_running_time_of_min_conflicts():
    times = defaultdict(list)
    for n in sizes:
        for _ in range(runs_per_n):
            alg = MinConflictsAlgorithm(n, 100 * n)
            start = time.time()
            alg.solve()
            if alg.get_all_conflicts() == 0:
                times[n].append(time.time() - start)
    plt.figure()
    plt.title('Average Time vs N')
    plt.xlabel('N')
    plt.ylabel('Average Time (s)')
    for n, t in times.items():
        plt.scatter([n] * len(t), t, color='blue')


def plot_step_to_success_min_conflicts():
    steps = defaultdict(list)
    for n in sizes:
        for _ in range(runs_per_n):
            alg = MinConflictsAlgorithm(n, 100 * n)
            alg.solve()
            if alg.get_all_conflicts() == 0:
                steps[n].append(alg.steps)
    plt.figure()
    plt.title('Steps to Success vs N')
    plt.xlabel('N')
    plt.ylabel('Steps')
    for n, s in steps.items():
        plt.scatter([n] * len(s), s, color='blue')
    plt.show()


# plot_time_vs_n()
# plot_mutation_rate_vs_n()
# plot_time_vs_max_steps()
# plot_time_vs_max_generations()
# plot_min_conflict_success_rate()
# plt_avarege_running_time_of_min_conflicts()
# plot_step_to_success_min_conflicts()

N = 16
start = time.time()
NaiveAlgorithm(N).solve()
print(f"Naive algorithm took {time.time() - start} seconds")
