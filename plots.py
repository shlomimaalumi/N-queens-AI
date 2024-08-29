import matplotlib.pyplot as plt
import time
import random
import numpy as np
import psutil

from Genetic_algorithm import GeneticAlgorithmNQueens
from MIN_CONFLICTS_algorithm import MinConflictsAlgorithm


def measure_time_for_algorithms(n, max_steps=1000):
    min_conflicts_time = 0
    genetic_algorithm_time = 0

    # Measure time for MinConflictsAlgorithm
    start = time.time()
    min_conflicts = MinConflictsAlgorithm(n)
    min_conflicts.solve()
    min_conflicts_time = time.time() - start

    # Measure time for GeneticAlgorithmNQueens
    start = time.time()
    genetic_algorithm = GeneticAlgorithmNQueens(n)
    genetic_algorithm.evolve()
    genetic_algorithm_time = time.time() - start

    return min_conflicts_time, genetic_algorithm_time


def measure_success_rate(n, trials=10):
    min_conflicts_success = 0
    genetic_algorithm_success = 0

    for _ in range(trials):
        min_conflicts = MinConflictsAlgorithm(n)
        min_conflicts.solve()
        if min_conflicts.termination_criteria():
            min_conflicts_success += 1

        genetic_algorithm = GeneticAlgorithmNQueens(n)
        if genetic_algorithm.evolve():
            genetic_algorithm_success += 1

    return min_conflicts_success / trials, genetic_algorithm_success / trials


def measure_iterations_for_algorithms(n, max_steps=1000):
    min_conflicts_iterations = 0
    genetic_algorithm_iterations = 0

    min_conflicts = MinConflictsAlgorithm(n)
    min_conflicts.solve()
    min_conflicts_iterations = min_conflicts.steps

    genetic_algorithm = GeneticAlgorithmNQueens(n)
    genetic_algorithm.evolve()
    genetic_algorithm_iterations = genetic_algorithm.generations

    return min_conflicts_iterations, genetic_algorithm_iterations


def measure_fitness_progress(n, population_size=50, mutation_rate=0.05, max_generations=1000):
    genetic_algorithm = GeneticAlgorithmNQueens(n, population_size, mutation_rate, max_generations)
    fitness_progress = []

    for generation in range(max_generations):
        genetic_algorithm.evolve()
        fitness_progress.append(genetic_algorithm.fitness(genetic_algorithm.best_solution))

    return fitness_progress


def measure_conflicts_progress(n):
    min_conflicts = MinConflictsAlgorithm(n)
    conflicts_progress = []

    while not min_conflicts.termination_criteria():
        i, j = min_conflicts.get_most_conflicted_queen()
        min_conflicts.board[i][j] = 0
        i, j = min_conflicts.get_best_position()
        min_conflicts.board[i][j] = 1
        conflicts_progress.append(sum(min_conflicts.get_conflicts(i, j) for i in range(n) for j in range(n)))

    return conflicts_progress


def measure_memory_usage_for_algorithms(n):
    process = psutil.Process()
    min_conflicts_memory = 0
    genetic_algorithm_memory = 0

    start_memory = process.memory_info().rss / 1024 / 1024  # Memory usage in MB

    min_conflicts = MinConflictsAlgorithm(n)
    min_conflicts.solve()
    min_conflicts_memory = process.memory_info().rss / 1024 / 1024 - start_memory

    genetic_algorithm = GeneticAlgorithmNQueens(n)
    genetic_algorithm.evolve()
    genetic_algorithm_memory = process.memory_info().rss / 1024 / 1024 - start_memory

    return min_conflicts_memory, genetic_algorithm_memory


def measure_population_size_effect(n, population_sizes):
    average_fitness_scores = []

    for population_size in population_sizes:
        genetic_algorithm = GeneticAlgorithmNQueens(n, population_size=population_size)
        genetic_algorithm.evolve()
        average_fitness_scores.append(genetic_algorithm.fitness(genetic_algorithm.best_solution))

    return average_fitness_scores


def measure_convergence_comparison(n):
    min_conflicts = MinConflictsAlgorithm(n)
    genetic_algorithm = GeneticAlgorithmNQueens(n)

    min_conflicts_conflicts_progress = []
    genetic_algorithm_fitness_progress = []

    while not min_conflicts.termination_criteria():
        i, j = min_conflicts.get_most_conflicted_queen()
        min_conflicts.board[i][j] = 0
        i, j = min_conflicts.get_best_position()
        min_conflicts.board[i][j] = 1
        min_conflicts_conflicts_progress.append(
            sum(min_conflicts.get_conflicts(i, j) for i in range(n) for j in range(n)))

    for generation in range(genetic_algorithm.max_generations):
        if genetic_algorithm.evolve():
            genetic_algorithm_fitness_progress.append(genetic_algorithm.fitness(genetic_algorithm.best_solution))

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(min_conflicts_conflicts_progress)), min_conflicts_conflicts_progress,
             label='Min-Conflicts Conflicts')
    plt.plot(range(len(genetic_algorithm_fitness_progress)), genetic_algorithm_fitness_progress,
             label='Genetic Algorithm Fitness')
    plt.xlabel('Iteration or Generation')
    plt.ylabel('Progress Metric')
    plt.title('Convergence Rate Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main code to generate plots
n_values = list(range(4, 15))  # Example range
population_sizes = [20, 50, 100, 200]

# Running Time vs. Number of Queens
min_conflicts_times = []
genetic_algorithm_times = []

for n in n_values:
    min_time, gen_time = measure_time_for_algorithms(n)
    min_conflicts_times.append(min_time)
    genetic_algorithm_times.append(gen_time)

plt.figure(figsize=(10, 6))
plt.plot(n_values, min_conflicts_times, label='Min-Conflicts Algorithm')
plt.plot(n_values, genetic_algorithm_times, label='Genetic Algorithm')
plt.xlabel('Number of Queens (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Running Time vs. Number of Queens')
plt.legend()
plt.grid(True)
plt.show()

# Success Rate vs. Number of Queens
success_rates = {'Min-Conflicts': [], 'Genetic Algorithm': []}

for n in n_values:
    min_success, gen_success = measure_success_rate(n)
    success_rates['Min-Conflicts'].append(min_success)
    success_rates['Genetic Algorithm'].append(gen_success)

plt.figure(figsize=(10, 6))
plt.plot(n_values, success_rates['Min-Conflicts'], label='Min-Conflicts Algorithm')
plt.plot(n_values, success_rates['Genetic Algorithm'], label='Genetic Algorithm')
plt.xlabel('Number of Queens (N)')
plt.ylabel('Success Rate')
plt.title('Success Rate vs. Number of Queens')
plt.legend()
plt.grid(True)
plt.show()

# Number of Iterations vs. Number of Queens
iterations = {'Min-Conflicts': [], 'Genetic Algorithm': []}

for n in n_values:
    min_iterations, gen_iterations = measure_iterations_for_algorithms(n)
    iterations['Min-Conflicts'].append(min_iterations)
    iterations['Genetic Algorithm'].append(gen_iterations)

plt.figure(figsize=(10, 6))
plt.plot(n_values, iterations['Min-Conflicts'], label='Min-Conflicts Algorithm')
plt.plot(n_values, iterations['Genetic Algorithm'], label='Genetic Algorithm')
plt.xlabel('Number of Queens (N)')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs. Number of Queens')
plt.legend()
plt.grid(True)
plt.show()

# Fitness Value vs. Generation (For Genetic Algorithm Only)
for n in [8]:  # Example value for demonstration
    fitness_progress = measure_fitness_progress(n)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fitness_progress)), fitness_progress)
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Value vs. Generation')
    plt.grid(True)
    plt.show()

# Conflicts vs. Iterations (For Min-Conflicts Algorithm Only)
for n in [8]:  # Example value for demonstration
    conflicts_progress = measure_conflicts_progress(n)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(conflicts_progress)), conflicts_progress)
    plt.xlabel('Iteration Number')
    plt.ylabel('Number of Conflicts')
    plt.title('Conflicts vs. Iterations')
    plt.grid(True)
    plt.show()

# Memory Usage vs. Number of Queens
memory_usages = {'Min-Conflicts': [], 'Genetic Algorithm': []}

for n in n_values:
    min_memory, gen_memory = measure_memory_usage_for_algorithms(n)
    memory_usages['Min-Conflicts'].append(min_memory)
    memory_usages['Genetic Algorithm'].append(gen_memory)

plt.figure(figsize=(10, 6))
plt.plot(n_values, memory_usages['Min-Conflicts'], label='Min-Conflicts Algorithm')
plt.plot(n_values, memory_usages['Genetic Algorithm'], label='Genetic Algorithm')
plt.xlabel('Number of Queens (N)')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage vs. Number of Queens')
plt.legend()
plt.grid(True)
plt.show()

# Average Fitness Score vs. Population Size (For Genetic Algorithm)
average_fitness_scores = measure_population_size_effect(8, population_sizes)  # Example N

plt.figure(figsize=(10, 6))
plt.plot(population_sizes, average_fitness_scores)
plt.xlabel('Population Size')
plt.ylabel('Average Fitness Score')
plt.title('Average Fitness Score vs. Population Size')
plt.grid(True)
plt.show()

# Convergence Rate Comparison
measure_convergence_comparison(8)  # Example N
