import time

from matplotlib import pyplot as plt

from naive_algorithm import NaiveAlgorithm


def plot_algorithm_comparisons():
    # Data for the algorithms
    n = 14
    algorithms = ['Min-Conflict', 'Genetic', 'Naive']
    execution_times = [0.06, 0.491111, 348.63448214530945]

    plt.figure(figsize=(10, 5))
    plt.bar(algorithms, execution_times, color=['blue', 'green', 'red'])
    plt.title(f"Execution Time Comparison for N={n}")
    plt.ylabel('Execution Time (seconds)')
    plt.yscale('log')
    plt.show()


plot_algorithm_comparisons()