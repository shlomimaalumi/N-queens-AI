from statistics import mean, median
from collections import defaultdict
from time import time
from statistics import mean, median
from collections import defaultdict
from time import time

import numpy as np
from tabulate import tabulate
import random


class GeneticAlgorithmNQueens:
    """
    A class to represent a genetic algorithm for solving the N-Queens problem.

    Attributes:
    n (int): The number of queens and the size of the board (n x n).
    population_size (int): The size of the population.
    mutation_rate (float): The mutation rate.
    max_generations (int): The maximum number of generations.
    population (list): The current population of solutions.
    best_solution (list): The best solution found.
    generations (int): The number of generations evolved.
    """

    def __init__(self, n, population_size, mutation_rate, max_generations):
        """
        Initializes the genetic algorithm with the given parameters.

        Parameters:
        n (int): The number of queens and the size of the board (n x n).
        population_size (int): The size of the population.
        mutation_rate (float): The mutation rate.
        max_generations (int): The maximum number of generations.
        """
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.init_population()
        self.best_solution = None
        self.generations = 0
        self.tournament_size = 5

    def init_population(self):
        """
        Initializes the population with random boards.

        Returns:
        list: A list of random boards.
        """
        return [self.random_board() for _ in range(self.population_size)]

    def random_board(self):
        """
        Generates a random board configuration.

        Returns:
        list: A random board configuration.
        """
        return [random.randint(0, self.n - 1) for _ in range(self.n)]

    def fitness(self, board):
        """
        Calculates the fitness of a board configuration.

        Parameters:
        board (list): A board configuration.

        Returns:
        int: The fitness score of the board.
        """
        max_pairs = self.n * (self.n - 1) // 2
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                    conflicts += 1
        return max_pairs - conflicts

    def select_parents(self):
        """
        Selects two parents from the population using tournament selection.

        Returns:
        list: A list containing two parent boards.
        """
        return [max(random.sample(self.population, self.tournament_size), key=self.fitness) for _ in range(2)]

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent boards.

        Parameters:
        parent1 (list): The first parent board.
        parent2 (list): The second parent board.

        Returns:
        list: The child board resulting from the crossover.
        """
        crossover_point = random.randint(0, self.n - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, board):
        """
        Mutates a board configuration with a given mutation rate.

        Parameters:
        board (list): The board configuration to mutate.

        Returns:
        list: The mutated board configuration.
        """
        if random.random() < self.mutation_rate:
            board[random.randint(0, self.n - 1)] = random.randint(0, self.n - 1)
        return board

    def solve(self):
        """
        Evolves the population to find a solution to the N-Queens problem.
        running time is O(max_generations * population_size * n ^ 2)

        Returns:
        bool: True if a solution is found, False otherwise.
        """
        for generation in range(self.max_generations):  # O(max_generations)
            new_population = []
            for _ in range(self.population_size):  # O(population_size)
                parent1, parent2 = self.select_parents()  # O(tournament_size)
                child = self.crossover(parent1, parent2)  # O(n)
                child = self.mutate(child)  # O(1)
                new_population.append(child)

            self.population = new_population
            best_board = max(self.population, key=self.fitness)  # O(population_size* n^2)
            if self.fitness(best_board) == self.n * (self.n - 1) // 2:
                self.best_solution = best_board
                self.generations = generation + 1
                return True

        self.best_solution = max(self.population, key=self.fitness)
        self.generations = self.max_generations
        return False

    def calculate_solution_conflicts(self):
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.best_solution[i] == self.best_solution[j] or abs(
                        self.best_solution[i] - self.best_solution[j]) == abs(i - j):
                    conflicts += 1
        return conflicts // 2


def mutation_rate_vs_success_rate_vs_steps(sizes=[4, 6, 8, ], mutation_rates=[0.1, 0.2, 0.7, 1]):
    success_rates = defaultdict(list)
    steps_data = defaultdict(list)
    running_times_data = defaultdict(list)
    fail_data = defaultdict(list)

    for n in sizes:
        print(n)
        for mutation_rate in mutation_rates:
            iterations = 2
            running_times = []
            steps = []
            conflicts = []
            success = 0
            for i in range(iterations):
                alg = GeneticAlgorithmNQueens(n, max(10, n), mutation_rate, 1000 * n ** 2)
                start = time()
                alg.solve()
                if alg.calculate_solution_conflicts() == 0:
                    success += 1
                    running_times.append(time() - start)
                    steps.append(alg.generations)
                else:
                    conflicts.append(alg.calculate_solution_conflicts())

            success_rates[(n, mutation_rate)].append(success / iterations)

            # Calculate steps and running time
            steps_data[(n, mutation_rate)].append((mean(steps) if steps else 0, median(steps) if steps else 0))
            running_times_data[(n, mutation_rate)].append(mean(running_times) if running_times else 0)

            # Check for conflicts before calculating stats
            if conflicts:
                fail_data[(n, mutation_rate)].append((mean(conflicts), median(conflicts)))
            else:
                fail_data[(n, mutation_rate)].append((0, 0))  # No conflicts, so set mean and median to 0

    success_table = []
    fail_table = []

    for (n, mutation_rate), success_rate in success_rates.items():
        success_table.append([n, mutation_rate, success_rate[0], steps_data[(n, mutation_rate)][0][0],
                              steps_data[(n, mutation_rate)][0][1], running_times_data[(n, mutation_rate)][0]])

    for (n, mutation_rate), fail_stats in fail_data.items():
        fail_table.append(
            [n, mutation_rate, 1 - success_rates[(n, mutation_rate)][0], fail_stats[0][0], fail_stats[0][1]])

    print("Success Rates:")
    print(tabulate(success_table, headers=["N", "Mutation Rate", "Success Rate", "Average Steps", "Median Steps",
                                           "Average Running Time"], tablefmt="grid"))

    print("Fail Rates:")
    print(tabulate(fail_table, headers=["N", "Mutation Rate", "Fail Rate", "Average Conflicts", "Median Conflicts"],
                   tablefmt="grid"))


#
#
# def min_conflicts_algorithm_results():
#     steps = defaultdict(list)
#     fails = defaultdict(list)
#     running_time = defaultdict(list)
#     runs_per_n1 = 100
#
#     for n in [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 52, 64]:
#         print(n)
#         for i in range(runs_per_n1):
#             population_size = max(10 , n)
#             alg = GeneticAlgorithmNQueens(n, population_size, 0.1, 1000)
#             start = time()
#             alg.solve()
#             if alg.get_all_conflicts() == 0:
#                 running_time[n].append(time.time() - start)
#                 steps[n].append(alg.steps)
#             else:
#                 fails[n].append(alg.get_all_conflicts())
#
#     # Prepare data for success rates
#     success_data = []
#     for n, s in steps.items():
#         success_data.append([n, len(s) / runs_per_n1, average(s), median(s), average(running_time[n])])
#
#     # Prepare data for fail rates
#     fail_data = []
#     for n, s in fails.items():
#         fail_data.append([n, len(s) / runs_per_n1, average(s), median(s)])
#
#     # Print success rates table
#     print("Success Rates:")
#     print(tabulate(success_data, headers=["N", "Success Rate", "Average Steps", "Median Steps, running time"],
#                    tablefmt="grid"))
#
#     # Print fail rates table
#     print("Fail Rates:")
#     print(tabulate(fail_data, headers=["N", "Fail Rate", "Average Conflicts", "Median Conflicts"], tablefmt="grid"))


print("start")
mutation_rate_vs_success_rate_vs_steps()
