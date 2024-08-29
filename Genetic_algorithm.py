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
        tournament_size = 5
        return [max(random.sample(self.population, tournament_size), key=self.fitness) for _ in range(2)]

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
        Evolves the population over a number of generations to find the best solution.
        Running time is O(max_generations * population_size * n).

        Returns:
        bool: True if a solution is found, False otherwise.
        """
        for generation in range(self.max_generations):
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population
            best_board = max(self.population, key=self.fitness)
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
        return conflicts / 2


if __name__ == '__main__':
    n = 8
    population_size = 100
    mutation_rate = 0.1
    max_generations = 1000
    genetic_algorithm = GeneticAlgorithmNQueens(n, population_size, mutation_rate, max_generations)
    genetic_algorithm.evolve()
    print(genetic_algorithm.best_solution)
    print(genetic_algorithm.fitness(genetic_algorithm.best_solution))
