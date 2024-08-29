import random


class GeneticAlgorithmNQueens:
    """
    Genetic Algorithm for solving the N-Queens problem.

    Steps:
    1. Initialize a random population of solutions.
    2. Evaluate fitness of each individual.
    3. Select parents based on fitness.
    4. Perform crossover and mutation to produce offspring.
    5. Repeat the process until a solution is found or termination criteria are met.

    The running time is O(t * n^2) where t is the number of generations and n is the size of the board.
    """

    def __init__(self, n, population_size=100, mutation_rate=0.05, max_generations=1000):
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.init_population()
        self.best_solution = None

    def init_population(self):
        """Initialize a random population of individuals."""
        return [self.random_board() for _ in range(self.population_size)]

    def random_board(self):
        """Generate a random board configuration."""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]

    def fitness(self, board):
        """Calculate fitness based on the number of non-conflicting pairs."""
        max_pairs = self.n * (self.n - 1) // 2
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                    conflicts += 1
        return max_pairs - conflicts

    def select_parents(self):
        """Select two parents using tournament selection."""
        tournament_size = 5
        return [max(random.sample(self.population, tournament_size), key=self.fitness) for _ in range(2)]

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create an offspring."""
        crossover_point = random.randint(0, self.n - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, board):
        """Mutate the board by randomly changing the position of one queen."""
        if random.random() < self.mutation_rate:
            board[random.randint(0, self.n - 1)] = random.randint(0, self.n - 1)
        return board

    def evolve(self):
        """Run the Genetic Algorithm to evolve the population."""
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
                break

        if not self.best_solution:
            self.best_solution = max(self.population, key=self.fitness)

    def print_solution(self):
        """Print the best solution found."""
        if self.best_solution:
            board = self.best_solution
            for i in range(self.n):
                row = ['Q' if board[j] == i else '.' for j in range(self.n)]
                print(' '.join(row))
            print("Number of conflicts:", self.n * (self.n - 1) // 2 - self.fitness(self.best_solution))
        else:
            print("No solution found.")


if __name__ == '__main__':
    n = 8
    total_steps = 0
    for _ in range(10):
        genetic_algorithm = GeneticAlgorithmNQueens(n)
        genetic_algorithm.evolve()
        genetic_algorithm.print_solution()
        print()

    print(f"Average number of generations: {total_steps / 10}")
