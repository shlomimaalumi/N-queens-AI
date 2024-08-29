import random


class GeneticAlgorithmNQueens:
    def __init__(self, n, population_size=50, mutation_rate=0.05, max_generations=1000):
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.init_population()
        self.best_solution = None
        self.generations = 0

    def init_population(self):
        return [self.random_board() for _ in range(self.population_size)]

    def random_board(self):
        return [random.randint(0, self.n - 1) for _ in range(self.n)]

    def fitness(self, board):
        max_pairs = self.n * (self.n - 1) // 2
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                    conflicts += 1
        return max_pairs - conflicts

    def select_parents(self):
        tournament_size = 5
        return [max(random.sample(self.population, tournament_size), key=self.fitness) for _ in range(2)]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(0, self.n - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, board):
        if random.random() < self.mutation_rate:
            board[random.randint(0, self.n - 1)] = random.randint(0, self.n - 1)
        return board

    def evolve(self):
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


