

import random
import time

class GeneticAlgorithmNQueens:
    def __init__(self, nQSize=8, maxIteration=500000, parentCount=200):
        self.N = nQSize
        self.maxIteration = maxIteration
        self.parents = [[0] * self.N for _ in range(parentCount)]
        self.fitnessArray = [0] * parentCount

    def initialise_parents(self):
        number_set = list(range(self.N))
        for i in range(len(self.parents)):
            random.shuffle(number_set)
            self.parents[i] = number_set[:]

    def calculate_fitness(self, parent):
        fitness = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if parent[i] == parent[j] or abs(parent[i] - parent[j]) == abs(i - j):
                    fitness += 1
        return fitness

    def perform_mutation(self, array):
        random_index1 = random.randint(0, self.N - 1)
        random_index2 = random.randint(0, self.N - 1)
        array[random_index1], array[random_index2] = array[random_index2], array[random_index1]

    def tournament_selection(self):
        first_index = random.randint(0, len(self.parents) - 1)
        second_index = random.randint(0, len(self.parents) - 1)
        return first_index if self.fitnessArray[first_index] < self.fitnessArray[second_index] else second_index

    def parent_selection(self, children, children_fitness):
        for i in range(len(self.parents)):
            if self.fitnessArray[i] < 3 or children_fitness[i] < 3:
                if children_fitness[i] < self.fitnessArray[i]:
                    self.fitnessArray[i] = children_fitness[i]
                    self.parents[i] = children[i][:]
            else:
                if random.randint(0, 1) == 1 and children_fitness[i] < self.fitnessArray[i]:
                    self.fitnessArray[i] = children_fitness[i]
                    self.parents[i] = children[i][:]

    def next_generation(self):
        children = [[0] * self.N for _ in range(len(self.parents))]
        children_fitness = [0] * len(self.parents)

        for i in range(len(self.parents)):
            first_parent_index = self.tournament_selection()
            children[i] = self.parents[first_parent_index][:]
            self.perform_mutation(children[i])
            children_fitness[i] = self.calculate_fitness(children[i])

        self.parent_selection(children, children_fitness)

    def check_for_solution(self):
        return 0 in self.fitnessArray

    def solve(self):
        for iteration in range(self.maxIteration):
            self.next_generation()
            if self.check_for_solution():
                return self.parents[self.fitnessArray.index(0)]

