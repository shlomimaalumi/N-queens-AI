from datetime import time

from MIN_CONFLICTS_algorithm import min_conflicts_algorithm_results, MinConflictsAlgorithm
from Genetic_algorithm import mutation_rate_vs_success_rate_vs_steps, GeneticAlgorithmNQueens

# min_conflicts_algorithm_results(list(range(4, 42, 2))+ [50,60])
alg = MinConflictsAlgorithm(64,10000000,False)
mutations = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for m in mutations:
    print("Mutation rate: ", m)
    print("\n")
    n = 64
    alg = GeneticAlgorithmNQueens(n, max(10, n), m, 1000 * n ** 2)
    start = time()
    alg.solve()
    end = time() - start
    print(end)

    print(alg.generations)

# print(alg.calculate_solution_conflicts())
# print the board:
board = alg.board
for row in board:
    print(row)
print(alg.get_all_conflicts())
# mutation_rate_vs_success_rate_vs_steps(list(range(4, 60, 2)), [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
