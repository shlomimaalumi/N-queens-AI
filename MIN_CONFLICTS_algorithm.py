""""
Min-Conflict
To solve the N-Queens problem, we implemented a hybrid approach combining the Min-Conflicts Heuristic with elements of local search. Our algorithm iteratively adjusts the positions of queens on an N x N chessboard, aiming to minimize conflicts (queen attacks) until a valid solution is found or a preset limit of steps is reached. This method leverages the strength of the Min-Conflicts Heuristic to efficiently navigate large search spaces in a good running time complexity.
Model Description The algorithm follows these steps:
Initialization: We randomly place N queens on the board. This random placement serves as the starting point for the algorithm.
Conflict Resolution Loop: The algorithm identifies the most conflicted queen, i.e., the queen in the position where the most attacks occur. The queen is then removed from its position and relocated to the best position in its column, where it has the fewest conflicts. This process repeats iteratively until either: A solution is found with zero conflicts, or A maximum number of steps is reached. Termination: The algorithm stops if all conflicts are resolved, or the step limit is reached, at which point the solution is either valid or failed.
Assumptions Random Initial Placement: We assume that starting with a randomly generated board configuration gives enough diversity to explore the solution space effectively.
Local Conflict Minimization: We assume that making greedy choices (moving the most conflicted queen to a better position) can lead to a global solution or minimize conflicts.
Step Limit: The algorithm assumes that if a solution is not found within a certain number of steps (exact number will be selected after result research), the current configuration is unlikely to lead to a valid solution without starting over.

Success Criteria The success of the algorithm is determined by the following:
Conflict-Free Solution: The ultimate goal is to position all queens such that no two queens threaten each other, which is a state of zero conflicts.
Efficiency: The algorithm is considered successful if it consistently finds a solution within the preset step limit, minimizing the number of steps taken.
Scalability: The ability to maintain solution quality and computational efficiency as n increases (i.e., for larger board sizes) is a key measure of success.
Key Features of the Algorithm The core of the algorithm is the conflict evaluation function, which computes the total number of attacks for each queen based on its row, column, and diagonals. This function runs in O(n) time for each queen, making it efficient for large boards.
Most Conflicted Queen Selection: At each step, the algorithm identifies the queen with the highest number of conflicts, ensuring that the biggest problem areas are addressed first. This step runs in O(n²) time.
Best Position Search: After removing a conflicted queen, the algorithm searches for the position in the same column that minimizes conflicts. This step runs in O(n³) time for each iteration, as we are checking each cell for its conflict value.
running time complexity: O(t * n^3), where t is the number of steps and n is the size of the board.
Evaluation To evaluate the quality of our solution, we measure the following: The average, median number of steps required to find a valid solution. The percentage of successful runs that result in a valid solution within the step limit.  Run-time efficiency as board size increases. average Number of conflicts in un-success termination.

"""
import csv
import time

from tabulate import tabulate

import math
import random
from collections import defaultdict
from itertools import count

from re import match
from statistics import median

from matplotlib import pyplot as plt
from numpy.ma.extras import average


class MinConflictsAlgorithm:
    """
    Min-Conflicts algorithm for solving the N-Queens problem
    the algorithm is based on the following steps:
    1. randomly initialize the board with n queens
    2. while the termination criteria is not met:
        a. get the most conflicted queen
        b. remove the queen
        c. get the best position for the queen
        d. place the queen in the best position
    3. return the solution

    The running time of the algorithm is O(t * n^3) where t is the number of steps and n is the size of the board.
    """

    def_max_steps = 1000

    def __init__(self, n, max_steps=def_max_steps, limit_steps=True):
        self.limit_steps = limit_steps
        self.max_steps = max_steps
        self.n = n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.steps = 0
        self.best_board = None
        self.best_board_conflicts = math.inf
        self.step_to_reach_best = math.inf

    # region conflicts calc
    def get_conflicts(self, row, col):
        """Calculate the number of conflicts for a given cell at (row, col)."""
        return (self.conflicts_per_row(row, col) +
                self.conflicts_per_col(row, col) +
                self.conflicts_per_main_diag(row, col) +
                self.conflicts_per_secondary_diag(row, col))

    def get_all_conflicts(self):
        """Calculate the total number of conflicts on the board."""
        conflicts = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 1:
                    conflicts += self.get_conflicts(i, j)
        return conflicts // 2  # Each conflict is counted twice, so divide by 2

    def conflicts_per_row(self, row, col):
        """Count conflicts in the same row, excluding the current cell."""
        return sum(1 for i in range(self.n) if i != col and self.board[row][i] == 1)

    def conflicts_per_col(self, row, col):
        """Count conflicts in the same column, excluding the current cell."""
        return sum(1 for i in range(self.n) if i != row and self.board[i][col] == 1)

    def conflicts_per_main_diag(self, row, col):
        """Count conflicts in the main diagonal (top-left to bottom-right)."""
        count = 0
        # Check top-left to bottom-right direction
        for d in range(-self.n + 1, self.n):
            r, c = row + d, col + d
            if 0 <= r < self.n and 0 <= c < self.n and (r, c) != (row, col) and self.board[r][c] == 1:
                count += 1
        return count

    def conflicts_per_secondary_diag(self, row, col):
        """Count conflicts in the secondary diagonal (top-right to bottom-left)."""
        count = 0
        # Check top-right to bottom-left direction
        for d in range(-self.n + 1, self.n):
            r, c = row + d, col - d
            if 0 <= r < self.n and 0 <= c < self.n and (r, c) != (row, col) and self.board[r][c] == 1:
                count += 1
        return count

    # endregion

    def solve(self, max_steps=def_max_steps):
        """running time is O(t * n^3)"""
        self.init_N_queens()
        while not self.termination_criteria():
            i, j = self.get_most_conflicted_queen()
            self.board[i][j] = 0
            i, j = self.get_best_position()
            self.board[i][j] = 1
            self.steps += 1
            conflicts = self.get_all_conflicts()
            if conflicts < self.best_board_conflicts:
                self.best_board = [row[:] for row in self.board]
                self.best_board_conflicts = conflicts
                self.step_to_reach_best = self.steps
        self.board = self.best_board


    def init_N_queens(self):
        """ running time is O(n)"""
        count = 0
        s = set()
        while count < self.n:
            row = random.randint(0, self.n - 1)
            col = random.randint(0, self.n - 1)
            if frozenset([row, col]) not in s:
                s.add(frozenset([row, col]))
                self.board[row][col] = 1
                count += 1

    def termination_criteria(self):
        """running time is O(n^2)"""
        conflicts_sum = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 1:
                    conflicts_sum += self.get_conflicts(i, j)
        return conflicts_sum == 0 or (self.limit_steps and self.steps >= self.max_steps)

    def get_most_conflicted_queen(self):
        """running time is O(n^2)"""
        max_conflicts = 0
        max_conflicts_queen = []
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 0:
                    continue
                conflicts = self.get_conflicts(i, j)
                if conflicts > max_conflicts:
                    max_conflicts = conflicts
                    max_conflicts_queen = [(i, j)]
                elif conflicts == max_conflicts:
                    max_conflicts_queen.append((i, j))
        return random.choice(max_conflicts_queen)

    def get_best_position(self):
        """running time is O(n^3)"""
        min_conflicts = math.inf
        best_position = []
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 0:
                    conflicts = self.get_conflicts(i, j)
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_position = [(i, j)]
                    elif conflicts == min_conflicts:
                        best_position.append((i, j))
        return random.choice(best_position)

    def print_solution(self):
        for i in range(self.n):
            for j in range(self.n):
                print('Q' if self.board[i][j] == 1 else '.', end=' ')
            print()
        print("nuber of steps: ", self.steps)
        print("number of conflicts: ", self.get_all_conflicts())


def min_conflicts_algorithm_results(ns=[4, 6, 8]):
    steps = defaultdict(list)
    step_to_best = defaultdict(list)
    fails = defaultdict(list)
    running_time = defaultdict(list)
    runs_per_n1 = 17

    for n in ns:
        print(n)
        for i in range(runs_per_n1):
            alg = MinConflictsAlgorithm(n, 10 * n)
            start = time.time()
            alg.solve()
            if alg.get_all_conflicts() == 0:
                running_time[n].append(time.time() - start)
                steps[n].append(alg.steps)
            else:
                fails[n].append(alg.get_all_conflicts())
                step_to_best[n].append(alg.step_to_reach_best)

    # Combine success and fail data into a single table
    combined_data = []
    for n in steps.keys():
        success_rate = len(steps[n]) / runs_per_n1 if n in steps else 0
        avg_steps = average(steps[n]) if n in steps else 0
        median_steps = median(steps[n]) if n in steps else 0
        avg_running_time = average(running_time[n]) if n in running_time else 0
        fail_rate = len(fails[n]) / runs_per_n1 if n in fails else 0
        avg_conflicts = average(fails[n]) if n in fails else 0
        median_conflicts = median(fails[n]) if n in fails else 0
        avg_step_to_best = average(step_to_best[n]) if n in step_to_best else 0
        combined_data.append(
            [n, success_rate, avg_steps, median_steps, avg_running_time, fail_rate, avg_conflicts, median_conflicts,
             avg_step_to_best])

    # Print combined table
    print("Combined Results:")
    print(tabulate(combined_data,
                   headers=["N", "Success Rate", "Average Steps", "Median Steps", "Running Time", "Fail Rate",
                            "Average Conflicts", "Median Conflicts", "Average Steps to Best"], tablefmt="grid"))
