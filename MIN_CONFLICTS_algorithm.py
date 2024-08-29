import math
import random
from itertools import count

from re import match


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
        self.boards_history = set()

    # region conflicts calc
    def get_conflicts(self, row, col):
        """running time is O(n)"""
        return (self.conflicts_per_row(row, col) + self.conflicts_per_col(row, col) +
                self.conflicts_per_main_diag(row, col) + self.conflicts_per_secondary_diag(row, col))

    def get_all_conflicts(self):
        """running time is O(n^2)"""
        conflicts = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == 1:
                    conflicts += self.get_conflicts(i, j)
        return conflicts // 2

    def conflicts_per_row(self, row, col):
        count = 0
        for i in range(self.n):
            if i != col and self.board[row][i] == 1:
                count += 1
        return count

    def conflicts_per_col(self, row, col):
        count = 0
        for i in range(self.n):
            if i != row and self.board[i][col] == 1:
                count += 1
        return count

    def conflicts_per_main_diag(self, row, col):
        count = 0
        for i in range(self.n):
            if 0 <= row + i < self.n and 0 <= col + i < self.n and (row + i, col + i) != (row, col) and \
                    self.board[row + i][col + i] == 1:
                count += 1
            if 0 <= row - i < self.n and 0 <= col - i < self.n and (row - i, col - i) != (row, col) and \
                    self.board[row - i][col - i] == 1:
                count += 1
        return count

    def conflicts_per_secondary_diag(self, row, col):
        count = 0
        for i in range(self.n):
            if 0 <= row + i < self.n and 0 <= col - i < self.n and (row + i, col - i) != (row, col) and \
                    self.board[row + i][col - i] == 1:
                count += 1
            if 0 <= row - i < self.n and 0 <= col + i < self.n and (row - i, col + i) != (row, col) and \
                    self.board[row - i][col + i] == 1:
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


        return

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
        if self.boards_history.__contains__(str(self.board)):
            return True
        self.boards_history.add(str(self.board))
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


if __name__ == '__main__':
    count = 0
    n = 8
    for i in range(10):
        min_conflicts = MinConflictsAlgorithm(n)
        min_conflicts.solve()
        min_conflicts.print_solution()
        count += min_conflicts.steps

    print("average number of steps: ", count / 10)
