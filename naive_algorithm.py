class NaiveAlgorithm:
    """N-Queens problem solver using a naive backtracking algorithm."""

    def __init__(self, n):
        self.n = n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.solution_found = False

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

    def is_safe(self, row, col):
        """Check if it's safe to place a queen at board[row][col]."""
        # Check this row on left side
        for i in range(col):
            if self.board[row][i] == 1:
                return False

        # Check upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if self.board[i][j] == 1:
                return False

        # Check lower diagonal on left side
        for i, j in zip(range(row, self.n, 1), range(col, -1, -1)):
            if self.board[i][j] == 1:
                return False

        return True

    def solve_util(self, col):
        """Utilizes backtracking to solve the N-Queens problem."""
        if col >= self.n:
            self.solution_found = True
            return True

        for i in range(self.n):
            if self.is_safe(i, col):
                self.board[i][col] = 1
                if self.solve_util(col + 1):
                    return True
                self.board[i][col] = 0  # Backtrack

        return False

    def solve(self):
        """Solve the N-Queens problem and return the solution."""
        if self.solve_util(0):
            return self.board
        else:
            return None

    def print_solution(self):
        if self.solution_found:
            for row in self.board:
                print(' '.join('Q' if cell else '.' for cell in row))
        else:
            print("No solution found.")


if __name__ == '__main__':
    n = 8  # Size of the N-Queens problem
    naive_solver = NaiveAlgorithm(n)
    solution = naive_solver.solve()
    naive_solver.print_solution()
