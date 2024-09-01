from itertools import permutations

class NaiveAlgorithm:
    def __init__(self, n):
        self.n = n

    def solve(self):
        """Solve the N-Queens problem using a naive brute-force approach."""
        # Generate all permutations of column indices (0 to n-1) for n queens
        for perm in permutations(range(self.n)):
            if self.is_valid_solution(perm):
                return perm
        return None

    def is_valid_solution(self, queens):
        """Check if a given permutation is a valid solution for N-Queens."""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Check if two queens are in the same diagonal
                if abs(queens[i] - queens[j]) == abs(i - j):
                    return False
        return True
