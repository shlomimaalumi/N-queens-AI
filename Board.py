# class Board:
#     def __init__(self, size):
#         self.size = size
#         self.board = [[0 for _ in range(size)] for _ in range(size)]
#
#     def put(self, x, y):
#         self.board[x][y] = 1
#
#     def get(self, x, y):
#         return self.board[x][y]
#
#     def remove(self, x, y):
#         self.board[x][y] = 0
#
#     def __str__(self):
#         return '\n'.join([' '.join([str(cell) for cell in row]) for row in self.board])
class Piece:
    def __init__(self, is_queen=False):
        self.is_queen = is_queen

    def __repr__(self):
        return 'Q' if self.is_queen else '.'

    def place_queen(self):
        self.is_queen = True

    def remove_queen(self):
        self.is_queen = False

    def is_empty(self):
        return not self.is_queen


class Board:
    def __init__(self, size):
        self.size = size
        self.board = [[Piece() for _ in range(size)] for _ in range(size)]

    def __repr__(self):
        return '\n'.join(' '.join(str(piece) for piece in row) for row in self.board)

    def place_queen(self, row, col):
        if not self.is_under_attack(row, col):
            self.board[row][col].place_queen()
            return True
        return False

    def remove_queen(self, row, col):
        self.board[row][col].remove_queen()

    def is_under_attack(self, row, col):
        # Check row and column
        for i in range(self.size):
            if self.board[row][i].is_queen or self.board[i][col].is_queen:
                return True

        # Check diagonals
        for i in range(self.size):
            for j in range(self.size):
                if abs(row - i) == abs(col - j) and self.board[i][j].is_queen:
                    return True

        return False

    def solve(self, col=0):
        if col >= self.size:
            return True

        for row in range(self.size):
            if self.place_queen(row, col):
                if self.solve(col + 1):
                    return True
                self.remove_queen(row, col)

        return False

    def print_solution(self):
        print(self)

if __name__ == '__main__':
    board = Board(8)
    board.solve()

