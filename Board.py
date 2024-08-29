class Board:
    def __init__(self, size):
        self.size = size
        self.board = [[0 for _ in range(size)] for _ in range(size)]

    def put(self, x, y):
        self.board[x][y] = 1

    def get(self, x, y):
        return self.board[x][y]

    def remove(self, x, y):
        self.board[x][y] = 0

    def __str__(self):
        return '\n'.join([' '.join([str(cell) for cell in row]) for row in self.board])
