import Board


class Game:
    """represent the game by the n-queens problem"""
    @staticmethod
    def conflicts_count_per_index(board, x, y):
        """count the conflicts for a queen at position (x, y)"""
        count = 0
        for i in range(board.size):
            if i != x and board.get(i, y) == 1:
                count += 1
            if i != y and board.get(x, i) == 1:
                count += 1
            if 0 <= x + i < board.size and 0 <= y + i < board.size and (x + i, y + i) != (x, y) and board.get(x + i, y + i) == 1:
                count += 1
            if 0 <= x - i < board.size and 0 <= y - i < board.size and (x - i, y - i) != (x, y) and board.get(x - i, y - i) == 1:
                count += 1
            if 0 <= x + i < board.size and 0 <= y - i < board.size and (x + i, y - i) != (x, y) and board.get(x + i, y - i) == 1:
                count += 1
            if 0 <= x - i < board.size and 0 <= y + i < board.size and (x - i, y + i) != (x, y) and board.get(x - i, y + i) == 1:
                count += 1
        return count

    @staticmethod
    def conflicts_count(board):
        """count the conflicts for all queens"""
        count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.get(i, j) == 1:
                    count += Game.conflicts_count_per_index(board, i, j)
        return count