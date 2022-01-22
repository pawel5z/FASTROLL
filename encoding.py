import numpy as np
from chess import Board

chess_pieces = dict(zip(".prnbqkPRNBQK", range(13)))


def squares(fen: str):
    """Encode game state as a vector of squares,
    where each square is associated with the piece
    that it is occupied by or with zero otherwise."""
    board = str(Board(fen))
    return np.array([chess_pieces[p] for p in board if p not in '\n '])


def pieces(fen: str):
    """Encode game state as a vector of pieces,
    where each piece is associated with the square
    it occupies or with zero otherwise."""
    raise NotImplemented


