import numpy as np
import chess
from chess import Board

chess_pieces = dict(zip(".prnbqkPRNBQK", range(13)))
weights = {
    "P": 1, "N": 3, "B": 3,
    "R": 5, "Q": 9, "K": 0
}


def squares(fen: str):
    """Encode game state as a vector of squares,
    where each square is associated with the piece
    that it is occupied by or with zero otherwise."""
    board = str(Board(fen))
    return np.array([chess_pieces[p] for p in board if p not in '\n '])


def one_hot_enc_piece(piece: str):
    """One-hot encode chess piece as 12-dimensional (6 types of pieces,
    2 colors each) vector v, where \n
    v_{0}  indicates black pawn, \n
    v_{1}  indicates black rook, \n
    v_{2}  indicates black knight, \n
    v_{3}  indicates black bishop, \n
    v_{4}  indicates black queen, \n
    v_{5}  indicates black king, \n
    v_{6}  indicates white pawn, \n
    v_{7}  indicates white rook, \n
    v_{8}  indicates white knight, \n
    v_{9}  indicates white bishop, \n
    v_{10} indicates white queen, \n
    v_{11} indicates white king.
    """
    if piece == '.':
        return np.zeros(len(chess_pieces) - 1)
    index = chess_pieces[piece] - 1
    zeros_before = [0] * index
    # Why -2:
    # The 1st -1 becasue we have to count in dot at the beginning of `chess_pieces`.
    # The 2nd -1 to get index of last piece.
    zeros_after = [0] * (len(chess_pieces) - 2 - index)
    return np.array(zeros_before + [1] + zeros_after)


def binary(fen: str):
    """Encode game state as a 773-dimensional vector v, where for i in
    {0, 12, ..., 756}
    v_{i}   indicates that there's black pawn on square floor(i/12), \n
    v_{i+1} indicates that there's black rook on square floor(i/12), \n
    v_{i+2} indicates that there's black knight on square floor(i/12), \n
    v_{i+3} indicates that there's black bishop on square floor(i/12), \n
    v_{i+4} indicates that there's black queen on square floor(i/12), \n
    v_{i+5} indicates that there's black king on square floor(i/12), \n
    v_{i+6}  indicates that there's white pawn on square floor(i/12), \n
    v_{i+7}  indicates that there's white rook on square floor(i/12), \n
    v_{i+8}  indicates that there's white knight on square floor(i/12), \n
    v_{i+9}  indicates that there's white bishop on square floor(i/12), \n
    v_{i+10} indicates that there's white queen on square floor(i/12), \n
    v_{i+11} indicates that there's white king on square floor(i/12). \n

    Squares are ordered in row-major way from top left (A8) to bottom right (H1). \n

    v_768 indicates player to move, \n
    v_769 indicates whether black has kingside castling right, \n
    v_770 indicates whether black has queenside castling right, \n
    v_771 indicates whether white has kingside castling right, \n
    v_772 indicates whether white has queenside castling right. \n

    \"indicates\" means \"is set to 0 or 1 depending on whether condition is
    false or true\"."""
    board = Board(fen)
    encoding = np.array([])
    for p in str(board):
        if p in '\n ':
            continue
        encoding = np.hstack((encoding, one_hot_enc_piece(p)))
    encoding = np.hstack((encoding, int(board.turn)))
    castling_rights = []
    for castling_rook_pos in [chess.BB_H8, chess.BB_A8, chess.BB_H1, chess.BB_A1]:
        castling_rights.append(
            int(bool(board.castling_rights & castling_rook_pos)))
    encoding = np.hstack((encoding, np.array(castling_rights)))
    return encoding


def advantage(fen: str):
    """Calculate player advantage based only on moves and pieces they possess.

        - move advantage is the difference between the number of moves
        available to each player
        - material advantage is the difference of sums of weights of all the pieces
        on the board

    Values are positive if White has the advantage and negative otherwise.
    """
    board = Board(fen)
    material_advantage = sum(
        weights[p] if p in weights else -weights[p.upper()]
        for p in str(board) if p not in '\n .'
    )
    board.turn = 1
    move_advantage = len(list(board.legal_moves))
    board.turn = 0
    move_advantage -= len(list(board.legal_moves))
    return np.array([move_advantage, material_advantage])


def binary_and_advantage(fen: str):
    """Combined binary and advantage encoding.
    See their respective descriptions for details."""
    return np.append(binary(fen), advantage(fen))
