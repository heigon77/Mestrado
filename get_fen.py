import chess
import numpy as np
import pandas as pd

def get_fen(bitboard):
    board = chess.Board()
    board.clear()

    piece_idx = {0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING}

    for i in range(64):
        for j in range(12):
            if bitboard[j + 12 * i] == 1:
                color = chess.BLACK if j < 6 else chess.WHITE
                piece_type = piece_idx[j % 6]
                square = chess.SQUARES[i]
                board.set_piece_at(square, chess.Piece(piece_type, color))

    board.turn = bool(bitboard[-1])
    castling_fen = ""
    if bitboard[-2]:
        castling_fen += "K"
    if bitboard[-3]:
        castling_fen += "Q"
    if bitboard[-4]:
        castling_fen += "k"
    if bitboard[-5]:
        castling_fen += "q"
    board.set_castling_fen(castling_fen)

    fen = board.fen()
    return fen

def get_bitboard(board):

    bitboard = np.zeros(773, dtype=int)

    piece_idx = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

    for i in range(64):
        if board.piece_at(i):
            color = int(board.piece_at(i).color)
            piece = 6*color + piece_idx[board.piece_at(i).piece_type]
            bitboard[piece + 12*i] = 1

    bitboard[-1] = int(board.turn)
    bitboard[-2] = int(board.has_kingside_castling_rights(chess.WHITE))
    bitboard[-3] = int(board.has_kingside_castling_rights(chess.BLACK))
    bitboard[-4] = int(board.has_queenside_castling_rights(chess.WHITE))
    bitboard[-5] = int(board.has_queenside_castling_rights(chess.BLACK))

    return bitboard


df = pd.read_csv('Dataset\data_bits_normal.csv')

aux = df['Position'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))

df['FEN'] = aux.apply(get_fen)

df = df[['Position', 'FEN', 'Result', 'Rating', 'Capture']]

df.to_csv('novo_arquivo.csv', index=False)