import pandas as pd
import numpy as np
import math

df = pd.read_csv("ChessMovesTable.csv")

def w(piece):
    weights = {
        "K": 12,
        "Q": 9,
        "R": 5,
        "B": 3,
        "N": 2,
        "P": 1,
        "k": 12,
        "q": 9,
        "r": 5,
        "b": 3,
        "n": 2,
        "p": 1,
        "NULL": 0,
        "NaN": 0,
        None: 0
    }
    return weights.get(piece, 0)

def add_null_moves(df_game, null_moves_needed):
    num_moves = len(df_game)

    last_move_number = num_moves
    starting_move_number = last_move_number
    null_moves = []

    for i in range(null_moves_needed):
        move_number = starting_move_number + i
        piece_before_move = "NULL"
        piece_after_move = "NULL"
        piece_captured = "NULL"
        moved_from = "e1" if (last_move_number + i) % 2 == 0 else "e8"
        moved_to = "e1" if (last_move_number + i) % 2 != 0 else "e8"
        captured_at = "e1" if (last_move_number + i) % 2 == 0 else "e8"
        null_moves.append({"Game Number": df_game.iloc[0]["Game Number"],
                            "Move Number": move_number,
                            "Piece Before Move": piece_before_move,
                            "Piece After Move": piece_after_move,
                            "Piece Captured": piece_captured,
                            "Moved From": moved_from,
                            "Moved To": moved_to,
                            "Captured At": captured_at})
    df_nullmoves = pd.DataFrame.from_dict(null_moves, orient='columns')
    df_game = pd.concat([df_game, df_nullmoves], ignore_index=True)

    return df_game

def tranform_to_vector(df_game):
    vec_moves = []

    for index, item in df_game.iterrows():

        piece_before_move_coord = np.array([ord(item["Moved From"][0]) - ord('a'), int(item["Moved From"][1]) - 1])
        piece_after_move_coord = np.array([ord(item["Moved To"][0]) - ord('a'), int(item["Moved To"][1]) - 1])
        piece_captured_coord = np.array([ord(item["Captured At"][0]) - ord('a'), int(item["Captured At"][1]) - 1])
        
        weight_piece_before_move = w(item["Piece Before Move"])
        weight_piece_after_move = w(item["Piece After Move"])
        weight_piece_captured = w(item["Piece Captured"])

        move_vector = np.array([
            piece_before_move_coord[0], piece_before_move_coord[1],
            piece_after_move_coord[0], piece_after_move_coord[1],
            piece_captured_coord[0], piece_captured_coord[1],
            weight_piece_before_move, weight_piece_after_move, weight_piece_captured
        ])
        
        vec_moves.append(move_vector)

    return np.array(vec_moves)

def calculate_distance(df1, df2):
    
    if len(df1) > len(df2):
        df2 = add_null_moves(df2, len(df1) - len(df2))

    elif len(df1) < len(df2):
        df1 = add_null_moves(df1, len(df2) - len(df1))


    vec_g1 = tranform_to_vector(df1)
    vec_g2 = tranform_to_vector(df2)

    distances = []
    for i in range(len(vec_g1)):
        distance = np.linalg.norm(vec_g2[i] - vec_g1[i])
        distances.append(distance)

    return np.sum(distances)

def create_distance_matrix(df):
    num_games = 10000
    distance_matrix = np.zeros((num_games, num_games))

    for i in range(7001, 8000 + 1):
        df1 = df[df["Game Number"] == i]
        if len(df1) == 0:
            continue
        for j in range(i + 1, num_games + 1): 
            if j % 100 == 0:
                print(f"{i}, {i/num_games * 100:.2f}% | {j}, {j/num_games * 100:.2f}%")

            df2 = df[df["Game Number"] == j]
            if len(df2) == 0:
                continue
            distance = calculate_distance(df1, df2)
            distance_matrix[i - 1][j - 1] = distance

    return distance_matrix


df = pd.read_csv("ChessMovesTable.csv")

distance_matrix = create_distance_matrix(df)

print("Finished")

np.save("distance_matrix_8k.npy", distance_matrix)


