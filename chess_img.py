import bpy
import numpy as np
import chess as ch

pieces = ["B a Pawn", "B b Pawn", "B c Pawn", "B d Pawn", "B e Pawn", "B f Pawn", "B g Pawn", "B h Pawn",
          "B b Knight", "B g Knight",
          "B DS Bishop", "B WS Bishop",
          "B a Rook", "B h Rook",
          "B King", "B Queen",
          "W a Pawn", "W b Pawn", "W c Pawn", "W d Pawn", "W e Pawn", "W f Pawn", "W g Pawn", "W h Pawn",
          "W b Knight", "W g Knight",
          "W DS Bishop", "W WS Bishop",
          "W a Rook", "W h Rook",
          "W King", "W Queen"]

arr = np.arange(-3.5, 4.0, 1.0)
idx1, idx2 = np.meshgrid(arr, arr, indexing='ij')
positions = np.stack((idx1, idx2), axis=-1)

squares = np.array([[ch.H8,ch.H7,ch.H6,ch.H5,ch.H4,ch.H3,ch.H2,ch.H1],
                    [ch.G8,ch.G7,ch.G6,ch.G5,ch.G4,ch.G3,ch.G2,ch.G1],
                    [ch.F8,ch.F7,ch.F6,ch.F5,ch.F4,ch.F3,ch.F2,ch.F1],
                    [ch.E8,ch.E7,ch.E6,ch.E5,ch.E4,ch.E3,ch.E2,ch.E1],
                    [ch.D8,ch.D7,ch.D6,ch.D5,ch.D4,ch.D3,ch.D2,ch.D1],
                    [ch.C8,ch.C7,ch.C6,ch.C5,ch.C4,ch.C3,ch.C2,ch.C1],
                    [ch.B8,ch.B7,ch.B6,ch.B5,ch.B4,ch.B3,ch.B2,ch.B1],
                    [ch.A8,ch.A7,ch.A6,ch.A5,ch.A4,ch.A3,ch.A2,ch.A1]])

integers = np.arange(8)

bpy.ops.wm.open_mainfile(filepath='Blender\chess.blend')

bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = 'PNG'

num_imagens = 10000

ftrain = open("Dataset\img_fen.csv", "w")
ftrain.write("IMG,FEN\n")

for i in range(num_imagens):

    tabuleiro = ch.Board()
    tabuleiro.clear()
    combinations = []

    while len(combinations) < 32:
        np.random.shuffle(integers)
        combination = tuple(integers[:2])
        if combination not in combinations:
            combinations.append(combination)
    
    for idx, piece in enumerate(pieces):

        coord = combinations[idx]
        square = squares[coord[0],coord[1]]

        if piece[0] == 'B':
            if "Pawn" in piece:
                aux_piece = ch.Piece(ch.PAWN, ch.BLACK)
            elif "Knight" in piece:
                aux_piece = ch.Piece(ch.KNIGHT, ch.BLACK)
            elif "Bishop" in piece:
                aux_piece = ch.Piece(ch.BISHOP, ch.BLACK)
            elif "Rook" in piece:
                aux_piece = ch.Piece(ch.ROOK, ch.BLACK)
            elif "King" in piece:
                aux_piece = ch.Piece(ch.KING, ch.BLACK)
            elif "Queen" in piece:
                aux_piece = ch.Piece(ch.QUEEN, ch.BLACK)
        
        elif piece[0] == 'W':
            if "Pawn" in piece:
                aux_piece = ch.Piece(ch.PAWN, ch.WHITE)
            elif "Knight" in piece:
                aux_piece = ch.Piece(ch.KNIGHT, ch.WHITE)
            elif "Bishop" in piece:
                aux_piece = ch.Piece(ch.BISHOP, ch.WHITE)
            elif "Rook" in piece:
                aux_piece = ch.Piece(ch.ROOK, ch.WHITE)
            elif "King" in piece:
                aux_piece = ch.Piece(ch.KING, ch.WHITE)
            elif "Queen" in piece:
                aux_piece = ch.Piece(ch.QUEEN, ch.WHITE)
        
        tabuleiro.set_piece_at(square, aux_piece)
        
        obj = bpy.data.objects[piece]
        xy_coord = combinations[idx]
        pos_coord = positions[xy_coord[0],xy_coord[1],:]
        new_location = (pos_coord[0], pos_coord[1], obj.location.z)

        obj.location = new_location

    img_name = 'i' + str(i)
    fen = tabuleiro.fen()
    ftrain.write(f"{img_name},{fen}\n")

    bpy.context.scene.render.filepath = 'E:\Mestrado\imagens/i' + str(i)
    bpy.context.scene.frame_set(i + 1)
    bpy.ops.render.render(write_still=True)


print('Sequência de imagens gerada com sucesso!')





