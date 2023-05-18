import bpy
import numpy as np
import chess as ch
import pandas as pd

Bpawns = ["B a Pawn", "B b Pawn", "B c Pawn", "B d Pawn", "B e Pawn", "B f Pawn", "B g Pawn", "B h Pawn"]
Bknight = ["B b Knight", "B g Knight"]
Bbishop = ["B DS Bishop", "B WS Bishop"]
Brook = ["B a Rook", "B h Rook"]
Bking = ["B King"]
Bqueen = ["B Queen"]

Wpawns = ["W a Pawn", "W b Pawn", "W c Pawn", "W d Pawn", "W e Pawn", "W f Pawn", "W g Pawn", "W h Pawn"]
Wknight = ["W b Knight", "W g Knight"]
Wbishop = ["W DS Bishop", "W WS Bishop"]
Wrook = ["W a Rook", "W h Rook"]
Wking = ["W King"]
Wqueen = ["W Queen"]

all_pieces = [Bpawns, Bknight, Bbishop, Brook, Bking, Bqueen, Wpawns, Wknight, Wbishop, Wrook, Wking, Wqueen]

arr = np.arange(-3.5, 4.0, 1.0)
idx1, idx2 = np.meshgrid(arr, arr, indexing='ij')
positions = np.stack((idx1, idx2), axis=-1)

integers = np.arange(8)

bpy.ops.wm.open_mainfile(filepath='Blender\chess.blend')

bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = 'PNG'

num_imagens = 10500
df = pd.read_csv('Dataset\JogosPosFen.csv', nrows=num_imagens)

fen_array = df['FEN'].to_numpy()

ftrain = open("Dataset\img_fen.csv", "w")
ftrain.write("IMG,FEN\n")

for i in range(10490,10500):

    Bpawns = ["B a Pawn", "B b Pawn", "B c Pawn", "B d Pawn", "B e Pawn", "B f Pawn", "B g Pawn", "B h Pawn"]
    Bknight = ["B b Knight", "B g Knight"]
    Bbishop = ["B DS Bishop", "B WS Bishop"]
    Brook = ["B a Rook", "B h Rook"]
    Bking = ["B King"]
    Bqueen = ["B Queen", "B Queen 2", "B Queen 3"]

    Wpawns = ["W a Pawn", "W b Pawn", "W c Pawn", "W d Pawn", "W e Pawn", "W f Pawn", "W g Pawn", "W h Pawn"]
    Wknight = ["W b Knight", "W g Knight"]
    Wbishop = ["W DS Bishop", "W WS Bishop"]
    Wrook = ["W a Rook", "W h Rook"]
    Wking = ["W King"]
    Wqueen = ["W Queen","W Queen 2","W Queen 3"]

    all_pieces = [Bpawns, Bknight, Bbishop, Brook, Bking, Bqueen, Wpawns, Wknight, Wbishop, Wrook, Wking, Wqueen]

    tabuleiro = ch.Board()
    fen = fen_array[i]
    tabuleiro.set_fen(fen)
    print(fen)

    peca_info = {}

    for x in range(8):
        for y in range(8):
            casa = ch.square_name(ch.square(x, y))
            peca = tabuleiro.piece_at(ch.parse_square(casa))
            
            if peca is not None:
                peca_nome = peca.symbol()
                peca_cor = peca.color
                peca_x = x
                peca_y = y
                
                peca_info[casa] = {'peca': peca_nome, 'cor': peca_cor, 'x': peca_x, 'y': peca_y}

    for casa, info in peca_info.items():
        if   info['peca'] == 'P':
            pc = all_pieces[6][-1]
            all_pieces[6].pop()

        elif info['peca'] == 'N':
            pc = all_pieces[7][-1]
            all_pieces[7].pop()

        elif info['peca'] == 'B':
            pc = all_pieces[8][-1]
            all_pieces[8].pop()
        
        elif info['peca'] == 'R':
            pc = all_pieces[9][-1]
            all_pieces[9].pop()
        
        elif info['peca'] == 'Q':
            pc = all_pieces[11][-1]
            all_pieces[11].pop()
        
        elif info['peca'] == 'K':
            pc = all_pieces[10][-1]
            all_pieces[10].pop()
        
        elif info['peca'] == 'p':
            pc = all_pieces[0][-1]
            all_pieces[0].pop()
        
        elif info['peca'] == 'n':
            pc = all_pieces[1][-1]
            all_pieces[1].pop()
        
        elif info['peca'] == 'b':
            pc = all_pieces[2][-1]
            all_pieces[2].pop()
        
        elif info['peca'] == 'r':
            pc = all_pieces[3][-1]
            all_pieces[3].pop()
        
        elif info['peca'] == 'q':
            pc = all_pieces[5][-1]
            all_pieces[5].pop()
        
        elif info['peca'] == 'k':
            pc = all_pieces[4][-1]
            all_pieces[4].pop()

        obj = bpy.data.objects[pc]
        pos_coord = positions[info['x'], info['y'],:]
        new_location = (pos_coord[0], pos_coord[1], obj.location.z)

        obj.location = new_location
    
    for rest in all_pieces:
        while rest != []:

            pc = rest[-1]
            rest.pop()

            obj = bpy.data.objects[pc]
            pos_coord = (-10,-10)
            new_location = (pos_coord[0], pos_coord[1], obj.location.z)

            obj.location = new_location


    img_name = 'i' + str(i)
    ftrain.write(f"{img_name},{fen}\n")

    bpy.context.scene.render.filepath = 'E:\Mestrado\imagens\i' + str(i)
    bpy.context.scene.frame_set(i + 1)
    bpy.ops.render.render(write_still=True)


print('Sequência de imagens gerada com sucesso!')





