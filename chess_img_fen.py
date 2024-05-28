import bpy
import numpy as np
import chess as ch
import pandas as pd

Bpawns = ["B_a_Pawn","B_b_Pawn","B_c_Pawn","B_d_Pawn","B_e_Pawn","B_f_Pawn","B_g_Pawn","B_h_Pawn"]
Bknight = ["B_b_Knight","B_g_Knight"]
Bbishop = ["B_DS_Bishop","B_WS_Bishop"]
Brook = ["B_a_Rook","B_h_Rook"]
Bking = ["B_King"]
Bqueen = ["B_Queen"]

Wpawns = ["W_a_Pawn","W_b_Pawn","W_c_Pawn","W_d_Pawn","W_e_Pawn","W_f_Pawn","W_g_Pawn","W_h_Pawn"]
Wknight = ["W_b_Knight","W_g_Knight"]
Wbishop = ["W_DS_Bishop","W_WS_Bishop"]
Wrook = ["W_a_Rook","W_h_Rook"]
Wking = ["W_King"]
Wqueen = ["W_Queen"]

all_pieces = [Bpawns, Bknight, Bbishop, Brook, Bking, Bqueen, Wpawns, Wknight, Wbishop, Wrook, Wking, Wqueen]

arr = np.arange(-3.5, 4.0, 1.0)
idx1, idx2 = np.meshgrid(arr, arr, indexing='ij')
positions = np.stack((idx1, idx2), axis=-1)

integers = np.arange(8)

bpy.ops.wm.open_mainfile(filepath='Blender\\testeReal.blend')

bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.image_settings.file_format = 'PNG'

num_imagens = 10200
df = pd.read_csv('Dataset\JogoPosFenNextClustered.csv', nrows=num_imagens)

fen_array = df['FEN'].to_numpy()

ftrain = open("Dataset\img_fen.csv", "w")
ftrain.write("IMG,FEN\n")

for i in range(10200):

    try:

        Bpawns = ["B_a_Pawn","B_b_Pawn","B_c_Pawn","B_d_Pawn","B_e_Pawn","B_f_Pawn","B_g_Pawn","B_h_Pawn"]
        Bknight = ["B_b_Knight","B_g_Knight"]
        Bbishop = ["B_DS_Bishop","B_WS_Bishop"]
        Brook = ["B_a_Rook","B_h_Rook"]
        Bking = ["B_King"]
        Bqueen = ["B_Queen_1","B_Queen_2","B_Queen"]

        Wpawns = ["W_a_Pawn","W_b_Pawn","W_c_Pawn","W_d_Pawn","W_e_Pawn","W_f_Pawn","W_g_Pawn","W_h_Pawn"]
        Wknight = ["W_b_Knight","W_g_Knight"]
        Wbishop = ["W_DS_Bishop","W_WS_Bishop"]
        Wrook = ["W_a_Rook","W_h_Rook"]
        Wking = ["W_King"]
        Wqueen = ["W_Queen_2","W_Queen_3","W_Queen"]

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
    
    except:
        print("An exception occurred", i)



print('Sequência de imagens gerada com sucesso!')





