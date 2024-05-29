import cv2 as cv
import numpy as np
import pandas as pd
import chess as ch
import sys

df = pd.read_csv('Dataset\img_fen.csv')
file = open("Dataset\img_piece_square.csv", "a")
file.write("Image,Piece,Square\n")

total_rows = len(df)

for index, row in df.iterrows():
    if index == 129:
        break
    if index % 10 == 0:
        percentage = (index / 129) * 100
        print(f"{percentage:.2f}% complete")

    img_name = row['IMG']
    fen = row['FEN']

    fen = fen.split()[0]

    board = ch.Board(fen)

    imagem = cv.imread(f"ResizedImages/{img_name}.png", cv.IMREAD_COLOR)

    # cv.imshow('Original', imagem)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    center_x = imagem.shape[1] // 2
    center_y = imagem.shape[0] // 2

    start_x = center_x - 460
    end_x = center_x + 510
    start_y = center_y - 400
    end_y = center_y + 480

    imagem = imagem[start_y:end_y, start_x:end_x]

    # cv.imshow('Mask', imagem)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    blurred = cv.GaussianBlur(imagem, (5, 5), 0)

    # cv.imshow('GaussianBlur', blurred)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    bordas = cv.Canny(blurred, 100, 150, apertureSize=3)

    # cv.imshow('Canny', bordas)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    linhas = cv.HoughLines(bordas, 1, np.pi/180, 150)

    images_linhas = imagem.copy()
    linhas_vr = []
    if linhas is not None:
        for linha in linhas:
            rho, theta = linha[0]
            
            # Convert theta from radians to degrees for easier interpretation
            angle = theta * 180 / np.pi
            
            # Define the range for nearly vertical lines (close to 90 degrees)
            if 50 <= angle <= 130 or 230 <= angle <= 310:
                linhas_vr.append(linha)
                # Draw the line (nearly vertical)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(images_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Define the range for nearly horizontal lines (close to 0 or 180 degrees)
            elif -40 <= angle <= 40 or 140<= angle <= 210:
                linhas_vr.append(linha)
                # Draw the line (nearly horizontal)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv.line(images_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cv.imshow('Hough', images_linhas)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    linhas_agrupadas = []
    
    for linha in linhas_vr:

        rho, theta = linha[0]
        # print(rho, theta)
        agrupada = False

        for id,linha_agrupada in enumerate(linhas_agrupadas):

            rho_agrupada, theta_agrupada = linha_agrupada[0]

            if abs(rho - rho_agrupada) < 30 and abs(theta - theta_agrupada) < np.pi/18:
                agrupada = True
                break

        if not agrupada:
            linhas_agrupadas.append(linha)

 


    linhas_casas = []

    for linha in linhas_agrupadas:
        rho,theta = linha[0]

        linhas_casas.append(linha)

    images_linhas = imagem.copy()
    linhas_pontos = []
    for linha in linhas_casas:
        rho, theta = linha[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        cv.line(images_linhas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        linhas_pontos.append([(x1, y1), (x2, y2), theta])

    # cv.imshow('Hough', images_linhas)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    pontos = []
    pontos_por_linhas = []
    for i in range(len(linhas_pontos)):
        pontos_dessa_linha = []

        for j in range(len(linhas_pontos)):

            x1, y1 = linhas_pontos[i][0]
            x2, y2 = linhas_pontos[i][1]
            theta = linhas_pontos[i][2]

            x3, y3 = linhas_pontos[j][0]
            x4, y4 = linhas_pontos[j][1]

            det = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

            if det != 0:

                inter_x = int(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / det)
                inter_y = int(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / det)

                if (inter_x >= 0 and inter_x <= 1920) and (inter_y >= 0 and inter_y <= 1080): 

                    if (inter_x, inter_y) not in pontos:
                        pontos.append((inter_x, inter_y))

                    if theta > 1 and theta < 3:
                        pontos_dessa_linha.append((inter_x, inter_y))
        
        if theta > 1 and theta < 3:
            pontos_por_linhas.append(pontos_dessa_linha)


    for sublist in pontos_por_linhas:
        sublist.sort(key=lambda x: x[0])
    pontos_por_linhas.sort(key=lambda x: x[0][0])

    # pontos_por_linhas.reverse()

    try:
        casas = []
        for i in range(len(pontos_por_linhas)-1):
            for j in range(len(pontos_por_linhas[i])-1):
                pontos = [pontos_por_linhas[i][j], pontos_por_linhas[i][j+1], pontos_por_linhas[i+1][j], pontos_por_linhas[i+1][j+1]]
                casas.append(pontos)

        imagem_pontos = imagem.copy()
        for i in casas:
            
            for j in i:
                
                x, y = j
                cv.circle(imagem_pontos, (x, y), 5, (255, 0, 0), 2)

        # cv.imshow('Intersecções', imagem_pontos)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        for i in range(64):
            ponto1 = casas[i][0]
            ponto2 = casas[i][1]
            ponto3 = casas[i][3]
            ponto4 = casas[i][2]

            pontos = [ponto1, ponto2, ponto3, ponto4]

            coordenadas_x = [p[0] for p in pontos]
            coordenadas_y = [p[1] for p in pontos]

            x_min = min(coordenadas_x)
            x_max = max(coordenadas_x)
            y_min = min(coordenadas_y)
            y_max = max(coordenadas_y)

            if y_min - 80 < 0:
                y_min = 80

            imagem_recortada = imagem[y_min-80:y_max, x_min:x_max]

            piece = board.piece_at(i)

            if piece is not None:
                symbol = piece.symbol()
                file.write(f"{img_name}_{i},{symbol},{i}\n")
                cv.imwrite(f"RecReais/{img_name}_{i}.png", imagem_recortada)
            else:
                symbol = '0'
                file.write(f"{img_name}_{i},{symbol},{i}\n")
                cv.imwrite(f"RecReais/{img_name}_{i}.png", imagem_recortada)

            # cv.imshow('Img', imagem_recortada)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
    
    except Exception as e:

        ferror = open("erros.txt", "a")
        ferror.write(f"{img_name} An unexpected error occurred: {e}\n")
        ferror.close()

print(f"100.00% complete")