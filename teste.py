import cv2 as cv
import numpy as np

imagem = cv.imread('i3.png', cv.IMREAD_COLOR)

cv.imshow('Original', imagem)
cv.waitKey(0)
cv.destroyAllWindows()

bordas = cv.Canny(imagem, 100, 150, apertureSize=3)

# cv.imshow('Canny', bordas)
# cv.waitKey(0)
# cv.destroyAllWindows()

linhas = cv.HoughLines(bordas, 1, np.pi/180, 160)

images_linhas = imagem.copy()
for linha in linhas:
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

# cv.imshow('Hough', images_linhas)
# cv.waitKey(0)
# cv.destroyAllWindows()


linhas_agrupadas = []
for linha in linhas:

    rho, theta = linha[0]
    agrupada = False

    for id,linha_agrupada in enumerate(linhas_agrupadas):

        rho_agrupada, theta_agrupada = linha_agrupada[0]

        if abs(rho - rho_agrupada) < 30 and abs(theta - theta_agrupada) < np.pi/18:
            agrupada = True
            break

    if not agrupada:
        linhas_agrupadas.append(linha)

max_rho1 = None
max_rho2 = None
min_rho1 = None
min_rho2 = None

for linha in linhas_agrupadas:
    rho, theta = linha[0]
    if theta > 1:

        if max_rho1 == None:
            max_rho1 = rho
        elif rho > max_rho1:
            max_rho1 = rho
        
        if min_rho1 == None:
            min_rho1 = rho
        elif rho < min_rho1:
            min_rho1 = rho
    else:

        if max_rho2 == None:
            max_rho2 = rho
        elif rho > max_rho2:
            max_rho2 = rho
        
        if min_rho2 == None:
            min_rho2 = rho
        elif rho < min_rho2:
            min_rho2 = rho

linhas_casas = []

for linha in linhas_agrupadas:
    rho,theta = linha[0]

    if not (rho == max_rho2 or rho == max_rho1 or rho == min_rho1 or rho == min_rho2):
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

                if theta < 1:
                    pontos_dessa_linha.append((inter_x, inter_y))
    
    if theta < 1:
        pontos_por_linhas.append(pontos_dessa_linha)


for sublist in pontos_por_linhas:
    sublist.sort(key=lambda x: x[0])
pontos_por_linhas.sort(key=lambda x: x[0][0])

casas = []
for i in range(len(pontos_por_linhas)-1):
    for j in range(len(pontos_por_linhas[i])-1):
        pontos = [pontos_por_linhas[i][j], pontos_por_linhas[i][j+1], pontos_por_linhas[i+1][j], pontos_por_linhas[i+1][j+1]]
        casas.append(pontos)

# for i in casas:
#     imagem_pontos = imagem.copy()
#     for j in i:
        
#         x, y = j
#         cv.circle(imagem_pontos, (x, y), 5, (255, 0, 0), 2)

#     cv.imshow('Intersecções', imagem_pontos)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

for i in range(64):
    ponto1 = casas[i][0] # Topo esquerdo
    ponto2 = casas[i][1] # Topo direito
    ponto3 = casas[i][3] # Base direita
    ponto4 = casas[i][2] # Base esquerda

    pontos = [ponto1, ponto2, ponto3, ponto4]

    coordenadas_x = [p[0] for p in pontos]
    coordenadas_y = [p[1] for p in pontos]

    x_min = min(coordenadas_x)
    x_max = max(coordenadas_x)
    y_min = min(coordenadas_y)
    y_max = max(coordenadas_y)

    imagem_recortada = imagem[y_min-120:y_max, x_min:x_max]

    # Exibir a imagem recortada
    cv.imshow('Imagem Recortada', imagem_recortada)
    cv.waitKey(0)
    cv.destroyAllWindows()

# imagem_pontos = imagem.copy()
# for ponto in pontos:
#     x, y = ponto
#     cv.circle(imagem_pontos, (x, y), 5, (255, 0, 0), 2)

# cv.imshow('Intersecções', imagem_pontos)
# cv.waitKey(0)
# cv.destroyAllWindows()

# for pontos_linha in pontos_por_linhas:
#     imagen_linha = imagem.copy()
#     for ponto in pontos_linha:
#         x, y = ponto
#         cv.circle(imagen_linha, (x, y), 5, (255, 0, 0), 2)

        # cv.imshow('Intersecções', imagen_linha)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


