import cv2
import numpy as np

imagem = cv2.imread('i3.png', cv2.IMREAD_COLOR)

bordas = cv2.Canny(imagem, 100, 150, apertureSize=3)

linhas = cv2.HoughLines(bordas, 1, np.pi/180, 155)

linhas_agrupadas = []
for linha in linhas:

    rho, theta = linha[0]
    agrupada = False

    for linha_agrupada in linhas_agrupadas:
        rho_agrupada, theta_agrupada = linha_agrupada[0]
        
        if abs(rho - rho_agrupada) < 30 and abs(theta - theta_agrupada) < np.pi/18:
            agrupada = True
            break

    if not agrupada:
        linhas_agrupadas.append(linha)

array_d = np.array(linhas_agrupadas)

arr_d1 = array_d[array_d[:,:,1] > 1.2]
arr_d2 = array_d[array_d[:,:,1] <= 1.2]

arr_d1 = arr_d1[arr_d1[:,0].argsort()]
arr_d2 = arr_d2[arr_d2[:,0].argsort()]

arr_d1 = arr_d1[1:-1,:]
arr_d2 = arr_d2[1:-1,:]

images_linhasd1 = imagem.copy()
for linha in arr_d1:
    rho = linha[0]
    theta = linha[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 500 * (-b))
    y1 = int(y0 + 500 * (a))
    x2 = int(x0 - 500 * (-b))
    y2 = int(y0 - 500 * (a))
    cv2.line(imagem, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Imagem com linhas detectadas', imagem)
cv2.waitKey(0)

images_linhasd2 = imagem.copy()
for linha in arr_d2:
    rho = linha[0]
    theta = linha[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 500 * (-b))
    y1 = int(y0 + 500 * (a))
    x2 = int(x0 - 500 * (-b))
    y2 = int(y0 - 500 * (a))
    cv2.line(imagem, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Imagem com linhas detectadas', imagem)
cv2.waitKey(0)

import numpy as np

def calcular_ponto_intersecao(rho1, theta1, rho2, theta2):
    # Converter coordenadas polares para coordenadas cartesianas
    x1 = rho1 * np.cos(theta1)
    y1 = rho1 * np.sin(theta1)
    x2 = rho2 * np.cos(theta2)
    y2 = rho2 * np.sin(theta2)

    # Escrever as equações paramétricas das retas em coordenadas cartesianas
    a1 = x1 - x2
    b1 = -x1
    c1 = y1 - y2
    d1 = -y1
    a2 = x2 - x1
    b2 = -x2
    c2 = y2 - y1
    d2 = -y2

    # Calcular o valor de t onde as retas se intersectam
    t = (d1 - d2) / (c1 - c2)

    # Calcular os valores correspondentes de x e y no ponto de interseção
    x_intersecao = a1 * t + b1
    y_intersecao = c1 * t + d1

    return x_intersecao, y_intersecao



def separar_quadrados(arr_d1, arr_d2, img_original):
    # Separa as coordenadas das retas em pares (rho, theta)
    coordenadas = []
    for linha1 in range(len(arr_d1) - 1):
        for linha2 in range(len(arr_d2) - 1):

            coordenadas.append((linha1[0], linha1[1], linha2[0], linha2[1]))

    # Cria uma imagem em branco para desenhar os quadrados
    img_quadrados = np.zeros_like(img_original)

    # Loop para desenhar os quadrados
    for coord in coordenadas:
        rho1, theta1, rho2, theta2 = coord
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        x01, y01 = a1 * rho1, b1 * rho1
        x02, y02 = a2 * rho2, b2 * rho2
        x11, y11 = int(x01 * (-b1)), int(y01 * (a1))
        x12, y12 = int(x01 * (-b1)), int(y01 * (a1))
        x21, y21 = int(x02 * (-b2)), int(y02 * (a2))
        x22, y22 = int(x02 * (-b2)), int(y02 * (a2))

        print(x11, y11)
        print(x22, y22)
        # Desenha os quadrados na imagem em branco
        cv2.rectangle(img_quadrados, (x11, y11), (x22, y22), (0, 255, 0), 2)
    
    cv2.imshow('Imagem com linhas detectadas', img_quadrados)
    cv2.waitKey(0)

    # Recorta os quadrados da imagem original
    quadrados = []
    for coord in coordenadas:
        rho1, theta1, rho2, theta2 = coord
        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        x01, y01 = a1 * rho1, b1 * rho1
        x02, y02 = a2 * rho2, b2 * rho2
        x11, y11 = int(x01 * (-b1)), int(y01 * (a1))
        x22, y22 = int(x02 * (-b2)), int(y02 * (a2))
        quadrado = img_original[y11:y22, x11:x22]
        quadrados.append(quadrado)

    return quadrados

quadrados = separar_quadrados(arr_d1, arr_d2, imagem)

print(quadrados[0].shape)

cv2.imshow('Imagem com linhas detectadas', quadrados[0])
cv2.waitKey(0)
