import cv2
import numpy as np

imagem = cv2.imread('imagens\i8457.png', cv2.IMREAD_COLOR)

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
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))
    cv2.line(images_linhasd1, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Imagem com linhas detectadas', images_linhasd1)
cv2.waitKey(0)

images_linhasd2 = imagem.copy()
for linha in arr_d2:
    rho = linha[0]
    theta = linha[1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 10000 * (-b))
    y1 = int(y0 + 10000 * (a))
    x2 = int(x0 - 10000 * (-b))
    y2 = int(y0 - 10000 * (a))
    cv2.line(images_linhasd2, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow('Imagem com linhas detectadas', images_linhasd2)
cv2.waitKey(0)



def calcular_ponto_intersecao(l1, l2):
    # Extrai os valores de rho e theta das equações paramétricas das retas em l1 e l2
    rho1 = l1[0]
    theta1 = l1[1]

    rho2 = l1[0]
    theta2 = l2[1]
    
    # Calcula o seno e cosseno dos ângulos theta1 e theta2
    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)
    cos_theta2 = np.cos(theta2)
    sin_theta2 = np.sin(theta2)
    
    # Calcula os coeficientes a e b do sistema de equações lineares
    a = np.array([[cos_theta1, sin_theta1],
                  [cos_theta2, sin_theta2]])
    b = np.array([rho1, rho2])
    
    # Resolve o sistema de equações lineares para obter os valores de x e y
    x, y = np.linalg.solve(a, b)
    
    # Retorna o ponto de interseção como uma tupla (x, y)
    return x, y


print(calcular_ponto_intersecao(arr_d1[0], arr_d2[0]))
# # Criação das imagens separadas para cada casa
# for i in range(len(linhas_casas)):
#     for j in range(i+1, len(linhas_casas)):
#         ponto_intersecao = obter_ponto_intersecao(linhas_casas[i], linhas_casas[j])
#         x, y = ponto_intersecao
#         if 0 <= x < imagem.shape[1] and 0 <= y < imagem.shape[0]:
#             imagem_casa = imagem[y-300:y+300, x-300:x+300]
#             if imagem_casa.shape[0] > 0 and imagem_casa.shape[1] > 0:
#                 cv2.imwrite(f'casa_{i+1}_{j+1}.png', imagem_casa)
