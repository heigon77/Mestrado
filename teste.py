import cv2
import numpy as np

# Load the image
imagem = cv2.imread('i3.png', cv2.IMREAD_COLOR)

# Perform edge detection
bordas = cv2.Canny(imagem, 100, 150, apertureSize=3)

# Detect lines using Hough transform
linhas = cv2.HoughLines(bordas, 1, np.pi/180, 155)

# Group similar lines
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

max_rho1 = None
max_rho2 = None
min_rho1 = None
min_rho2 = None

for linha in linhas_agrupadas:
    rho, theta = linha[0]
    if theta > 1.2:

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

lines_squares = []

for linha in linhas_agrupadas:
    rho,theta = linha[0]

    if not (rho == max_rho2 or rho == max_rho1 or rho == min_rho1 or rho == min_rho2):
        lines_squares.append(linha)

# Extract line coordinates
line_points = []
for line in lines_squares:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    line_points.append([(x1, y1), (x2, y2)])

# Find intersection points of lines
corner_points = []
for i in range(len(line_points)):
    for j in range(i+1, len(line_points)):
        x1, y1 = line_points[i][0]
        x2, y2 = line_points[i][1]
        x3, y3 = line_points[j][0]
        x4, y4 = line_points[j][1]
        denominator = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denominator != 0:
            intersection_x = int(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / denominator)
            intersection_y = int(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / denominator)
            corner_points.append((intersection_x, intersection_y))

# Draw circles around the corners
for point in corner_points:
    x, y = point
    cv2.circle(imagem, (x, y), 5, (255, 0, 0), 2)

# Display the imagem with corners and circles
cv2.imshow('Corners and Circles', imagem)
cv2.imwrite('corners.png', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
