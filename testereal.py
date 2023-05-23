import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2 as cv
from torchvision.transforms import ToTensor
import numpy as np
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)

piece_lookup = {
    0 : "P",
    1 : "N",
    2 : "B",
    3 : "R",
    4 : "Q",
    5 : "K",
    6 : "p",
    7 : "n",
    8 : "b",
    9 : "r",
    10 : "q",
    11 : "k",
}

value_lookup = {
    "P" : 0,
    "N" : 1,
    "B" : 2,
    "R" : 3,
    "Q" : 4,
    "K" : 5,
    "p" : 6,
    "n" : 7,
    "b" : 8,
    "r" : 9,
    "q" : 10,
    "k" : 11,
}

def y_to_fens(results):
    fens = []

    for i in range(results.shape[0]):
        fen = ''
        empty = 0

        for j in range(64):
            if np.argmax(results[i, j, :]) == 12:
                empty += 1
            else:
                if empty != 0:
                    fen += str(empty)
                    empty = 0
                piece_idx = np.argmax(results[i, j, :])
                fen += piece_lookup[piece_idx]

            if (j+1) % 8 == 0 and j < 63:
                if empty != 0:
                    fen += str(empty)
                    empty = 0
                fen += '/'

        if empty != 0:
            fen += str(empty)

        fen = fen.split('/')
        fen.reverse()
        fen = '/'.join(fen)

        fens.append(fen)
    
    return fens

def fens_to_y(fens):
    results = np.zeros((len(fens), 64, 13))

    for i, fen in enumerate(fens):

        fen = fen.split()[0]
        
        rows = fen.split('/')
        rows.reverse()

        col = 0
        for j, row in enumerate(rows):  
            for char in row:
                if char.isdigit():
                    aux = col + int(char)
                    results[i, col:aux, 12] = 1
                    col = aux
                else:
                    piece_idx = value_lookup[char]
                    results[i, col, piece_idx] = 1
                    col += 1

    return results


class MinhaRedeNeural(nn.Module):
    def __init__(self):
        super(MinhaRedeNeural, self).__init__()

        self.model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

        self.model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )

    def forward(self, x):
        aux = self.model.features(x)
        outputs = self.model.classifier(aux)
        # outputs = self.model(x)
        return outputs

class GamesDataset(Dataset):
    def __init__(self, images, fen):
        self.x = images
        self.y = torch.from_numpy(fen).type(torch.FloatTensor)
        self.n_samples = images.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

model = MinhaRedeNeural()
model.to(device)
model.load_state_dict(torch.load("modelo.pth"))
model.eval()

imagem = cv.imread(f"Dataset\Teste\CompressJPEG.online_1920x1080_image.jpg", cv.IMREAD_COLOR)

print(imagem.shape)

# cv.imshow('Original', imagem)
# cv.waitKey(0)
# cv.destroyAllWindows()

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
    # print(rho, theta)
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
    if theta > 1 and theta < 3:

        if max_rho1 == None:
            max_rho1 = rho
        elif rho > max_rho1:
            max_rho1 = rho
        
        if min_rho1 == None:
            min_rho1 = rho
        elif rho < min_rho1:
            min_rho1 = rho
    # else:

    #     if max_rho2 == None:
    #         max_rho2 = rho
    #     elif rho > max_rho2:
    #         max_rho2 = rho
        
    #     if min_rho2 == None:
    #         min_rho2 = rho
    #     elif rho < min_rho2:
    #         min_rho2 = rho

linhas_casas = []

for idx,linha in enumerate(linhas_agrupadas):
    rho,theta = linha[0]

    if idx not in [1,8,11,14,21,22,23,24,26,27,28]:
        linhas_casas.append(linha)

    # if not (rho == max_rho2 or rho == max_rho1 or rho == min_rho1 or rho == min_rho2):
    #     linhas_casas.append(linha)

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

cv.imshow('Hough', images_linhas)
cv.waitKey(0)
cv.destroyAllWindows()

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

fens = pd.read_csv('Dataset\img_fen.csv')

primeira_linha = fens.iloc[1]

fen_primeira_linha = [primeira_linha['FEN'].split()[0]]

print(fen_primeira_linha)

fen_expected = fens_to_y(fen_primeira_linha)

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

cv.imshow('Intersecções', imagem_pontos)
cv.waitKey(0)
cv.destroyAllWindows()

output = np.zeros(fen_expected.shape)

for i in range(64):

    if np.argmax(fen_expected[0, i, :]) != 12:

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

        if y_min - 40 < 0:
            y_min = 40

        imagem_recortada = imagem[y_min-40:y_max, x_min:x_max]

        cv.imshow('Img', imagem_recortada)
        cv.waitKey(0)
        cv.destroyAllWindows()

        imagem_path = "ImagemTeste.png"
        cv.imwrite(imagem_path, imagem_recortada)
        imagem_pil = Image.open(imagem_path)

        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        imagem_tensor = preprocess(imagem_pil)
        imagem_tensor_batch = imagem_tensor.unsqueeze(0)

        imagem_tensor_batch = imagem_tensor_batch.to(device)
        result = model(imagem_tensor_batch)

        result = result.cpu().detach().numpy().astype('float32')

        if np.argmax(fen_expected[0, i, :]) <= 5:
            ind = np.argmax(result[0,:])
            output[0,i,ind] = 1
        
        else:
            ind = np.argmax(result[0,:]) + 6
            output[0,i,ind] = 1

    else:
        output[0,i,12] = 1

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

        imagem_recortada = imagem[y_min:y_max, x_min:x_max]

        cv.imshow('Img', imagem_recortada)
        cv.waitKey(0)
        cv.destroyAllWindows()

        imagem_path = "ImagemTeste.png"
        cv.imwrite(imagem_path, imagem_recortada)
        imagem_pil = Image.open(imagem_path)

print("---Predição----")
print(y_to_fens(output))