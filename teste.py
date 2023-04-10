import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

# Definir a arquitetura da rede neural
class MinhaRedeNeural(nn.Module):
    def __init__(self, num_classes):
        super(MinhaRedeNeural, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carregar e pré-processar a imagem de entrada
imagem = cv2.imread('imagem.jpg')  # Substitua 'imagem.jpg' pelo caminho da sua imagem
imagem = cv2.resize(imagem, (224, 224))
imagem = np.transpose(imagem, (2, 0, 1))
imagem = torch.from_numpy(imagem).float()
imagem = imagem.unsqueeze(0)

# Criar a instância do modelo
num_classes = 10  # Substitua pelo número de classes da sua tarefa de classificação
modelo = MinhaRedeNeural(num_classes)

# Carregar os pesos treinados do modelo (se aplicável)
modelo.load_state_dict(torch.load('modelo.pth'))  # Substitua 'modelo.pth' pelo caminho dos seus pesos treinados

# Colocar o modelo em modo de avaliação
modelo.eval()

# Passar a imagem pelo modelo
with torch.no_grad():
    saida = modelo(imagem)
    predicao = torch.argmax(saida, dim=1).item()

# Imprimir a classe predita
print(f'Classe predita: {predicao}')
