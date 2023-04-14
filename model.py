import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define a arquitetura da rede neural personalizada
class MinhaRedeNeural(nn.Module):
    def __init__(self, num_classes, num_outputs):
        super(MinhaRedeNeural, self).__init__()
        # Carrega o modelo MobileNetV2 pré-treinado
        self.modelo_base = models.mobilenet_v2(pretrained=True)
        # Remove a camada de classificação original (classifier)
        self.modelo_base = nn.Sequential(*list(self.modelo_base.children())[:-1])
        # Adiciona camadas de classificação personalizadas
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1280, 2050),
                nn.ReLU(inplace=True),
                nn.Linear(2050, 13),
                nn.Softmax(dim=1)
            ) for _ in range(num_outputs)
        ])

    def forward(self, x):
        features = self.modelo_base(x)
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return outputs

# Define o número de classes do novo conjunto de dados
num_classes = 10
# Define o número de saídas desejadas
num_outputs = 64

# Cria uma instância da rede neural personalizada
modelo = MinhaRedeNeural(num_classes, num_outputs)

# Define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# Imprime a arquitetura do modelo
print(modelo)


import cv2
import torch
from torchvision import transforms

# Carrega a imagem de entrada
imagem = cv2.imread('imagens/i0.png')  # Substitua 'imagem.jpg' pelo caminho da sua imagem
# Redimensiona a imagem para o tamanho desejado (224x224 pixels)
imagem = cv2.resize(imagem, (224, 224))

print(imagem)
cv2.imshow("Original",imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()
# Converte a imagem para o formato de tensor do PyTorch
# imagem_tensor = transforms.ToTensor()(imagem)
# # Adiciona uma dimensão extra para representar o batch size (1)
# imagem_tensor = imagem_tensor.unsqueeze(0)

# # Carrega o modelo
# modelo = MinhaRedeNeural(num_classes, num_outputs)
# # Carrega os pesos treinados do modelo (se aplicável)
# modelo.load_state_dict(torch.load('modelo.pth'))  # Substitua 'modelo.pth' pelo caminho dos seus pesos treinados
# # Coloca o modelo em modo de avaliação
# modelo.eval()

# # Passa a imagem pelo modelo
# with torch.no_grad():
#     saidas = modelo(imagem_tensor)

# # Obtém os resultados para cada uma das saídas
# for i, saida in enumerate(saidas):
#     classe_predita = torch.argmax(saida, dim=1).item()  # Obtém a classe com maior probabilidade
#     probabilidade = torch.max(saida, dim=1).values.item()  # Obtém a probabilidade da classe predita
#     print(f'Saída {i+1}: Classe: {classe_predita}, Probabilidade: {probabilidade:.2f}')
