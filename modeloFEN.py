import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import torch
from torchvision.transforms import ToTensor
import numpy as np

class MinhaRedeNeural(nn.Module):
    def __init__(self, num_outputs):
        super(MinhaRedeNeural, self).__init__()
        # Carrega o modelo MobileNetV2 pré-treinado
        self.modelo_base = models.mobilenet_v2(pretrained=True)
        # Remove a última camada de classificação original (classifier)
        self.modelo_base = nn.Sequential(*list(self.modelo_base.children())[:-1])
        # Adiciona camadas de classificação personalizadas
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(62720, 13),
                nn.Softmax(dim=1)
            ) for _ in range(num_outputs)
        ])

    def forward(self, x):
        features = self.modelo_base(x)
        print(features.size())
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return outputs

# Define o número de saídas desejadas
num_outputs = 64

# Cria uma instância da rede neural personalizada
modelo = MinhaRedeNeural(num_outputs)

# Define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# Imprime a arquitetura do modelo
print(modelo)

# Define a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

# Carrega uma imagem de exemplo
image_path = "exemplo.jpeg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))  # Redimensiona a imagem para o tamanho de entrada do modelo

# Converte a imagem para tensor e adiciona uma dimensão de lote (batch)
image_tensor = ToTensor()(image)
image_tensor = torch.unsqueeze(image_tensor, dim=0)

# Cria uma instância do modelo
num_classes = 10
num_outputs = 64
modelo = MinhaRedeNeural(num_outputs)

# Carrega os pesos treinados do modelo
# checkpoint = torch.load('caminho/do/seu/checkpoint.pth') # substitua pelo caminho do seu arquivo de checkpoint
# modelo.load_state_dict(checkpoint['modelo_state_dict'])
modelo.eval()  # Define o modelo no modo de avaliação (não treinamento)

print(image_tensor.shape)
# Passa a imagem pelo modelo para obter as previsões
with torch.no_grad():
    outputs = modelo(image_tensor)
    # Converte as previsões em probabilidades usando a função de ativação Softmax
    probabilities = [torch.softmax(output, dim=1) for output in outputs]

# Converte as probabilidades em numpy array
probabilities = [prob.numpy() for prob in probabilities]

# Exibe as previsões
for i, prob in enumerate(probabilities):
    print(f"Saída {i+1}:")
    for j, class_prob in enumerate(prob[0]):
        print(f"Classe {j+1}: {class_prob:.4f}")
