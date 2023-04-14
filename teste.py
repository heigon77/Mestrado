import torch
import torch.nn as nn
import torch.optim as optim

# Dados de exemplo
num_classes = 3
input_size = 5
batch_size = 2

# Gerando dados de exemplo
inputs = torch.randn(batch_size, input_size)  # Entradas do modelo
labels = torch.empty(batch_size, dtype=torch.long).random_(num_classes)  # Rótulos verdadeiros

# Instanciando o modelo e a função de perda
model = nn.Linear(input_size, num_classes)  # Modelo de exemplo com uma camada linear
loss_fn = nn.CrossEntropyLoss()  # Função de perda CrossEntropyLoss

# Calculando a perda
outputs = model(inputs)  # Passando as entradas pelo modelo para obter as previsões
loss = loss_fn(outputs, labels)  # Calculando a perda com base nas previsões e rótulos verdadeiros

# Backpropagation e otimização
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Otimizador de exemplo (Gradiente Descendente Estocástico)
optimizer.zero_grad()  # Zerando os gradientes dos parâmetros do modelo
loss.backward()  # Backpropagation para calcular os gradientes
optimizer.step()  # Atualizando os parâmetros do modelo com base nos gradientes

print('Perda:', loss.item())
