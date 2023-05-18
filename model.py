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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)

piece_lookup = {
    0 : "K",
    1 : "Q",
    2 : "R",
    3 : "B",
    4 : "N",
    5 : "P",
    6 : "k",
    7 : "q",
    8 : "r",
    9 : "b",
    10 : "n",
    11 : "p",
    6 : "1",
}

value_lookup = {
    "K" : 0,
    "Q" : 1,
    "R" : 2,
    "B" : 3,
    "N" : 4,
    "P" : 5,
    "k" : 6,
    "q" : 7,
    "r" : 8,
    "b" : 9,
    "n" : 10,
    "p" : 11,
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

        fens.append(fen)
    
    return fens

def fens_to_y(fens):
    results = np.zeros((len(fens), 64, 13))

    for i, fen in enumerate(fens):

        fen = fen.split()[0]
        
        rows = fen.split('/')
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

imagens_array = torch.load("Dataset\Pecas\imagens_casa_tensor1000.pt")
fens = np.load("Dataset\Pecas\pecas_casa_tensor1000.npy")
for i in range(2000, 5001, 1000):
    imagens_array_load = torch.load(f"Dataset\Pecas\imagens_casa_tensor{i}.pt")
    fens_load = np.load(f"Dataset\Pecas\pecas_casa_tensor{i}.npy")

    imagens_array = torch.cat((imagens_array, imagens_array_load), dim=0)
    fens = np.concatenate((fens, fens_load), axis=0)


print(imagens_array.size())
print(fens.shape)

class_frequencies  = np.sum(fens, axis=0)

class_frequencies = torch.from_numpy(class_frequencies)


# weights = 1 / class_frequencies

weights = torch.tensor([0.1, 0.15, 0.15, 0.15, 0.25, 0.2])

print(class_frequencies)
print(weights)

datasetTrain = GamesDataset(imagens_array[:int(len(imagens_array)*.95)], fens[:int(len(fens)*.95)])

sampler = WeightedRandomSampler(weights, 1500, replacement = True )

dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=32)

datasetTest = GamesDataset(imagens_array[int(len(imagens_array)*.95):], fens[int(len(fens)*.95):])
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=32)

print(f"Train: {len(dataLoaderTrain.dataset)}")
print(f"Test: {len(dataLoaderTest.dataset)}")



model = MinhaRedeNeural()

# resnet.to(device)
model.to(device)

# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

num_epochs = 15

file_out = open('Dataset/outputs.csv','w')
file_out.write(f"Epoch,Loss\n")

model.train()
print("Start Training")
for epoch in range(num_epochs):

   
    total = 0
    right = 0

    classes_batch = np.zeros(6)

    for (pos, fen_y) in dataLoaderTrain:

        pos = pos.to(device)
        fen_y = fen_y.to(device)

        # inter = resnet(pos)
        fen_pred = model(pos)

        # print(fen_pred.size())
        # print(fen_y.size())

        loss = criterion(fen_pred, fen_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fen_pred = fen_pred.cpu().detach().numpy().astype('float32')
        fen_y = fen_y.cpu().detach().numpy().astype('float32')

        for i in range(len(fen_pred)):
            total += 1

            classes_batch = classes_batch + fen_y[i]

            if np.argmax(fen_pred[i, :]) == np.argmax(fen_y[i, :]):
                right += 1

    
    accuracy_train = right/total * 100
    
    print(f"Epoch: {epoch+1}, Loss Train: {loss.item()}, Acc Train: {accuracy_train:.2f}%, {classes_batch}")
    file_out.write(f"{epoch+1},{loss.item():.2f}\n")
file_out.close()
print("Finish Training")

torch.save(model.state_dict(), "modelo.pth")
    

model.eval()
with torch.no_grad():

    total = 0
    right = 0

    classes_batch = np.zeros(6)

    for (pos, fen_y) in dataLoaderTest:

        pos = pos.to(device)
        fen_y = fen_y.to(device)

        fen_pred = model(pos)

        loss_test = criterion(fen_pred, fen_y)

        fen_pred = fen_pred.cpu().detach().numpy().astype('float32')
        fen_y = fen_y.cpu().detach().numpy().astype('float32')

        for i in range(len(fen_pred)):
            total += 1

            classes_batch = classes_batch + fen_y[i]

            if np.argmax(fen_pred[i, :]) == np.argmax(fen_y[i, :]):
                right += 1

    accuracy_test = right/total * 100

    print(f"Loss Test: {loss_test.item():.2f}, Acc Test: {accuracy_test:.2f}%")

# probabilities = np.array([prob.numpy() for prob in probabilities])
# probabilities = probabilities.reshape((64,13))
# print(probabilities.shape)
# print(probabilities[0])


# for i, prob in enumerate(probabilities):
#     print(f"Saída {i+1}:")
#     for j, class_prob in enumerate(prob[0]):
#         print(f"Classe {j+1}: {class_prob:.4f}")
