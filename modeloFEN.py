import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2 as cv
import torch
from torchvision.transforms import ToTensor
import numpy as np
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
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
    12 : "1",
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

        self.custom_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(1280, 2050),
            nn.ReLU(inplace=True)
            )

        self.model.classifier = nn.ModuleList([nn.Sequential(
            nn.Linear(2050, 13),
            nn.Softmax(dim=1)
        ) for _ in range(64)])

    def forward(self, x):
        outputs = []
        aux = self.model.features(x)
        aux = self.custom_layer(aux)

        for i in range(64):
            out = self.model.classifier[i](aux)
            outputs.append(out)
        outputs = torch.stack(outputs)
        outputs = torch.transpose(outputs, 0, 1)
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

imagens_array = torch.load("Dataset/imagens_tensor.pt")
fens = np.load("Dataset/fens.npy")

# dt = pd.read_csv("Dataset\img_fen.csv")
# fens = dt['FEN'].values
# print(fens.shape)
# res = fens_to_y(fens)
# np.save("Dataset/fens.npy", res)


# cv.imshow("Original",cv.cvtColor(imagens_array[0], cv.COLOR_RGB2BGR))

# cv.waitKey(0)
# cv.destroyAllWindows()



datasetTrain = GamesDataset(imagens_array[:int(len(imagens_array)*.98)], fens[:int(len(fens)*.98)])
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=32)

datasetTest = GamesDataset(imagens_array[int(len(imagens_array)*.98):], fens[int(len(fens)*.98):])
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=32)

print(f"Train: {len(dataLoaderTrain.dataset)}")
print(f"Test: {len(dataLoaderTest.dataset)}")

model = MinhaRedeNeural()
model.to(device)

# print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

file_out = open('Dataset/outputs.csv','w')
file_out.write(f"Epoch,Loss\n")

model.train()
print("Start Training")
for epoch in range(num_epochs):

   
    total = 0
    right = 0
    total_empty = 0
    right_empty = 0
    total_non_empty = 0
    right_non_empty = 0
    whole_pos = 0
    whole_pos_right = 0
    for (pos, fen_y) in dataLoaderTrain:

        pos = pos.to(device)
        fen_y = fen_y.to(device)

        fen_pred = model(pos)

        loss = criterion(fen_pred, fen_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fen_pred = fen_pred.cpu().detach().numpy().astype('float32')
        fen_y = fen_y.cpu().detach().numpy().astype('float32')

        for i in range(len(fen_pred)):
            whole_pos += 1
            all_right = True
            for j in range(len(fen_pred[i])):
                total += 1

                if np.argmax(fen_pred[i, j, :]) == np.argmax(fen_y[i, j, :]):
                    right += 1
                
                else:
                    all_right = False
                
                if np.argmax(fen_y[i, j, :]) == 12:
                    total_empty += 1
                    if np.argmax(fen_pred[i, j, :]) == 12:
                        right_empty += 1
                
                else:
                    total_non_empty += 1
                    if np.argmax(fen_pred[i, j, :]) != 12:
                        right_non_empty += 1 

            if (all_right):
                whole_pos_right +=1
    
    accuracy_train = right/total * 100
    accuracy_empty = right_empty/total_empty * 100
    accuracy_non_empty = right_non_empty/total_non_empty * 100
    accuracy_whole_pos = whole_pos_right/whole_pos * 100
    
    print(f"Epoch: {epoch+1}, Loss Train: {loss.item():.2f}, Acc Train: {accuracy_train:.2f}%, Acc vazios:{accuracy_empty:.2f}%, Acc não vazios:{accuracy_non_empty:.2f}%, Posição inteira: {accuracy_whole_pos:.2f}%")
    file_out.write(f"{epoch+1},{loss.item():.2f}\n")
file_out.close()
print("Finish Training")

torch.save(model.state_dict(), "modelo.pth")
    
# image_path = "imagens\i0.png"
# image = cv.imread(image_path)
# image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# image = cv.resize(image, (224, 224))
# image_tensor = ToTensor()(image)
# image_tensor = torch.unsqueeze(image_tensor, dim=0)
model.eval()
with torch.no_grad():

    total = 0
    right = 0
    total_empty = 0
    right_empty = 0
    total_non_empty = 0
    right_non_empty = 0
    whole_pos = 0
    whole_pos_right = 0
    for (pos, fen_y) in dataLoaderTest:

        pos = pos.to(device)
        fen_y = fen_y.to(device)

        fen_pred = model(pos)

        loss_test = criterion(fen_pred, fen_y)

        fen_pred = fen_pred.cpu().detach().numpy().astype('float32')
        fen_y = fen_y.cpu().detach().numpy().astype('float32')

        for i in range(len(fen_pred)):
            whole_pos += 1
            all_right = True
            for j in range(len(fen_pred[i])):
                total += 1

                if np.argmax(fen_pred[i, j, :]) == np.argmax(fen_y[i, j, :]):
                    right += 1
                
                else:
                    all_right = False
                
                if np.argmax(fen_y[i, j, :]) == 12:
                    total_empty += 1
                    if np.argmax(fen_pred[i, j, :]) == 12:
                        right_empty += 1
                
                else:
                    total_non_empty += 1
                    if np.argmax(fen_pred[i, j, :]) != 12:
                        right_non_empty += 1 

    accuracy_test = right/total * 100
    accuracy_empty = right_empty/total_empty * 100
    accuracy_non_empty = right_non_empty/total_non_empty * 100
    accuracy_whole_pos = whole_pos_right/whole_pos * 100

    print(f"Loss Test: {loss_test.item():.2f}, Acc Test: {accuracy_test:.2f}%, Acc vazios:{accuracy_empty:.2f}%, Acc não vazios:{accuracy_non_empty:.2f}%, Posição inteira: {accuracy_whole_pos:.2f}%")

# probabilities = np.array([prob.numpy() for prob in probabilities])
# probabilities = probabilities.reshape((64,13))
# print(probabilities.shape)
# print(probabilities[0])


# for i, prob in enumerate(probabilities):
#     print(f"Saída {i+1}:")
#     for j, class_prob in enumerate(prob[0]):
#         print(f"Classe {j+1}: {class_prob:.4f}")
