import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import cv2
import torch
from torchvision.transforms import ToTensor
import numpy as np
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

class MinhaRedeNeural(nn.Module):
    def __init__(self):
        super(MinhaRedeNeural, self).__init__()
        
        self.modelo_base = models.mobilenet_v2(pretrained=True)
        
        self.modelo_base = nn.Sequential(*list(self.modelo_base.children())[:-1])
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(62720, 13),
                nn.Softmax(dim=1)
            ) for _ in range(64)
        ])

    def forward(self, x):
        features = self.modelo_base(x)
        print(features.size())
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return outputs

class GamesDataset(Dataset):
    def __init__(self, images, fen):
        self.x = torch.from_numpy(images).type(torch.FloatTensor)
        self.y = torch.from_numpy(fen).type(torch.FloatTensor)
        self.n_samples = images.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    

# image_path = "exemplo.jpeg"
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (224, 224))

# image_tensor = ToTensor()(image)
# image_tensor = torch.unsqueeze(image_tensor, dim=0)

# datasetTrain = GamesDataset(games[:int(len(games)*.98)])
# dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=64)

# datasetTest = GamesDataset(games[int(len(games)*.98):])
# dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=64)

print(f"Train: {len(dataLoaderTrain.dataset)}")
print(f"Test: {len(dataLoaderTest.dataset)}")

model = MinhaRedeNeural()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.001)

num_epochs = 200
model.train()

file_out = open('Data/outputsDBNlichessnormal.csv','w')
file_out.write(f"Epoch,Loss\n")


print("Start Training")
for epoch in range(num_epochs):

    for (pos,_) in dataLoaderTrain:

        pos = pos.to(device)

        fen = model(pos)
        loss = criterion(recon, pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch+1}, Loss: {loss.item():4f}")
    file_out.write(f"{epoch+1},{loss.item():4f}\n")
file_out.close()
print("Finish Training")
    


with torch.no_grad():
    outputs = modelo(image_tensor)
    
    probabilities = [torch.softmax(output, dim=1) for output in outputs]


probabilities = [prob.numpy() for prob in probabilities]


for i, prob in enumerate(probabilities):
    print(f"Saída {i+1}:")
    for j, class_prob in enumerate(prob[0]):
        print(f"Classe {j+1}: {class_prob:.4f}")
