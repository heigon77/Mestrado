from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

pieces_id = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

imagens_processadas = []
classes = []

df = pd.read_csv('Dataset\img_piece_square.csv')
df = df.drop_duplicates()

validPieces = 0

for index, row in df.iterrows():

    try:
        piece = row['Piece']

        if piece != '0':
            validPieces += 1

            img_name = row['Image']

            imagem_path = f"RecReais/{img_name}.png"
            imagem = Image.open(imagem_path)

            if imagem.mode != 'RGB':
                imagem = imagem.convert('RGB')

            image_array = np.array(imagem)
            
            preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            imagem_tensor = preprocess(imagem)
            imagens_processadas.append(imagem_tensor)

            classe = np.zeros(12, dtype=np.int32)
            classe[pieces_id[piece]] = 1

            classes.append(classe)
    
    except:
        print(f"Errr {img_name}")

print(validPieces)

imagens_tensor = torch.stack(imagens_processadas)
classes = np.array(classes, dtype=np.int32)

torch.save(imagens_tensor, f"Dataset/Pecas/imagens_casa_tensor_real.pt")
np.save(f"Dataset/Pecas/pecas_casa_tensor_real.npy", classes)