from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

indx_P = []
indx_p = []
indx_0 = []
all_ind = []

df_original = pd.read_csv('Dataset\img_piece_square.csv')

selected_indices = []
num = 0
at_img = 0
for index, row in df_original.iterrows():

    img = row['Image']
    img = int(img.split('_')[0][1:])

    
    
   
    if at_img != img:
        
        print(img, num, index)

        indices =  [num for num in range(index-num, index) if num not in all_ind]
        
        if len(indx_P) >= 1:
            indices.extend(random.sample(indx_P, 1))
        else:
            indices.extend(indx_P)
        
        if len(indx_p) >= 1:
            indices.extend(random.sample(indx_p, 1))
        else:
            indices.extend(indx_p)

        indices.sort()

        selected_indices.extend(indices)

        indx_P = []
        indx_p = []
        indx_0 = []
        all_ind = []

        num = 0 

    num += 1
    at_img = img

    if row['Piece'] == 'P':
        indx_P.append(index)
        all_ind.append(index)
    
    elif row['Piece'] == 'p':
        indx_p.append(index)
        all_ind.append(index)

df_selected = df_original[df_original.index.isin(selected_indices)]

df_selected.to_csv('Dataset\img_piece_square_balanced.csv', index=False)
