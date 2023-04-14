from PIL import Image
import torch
from torchvision import transforms

imagens_processadas = []

for i in range(10200):
    
    if i % 100 == 0:
        print(i)

    imagem_path = f"imagens/i{i}.png"
    imagem = Image.open(imagem_path)

    if imagem.mode != 'RGB':
        imagem = imagem.convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imagem_tensor = preprocess(imagem)
    
    imagens_processadas.append(imagem_tensor)
    
imagens_tensor = torch.stack(imagens_processadas)

print("Tamanho do tensor de imagens processadas:", imagens_tensor.size())
torch.save(imagens_tensor, "Dataset/imagens_tensor.pt")

