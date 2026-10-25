from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm 
import time
from segmentation_dataset import SegmentationDataset, images_to_masks, list_all_image_path, labels
import segmentation_models_pytorch as smp
from segmentation_train import test_image
from random import randint
from torchvision.transforms import ToPILImage

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # No normalization for masks
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset = SegmentationDataset(test_image, images_to_masks, image_transform=image_transform, mask_transform=mask_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = smp.Unet(
    encoder_name="resnet34",    
    encoder_weights="imagenet",   
    in_channels=3,                 
    classes=1                    
)
#model.load_state_dict(torch.load("segmentation_model_100_epo.pth"))
model.load_state_dict(torch.load("model_path/segmentation_model_100_epo.pth"))
model.to(device)
model.eval()

correct_pixels = 0
total_pixels = 0
score = []

with torch.no_grad():
    for img, mask, img_path in test_loader:
        img = img.to(device)
        mask = mask.to(device)
        mask_binary = (mask > 1/256).float()

        output = model(img)
        probs = torch.sigmoid(output)  
        preds = (probs > 0.005).float()  # binarize predictions

        Hadamar_prod = mask_binary * preds
        intersection = torch.sum(Hadamar_prod)
        print(intersection)
        print("mask_binary 1s:", torch.sum(mask_binary).item())
        print("preds 1s:", torch.sum(preds).item())
        eps = 1e-8
        dice = (2*intersection+eps) / (torch.sum(preds)+torch.sum(mask_binary)+eps)
        score.append(dice.item())


        correct_pixels += (preds == mask_binary).sum().item()
        total_pixels += torch.numel(mask_binary)

        print("Probabilities stats:", probs.min().item(), probs.max().item(), probs.mean().item())
        print("Predicted mask unique values:", torch.unique(preds))
        print("Predicted mask sample:", preds[0, 0, :5, :5])  # first batch, first channel, first 5x5 pixels


accuracy = correct_pixels / total_pixels

print("The accuracy is", accuracy)

print("The Dice is:", np.mean(score))

to_pil = ToPILImage()

test_list = list(test_loader)  # convert DataLoader to list to index

fig = plt.figure(figsize=(10, 15))

lst = []

for i in range(1, 16):
    
    ax = plt.subplot(5, 3, i)
    
    if i in {1, 4, 7, 10, 13}:
        rdn = randint(0, len(test_list) - 1)
        while rdn in lst : 
            rdn = randint(0, len(test_list) - 1)
        lst.append(rdn)

        batch_img, batch_mask, batch_path = test_list[rdn]
        
        img_pil = to_pil(batch_img[0].cpu())
        plt.imshow(img_pil, cmap='gray')
        plt.title(f"{os.path.splitext(os.path.basename(batch_path[0]) )[0]}", fontsize=8)
    
    elif i in {2, 5, 8, 11, 14}:
        mask_pil = to_pil(batch_mask[0].cpu())
        plt.imshow(mask_pil, cmap='gray')
        s = os.path.splitext(os.path.basename(batch_path[0]))[0]
        start = s.find('(')    # index of '('
        end = s.find(')')      # index of ')'
        number = s[start+1:end]
        plt.title(f"True Mask ({number})", fontsize=8)
    
    else:
        img_input = batch_img[0].unsqueeze(0).to(device)  # add batch dim
        output = model(img_input)
        probs = torch.sigmoid(output)
        preds = (probs > 0.005).float().squeeze().cpu()
        mask_image = Image.fromarray((preds.numpy() * 255).astype('uint8'))
        plt.imshow(mask_image, cmap='gray')
        s = os.path.splitext(os.path.basename(batch_path[0]))[0]
        start = s.find('(')    # index of '('
        end = s.find(')')      # index of ')'
        number = s[start+1:end]
        plt.title(f"Generated Mask ({number})", fontsize=8)
    
    plt.axis('off')

fig.suptitle("Data visualization", fontsize=20)
plt.savefig(f'Segmentation_visualization.png')
plt.close()
