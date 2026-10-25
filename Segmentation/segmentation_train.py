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

train_image, test_image, train_label, test_label= train_test_split(list_all_image_path, labels, test_size=0.3, stratify=labels)

val_image, test_image, val_label, test_label = train_test_split(test_image, test_label, test_size=1/3, stratify=test_label)

train_dataset = SegmentationDataset(train_image, images_to_masks, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = SegmentationDataset(val_image, images_to_masks, image_transform=image_transform, mask_transform=mask_transform)
test_dataset = SegmentationDataset(test_image, images_to_masks, image_transform=image_transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


model = smp.Unet(
    encoder_name="resnet34",     
    encoder_weights="imagenet",  
    in_channels=3,              
    classes=1                     
)

if __name__ == "__main__":

    num_epochs = 100
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    val_accuracy = []

    for epoch in range(num_epochs):
            start_epo = time.time()
            model.train()
            train_loss = 0.0
            

            pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=False)
            for img, mask, img_path in pbar_train : 
                
                img = img.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                output = model(img)
                loss = criterion(output, mask)

                loss.backward()
                optimizer.step()
                

                train_loss += loss.item()

                pbar_train.set_postfix({'loss': f"{loss:.4f}"})

                
            
            model.eval()
            val_loss = 0 
            total = 0 
            correct = 0
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val", leave=False)
            with torch.no_grad():
                for img, mask, img_path in pbar_val: 
                
                    img = img.to(device)
                    mask = mask.to(device)

                    output = model(img)
                    loss = criterion(output, mask)
                    val_loss += loss.item()

                    predicted = torch.sigmoid(output) > 0.5
                    correct += (predicted == mask).sum().item()
                    total += mask.numel()

                    pbar_val.set_postfix({'loss': f'{loss:.4f}'})
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total

            

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracy.append(val_acc)

            end_epo = time.time()
            time_epo = end_epo - start_epo
            formatted_time_epo = time.strftime("%H:%M:%S", time.gmtime(time_epo))
            print(f"Epoch {epoch + 1} Train loss: {train_loss:.4f} Val loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Time: {formatted_time_epo}")

    torch.save(model.state_dict(), 'segmentation_model.pth')
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(val_accuracy, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('segmentation_loss_acc_curve.png')
    plt.close()
