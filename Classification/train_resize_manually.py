from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import MyDataset
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision.models import vit_b_16, ViT_B_16_Weights
from dataset_2 import Dataset_2
from tqdm import tqdm 
import time
from data_resizing import list_image, list_labels



my_transform = transforms.Compose([
    #transforms.CenterCrop((500, 500))
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


images = list_image
labels = np.array(list_labels)


train_data, test_data, train_label, test_label = train_test_split(images, labels, test_size=0.1, stratify=labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

if __name__ == "__main__":

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_label)), train_label)):
        print(f"\nStarting Fold {fold+1}")
        start_fold = time.time()
        images_train = [train_data[i] for i in train_idx]
        images_val = [train_data[i] for i in val_idx]
        labels_train = train_label[train_idx]
        labels_val = train_label[val_idx]

        batch_size = 16

        train_dataset = MyDataset(images_train, labels_train, transform=my_transform)
        val_dataset = MyDataset(images_val, labels_val, transform=my_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)

        num_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_features, 2)
        
        num_epochs = 10
        lr = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)


        train_losses = []
        val_losses = []
        val_accuracy = []

        for epoch in range(num_epochs):
            start_epo = time.time()
            model.train()
            train_loss = 0.0
            

            pbar_train = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} Train", leave=False)
            for img, label in pbar_train : 
                
                img = img.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                output = model(img)
                loss = criterion(output, label)

                loss.backward()
                optimizer.step()
                

                train_loss += loss.item()

                pbar_train.set_postfix({'loss': f"{loss:.4f}"})

                
            
            model.eval()
            val_loss = 0 
            total = 0 
            correct = 0
            pbar_val = tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} Val", leave=False)
            with torch.no_grad():
                for img, label in pbar_val: 
                
                    img = img.to(device)
                    label = label.to(device)

                    output = model(img)
                    loss = criterion(output, label)
                    val_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

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

        end_fold = time.time()
        time_fold = end_fold - start_fold
        formatted_time_fold = time.strftime("%H:%M:%S", time.gmtime(time_fold))
        print(f"Fold {fold + 1} Time: {formatted_time_fold}")
        torch.save(model.state_dict(), f'model_fold_{fold + 1}.pth')
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Loss Evolution')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(val_accuracy, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Fold {fold + 1} Accuracy Evolution')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'Fold {fold + 1}_loss_acc_curve.png')
        plt.close()