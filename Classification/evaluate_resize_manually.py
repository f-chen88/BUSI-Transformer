from train_resize_manually import test_data, test_label
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import MyDataset
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

my_transform = transforms.Compose([
    #transforms.CenterCrop((500, 500)) 
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
batch_size = 16

test_dataset = MyDataset(test_data, test_label, transform=my_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

for i in range(5): 
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    num_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(num_features, 2) 
    model.load_state_dict(torch.load(f'model_fold_{i+1}.pth'))
    model.to(device)
    model.eval()

    test_loss = 0 
    total = 0 
    correct = 0

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad() :

        for img, label in test_loader : 
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = criterion(output, label)
            test_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            probs = F.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())  
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy()) 

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total

    try:
        auc_score = roc_auc_score(all_labels, [p[1] for p in all_probs])  # Use prob of class 1
    except ValueError:
        auc_score = float('nan')  # If AUC fails (e.g., single class)

    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f'Model {i+1} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}%, AUC_Score: {auc_score:.2f}') 
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix for Model {i+1}')
    plt.savefig(f'confusion_matrix_model_{i+1}.png')
    plt.close() 

