from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class Dataset_2(Dataset) : 
    def __init__(self, samples, transform=None) : 
        self.samples = samples
        self.transform = transform

    def __len__(self) : 
        return len(self.samples)
    
    def __getitem__(self, index):
        path , label = self.samples[index]
        image = Image.open(path)
        if self.transform : 
            image = self.transform(image)
        return (image, label)
    
my_transform = transforms.Compose([

    #transforms.CenterCrop((500, 500))
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),            
    transforms.Normalize(mean=[0], std=[1]) 
])


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dataset_path = os.path.join(project_root, 'Dataset_BUSI_with_GT')
dataset_3 = datasets.ImageFolder(dataset_path)

samples = [ele for ele in dataset_3.samples if ele[1] in [0, 1]]

dataset_2 = Dataset_2(samples, transform=my_transform)