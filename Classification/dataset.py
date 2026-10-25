from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from data_cropping import list_cropped_images, list_labels

  
class MyDataset(Dataset) : 
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self) : 
        return len(self.images)
    
    def __getitem__(self, idx) : 
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform : 
            img = self.transform(img)
        return img, label
    
my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1])
])

#dataset = MyDataset(list_cropped_images, list_labels, transform=my_transform)
#image, label = dataset[0]
#plt.imshow(image.numpy().transpose(1, 2, 0))
#plt.show(block=False)