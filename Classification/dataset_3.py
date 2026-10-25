import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dataset_path = os.path.join(project_root, 'Dataset_BUSI_with_GT')
dataset_3 = datasets.ImageFolder(dataset_path)

print("The length of the dataset is", len(dataset_3))
print("There are 3 classes : ", dataset_3.class_to_idx)

labels = [label for _, label in dataset_3]
count = [values for _, values in Counter(labels).items()]
plt.bar(dataset_3.classes, count)
plt.title("Data distribution")
plt.show(block=False)