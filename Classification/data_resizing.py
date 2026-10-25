from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

malignant_dir = os.path.join(project_root, 'Dataset_BUSI_with_GT', 'malignant')
benign_dir = os.path.join(project_root, 'Dataset_BUSI_with_GT', 'benign')

list_malignant_all = os.listdir(malignant_dir)
list_malignant = [ele for ele in list_malignant_all if 'mask' not in ele]
list_benign_all = os.listdir(benign_dir)
list_benign = [ele for ele in list_benign_all if 'mask' not in ele]


list_image = []
list_labels = []

for i in range(len(list_malignant)): 
    image = Image.open(os.path.join(malignant_dir, list_malignant[i]))
    width, height = image.size
    if width >= height : 
        transform = transforms.Resize((int(height*224/width), 224))
    else : 
        transform = transforms.Resize((224, int(width*224/height)))
    image = transform(image)

    width, height = image.size

    if width == 224 : 
        pad_bottom = (224 - height) // 2
        pad_top = 224 - height - pad_bottom
        pad_right = 0
        pad_left = 0 
    else : 
        pad_bottom = 0
        pad_top = 0 
        pad_right = (224 - width) // 2
        pad_left = 224 - width - pad_right

    image = transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom))

    list_image.append(image)
    list_labels.append(1)

for i in range(len(list_benign)): 
    image = Image.open(os.path.join(benign_dir, list_benign[i]))
    width, height = image.size
    if width >= height : 
        transform = transforms.Resize((int(height*224/width), 224))
    else : 
        transform = transforms.Resize((224, int(width*224/height)))
    image = transform(image)

    width, height = image.size

    if width == 224 : 
        pad_bottom = (224 - height) // 2
        pad_top = 224 - height - pad_bottom
        pad_right = 0
        pad_left = 0 
    else : 
        pad_bottom = 0
        pad_top = 0 
        pad_right = (224 - width) // 2
        pad_left = 224 - width - pad_right

    image = transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom))

    list_image.append(image)
    list_labels.append(0)

