import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dataset_base = os.path.join(project_root, 'Dataset_BUSI_with_GT')
malignant_dir = os.path.join(dataset_base, 'malignant')
benign_dir = os.path.join(dataset_base, 'benign')

list_malignant_all = os.listdir(malignant_dir)
list_malignant_mask = [ele for ele in list_malignant_all if 'mask' in ele]
list_malignant_image = [ele for ele in list_malignant_all if 'mask' not in ele]
list_benign_all = os.listdir(benign_dir)
list_benign_mask = [ele for ele in list_benign_all if 'mask' in ele]
list_benign_image = [ele for ele in list_benign_all if 'mask' not in ele]

list_malignant_image_path = [os.path.join(malignant_dir, ele) for ele in list_malignant_image]
list_benign_image_path = [os.path.join(benign_dir, ele) for ele in list_benign_image]
list_all_image_path = list_benign_image_path + list_malignant_image_path

images_to_masks = {}
labels = []
for list_image in list_all_image_path :
    base_name = os.path.splitext(os.path.basename(list_image))[0]
    matching_mask = [ele for ele in list_benign_mask + list_malignant_mask if base_name in ele]
    if 'malignant' in base_name: 
        list_masks = [os.path.join(malignant_dir, m) for m in matching_mask]
        labels.append(1)
    if 'benign' in base_name:
        list_masks = [os.path.join(benign_dir, m) for m in matching_mask]
        labels.append(0)
    images_to_masks[list_image] = list_masks


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, dic_img_mask, image_transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.dict = dic_img_mask
        self.image_transform = image_transform
        self.mask_transform = mask_transform


    def __len__(self) : 
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        img = Image.open(self.image_paths[index])
        list_of_masks = self.dict[self.image_paths[index]]

        mask_arrays = [np.array(Image.open(p).convert("L")) for p in list_of_masks]
        stacked_masks = np.stack(mask_arrays, axis=0)
        mask = np.any(stacked_masks, axis=0).astype(np.uint8)

        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        
        
        return img, mask, image_path

