import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dataset_base = os.path.join(project_root, 'Dataset_BUSI_with_GT')
malignant_dir = os.path.join(dataset_base, 'malignant')
benign_dir = os.path.join(dataset_base, 'benign')

list_malignant_all = os.listdir(malignant_dir)
list_malignant_mask = [ele for ele in list_malignant_all if 'mask' in ele]
list_benign_all = os.listdir(benign_dir)
list_benign_mask = [ele for ele in list_benign_all if 'mask' in ele]

image = Image.open(os.path.join(benign_dir, list_benign_mask[1]))
tab = np.array(image)
print(np.unique(tab))