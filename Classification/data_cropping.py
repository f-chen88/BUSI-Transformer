from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) 

list_malignant_all = os.listdir(os.path.join(project_root, 'Dataset_BUSI_with_GT', 'malignant'))
list_malignant_mask = [ele for ele in list_malignant_all if 'mask' in ele]
list_benign_all = os.listdir(os.path.join(project_root, 'Dataset_BUSI_with_GT', 'benign'))
list_benign_mask = [ele for ele in list_benign_all if 'mask' in ele]

for i in range (len(list_malignant_mask)): 
    list_malignant_mask[i] = os.path.join(project_root, 'Dataset_BUSI_with_GT', 'malignant', list_malignant_mask[i])
for j in range (len(list_benign_mask)):
    list_benign_mask[j] = os.path.join(project_root, 'Dataset_BUSI_with_GT', 'benign', list_benign_mask[j])
    
list_mask = list_malignant_mask + list_benign_mask

list_mask_double_2 = [path for path in list_mask if 'mask_2' in path]
list_mask_double_1_2 = [path[:-5] + '1' + path[-4:] for path in list_mask_double_2]
list_mask_double_0_2 = [path[:-6] + path[-4:] for path in list_mask_double_1_2]
list_mask = [path for path in list_mask if path not in list_mask_double_0_2]

list_mask_double_1 = [path for path in list_mask if 'mask_1' in path]
list_mask_double_0_1 = [path[:-6] + path[-4:] for path in list_mask_double_1]
list_mask = [path for path in list_mask if path not in list_mask_double_0_1]

max_width = 0 
max_height = 0 
max_height_image = 0 
max_width_image = 0 

path_max_height = 0
path_max_width = 0

for path in list_mask : 
    if 'mask_1' in path: 
        mask1 = Image.open(path)
        mask2 = Image.open(path[:-6] + path[-4:])
        img = Image.open(path[:-11] + path[-4:])
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        if mask1.ndim == 3 and mask1.shape[2] == 4: 
            mask1 = mask1[..., 0] > 0
        if mask2.ndim == 3 and mask2.shape[2] == 4: 
            mask2 = mask2[..., 0] > 0
        mask_np = np.logical_or(mask1, mask2)
        rows = np.any(mask_np, axis=1)
        columns = np.any(mask_np, axis=0)

    elif 'mask_2' in path : 
        mask2 = Image.open(path)
        mask3 = Image.open(path[:-5] + '1' + path[-4:])
        mask4 = Image.open(path[:-6] + path[-4:])
        image_width, image_height = mask2.size
        img = Image.open(path[:-11] + path[-4:])

        mask2 = np.array(mask2)
        mask3 = np.array(mask3)
        mask4 = np.array(mask4)

        if mask2.ndim == 3 and mask2.shape[2] == 4: 
            mask2 = mask2[..., 0] > 0
        if mask3.ndim == 3 and mask3.shape[2] == 4: 
            mask3 = mask3[..., 0] > 0
        if mask4.ndim == 3 and mask4.shape[2] == 4: 
            mask4 = mask4[..., 0] > 0
        mask_np1 = np.logical_or(mask2, mask3)
        mask_np2 = np.logical_or(mask_np1, mask4)
        rows = np.any(mask_np2, axis=1)
        columns = np.any(mask_np2, axis=0)

    
    else : 
        mask_pil_image = Image.open(path)
        mask_np = np.array(mask_pil_image) 
        img = Image.open(path[:-9] + path[-4:])
        if mask_np.ndim == 3 and mask_np.shape[2] == 4: 
            binary_mask = mask_np[..., 0] > 0
            rows = np.any(binary_mask, axis=1)  
            columns = np.any(binary_mask, axis=0)

        else : 
            rows = np.any(mask_np, axis=1)
            columns = np.any(mask_np, axis=0)
            
    true_indices_rows = np.where(rows)[0]
    true_indices_columns = np.where(columns)[0]
    top = true_indices_rows[0]
    bottom = true_indices_rows[-1]
    left = true_indices_columns[0]
    right = true_indices_columns[-1]
    height = bottom - top + 1
    width = right - left + 1
    if height > max_height : 
        max_height = height
        path_max_height = path
    if width > max_width : 
        max_width = width
        path_max_width = path

    width_image, height_image = mask_pil_image.size
    if height_image > max_height_image : 
        max_height_image = height_image
    if width_image > max_width_image : 
        max_width_image = width_image

list_cropped_images = []
list_labels = []
        
for path in list_mask : 
    if '/malignant/' in path:
        list_labels.append(1)
    elif '/benign/' in path:
        list_labels.append(0)

    if 'mask_1' in path: 
        mask1 = Image.open(path)
        mask2 = Image.open(path[:-6] + path[-4:])
        img = Image.open(path[:-11] + path[-4:])
        image_width, image_height = mask1.size
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        if mask1.ndim == 3 and mask1.shape[2] == 4: 
            mask1 = mask1[..., 0] > 0
        if mask2.ndim == 3 and mask2.shape[2] == 4: 
            mask2 = mask2[..., 0] > 0
        mask_np = np.logical_or(mask1, mask2)
        rows = np.any(mask_np, axis=1)
        columns = np.any(mask_np, axis=0)

    elif 'mask_2' in path : 
        mask2 = Image.open(path)
        mask3 = Image.open(path[:-5] + '1' + path[-4:])
        mask4 = Image.open(path[:-6] + path[-4:])
        image_width, image_height = mask2.size
        img = Image.open(path[:-11] + path[-4:])

        mask2 = np.array(mask2)
        mask3 = np.array(mask3)
        mask4 = np.array(mask4)

        if mask2.ndim == 3 and mask2.shape[2] == 4: 
            mask2 = mask2[..., 0] > 0
        if mask3.ndim == 3 and mask3.shape[2] == 4: 
            mask3 = mask3[..., 0] > 0
        if mask4.ndim == 3 and mask4.shape[2] == 4: 
            mask4 = mask4[..., 0] > 0
        mask_np1 = np.logical_or(mask2, mask3)
        mask_np2 = np.logical_or(mask_np1, mask4)
        rows = np.any(mask_np2, axis=1)
        columns = np.any(mask_np2, axis=0)

    else : 
        mask_pil_image = Image.open(path)
        img = Image.open(path[:-9] + path[-4:])
        image_width, image_height = mask_pil_image.size
        mask_np = np.array(mask_pil_image) 

        if mask_np.ndim == 3 and mask_np.shape[2] == 4: 
            binary_mask = mask_np[..., 0] > 0
            rows = np.any(binary_mask, axis=1)  
            columns = np.any(binary_mask, axis=0)

        else : 
            rows = np.any(mask_np, axis=1)
            columns = np.any(mask_np, axis=0)

    
    true_indices_rows = np.where(rows)[0]
    true_indices_columns = np.where(columns)[0]

    top = true_indices_rows[0]
    bottom = true_indices_rows[-1]
    left = true_indices_columns[0]
    right = true_indices_columns[-1]

    height = bottom - top + 1
    width = right - left + 1

    center_y = (top + bottom) // 2
    center_x = (left + right) // 2

    new_top = center_y - max_height // 2
    new_bottom = new_top + max_height 
    new_left = center_x - max_width // 2
    new_right = new_left + max_width

    if new_top < 0 : 
        new_top = 0
        new_bottom = max_height
    if new_bottom > image_height : 
        new_bottom = image_height
        new_top = image_height - max_height
    if new_left < 0 : 
        new_left = 0 
        new_right = max_width
    if new_right > image_width : 
        new_right = image_width
        new_left = image_width - max_width

    new_top = max(0, new_top)
    new_left = max(0, new_left) 

    cropped_image = transforms.functional.crop(img, new_top, new_left, new_bottom - new_top, new_right - new_left)

    pad_left = 0 
    pad_top = 0
    pad_right = max_width - (new_right - new_left)
    pad_bottom = max_height - (new_bottom - new_top)

    padded_image = transforms.functional.pad(cropped_image, (pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')
    list_cropped_images.append(padded_image)

#print(max_width, max_height)
#print(max_width_image, max_height_image)

#for img in list_cropped_images : 
#    print(img.size)

#plt.imshow(list_cropped_images[0])
#plt.show()    